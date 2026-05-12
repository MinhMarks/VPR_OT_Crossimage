# Giải thích codebase thư mục `deployment/`

## Tổng quan

Thư mục `deployment/` chứa **toàn bộ công cụ và cấu hình cần thiết** để đưa model VPR từ file `.ckpt` trên máy tính lên thành một dịch vụ API có thể truy cập từ bất kỳ đâu.

### Cấu trúc thư mục

```
deployment/
├── export_model.py              # Bước 1: Chuyển đổi model PyTorch → TorchScript
├── test_triton_client.py        # Bước 2: Kiểm thử Triton Server
├── Dockerfile.api               # Bước 3: Đóng gói FastAPI thành Docker image
├── docker-compose.yml           # Bước 4: Chạy toàn bộ hệ thống 1 lệnh
├── prometheus.yml               # Cấu hình thu thập metrics
├── triton_models/               # Kho chứa model cho Triton
│   └── vpr_encoder/
│       ├── config.pbtxt         # Khai báo input/output cho Triton
│       └── 1/
│           └── model.pt         # File TorchScript (tạo bởi export_model.py)
└── grafana/
    └── provisioning/
        └── datasources/
            └── prometheus.yml   # Kết nối Grafana → Prometheus tự động
```

---

## 1. `export_model.py` — Chuyển đổi Model

### Vai trò
Đây là **bước đầu tiên bắt buộc** trước khi deploy. Script này đọc file checkpoint `.ckpt` (định dạng PyTorch Lightning) và chuyển đổi sang định dạng **TorchScript** mà Triton Inference Server có thể hiểu.

### Tại sao cần chuyển đổi?
Model VPR được viết bằng PyTorch Lightning — framework này rất tiện cho training nhưng Triton không đọc trực tiếp được. TorchScript là định dạng "đông cứng" (frozen) của model, tối ưu hơn cho inference.

### Các thành phần chính

**`VPRInferenceWrapper`** — Lớp bọc model thuần túy cho inference:
```python
class VPRInferenceWrapper(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        descriptor = self.aggregator.forward_single(features)  # Chỉ dùng single-image path
        return descriptor
```
> Lý do tạo Wrapper: Model gốc `VPRModel` có nhiều logic dành cho training (loss, miner, cross-image). Wrapper chỉ giữ lại đúng phần cần cho inference, giúp TorchScript trace thành công.

**`load_vpr_model()`** — Nạp checkpoint:
```python
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)  # Tương thích nhiều format
model.load_state_dict(state_dict, strict=False)
```

**`export_torchscript()`** — Trace và lưu:
```python
traced = torch.jit.trace(wrapper, dummy_input)  # "Chụp ảnh" model với input mẫu
traced.save(output_path)  # Lưu ra model.pt
```

**`verify_output()`** — Kiểm tra sau khi export:
- Chạy 1 forward pass với ảnh ngẫu nhiên
- In ra chiều descriptor và L2 norm (phải ≈ 1.0 nếu đã normalize đúng)

### Cách chạy
```bash
python deployment/export_model.py \
    --ckpt_path  pretrainedWeight/your_model.ckpt \
    --output_dir deployment/triton_models/vpr_encoder/1 \
    --device     cpu \
    --image_size 322 322
```

| Tham số | Ý nghĩa | Mặc định |
|---------|---------|----------|
| `--ckpt_path` | Đường dẫn file `.ckpt` | Bắt buộc |
| `--output_dir` | Nơi lưu `model.pt` | `deployment/triton_models/vpr_encoder/1` |
| `--device` | Thiết bị trace (`cpu`/`cuda`) | `cpu` |
| `--image_size` | Kích thước ảnh input (H W) | `322 322` |

---

## 2. `triton_models/vpr_encoder/config.pbtxt` — Khai báo Model cho Triton

### Vai trò
File cấu hình này nói với Triton Server rằng: *"Này Triton, model tên là `vpr_encoder`, nó nhận input là gì, trả ra output là gì, và chạy trên GPU hay CPU?"*

### Giải thích từng phần
```protobuf
name: "vpr_encoder"           # Tên model — phải trùng khớp với tên thư mục
platform: "pytorch_libtorch"  # Định dạng TorchScript

max_batch_size: 16            # Tối đa 16 ảnh/batch

input [{
  name: "input__0"            # TorchScript tự đặt tên tham số theo thứ tự
  data_type: TYPE_FP32        # Float 32-bit
  dims: [ 3, -1, -1 ]         # [Channels, Height, Width] — -1 nghĩa là dynamic
}]

output [{
  name: "output__0"
  data_type: TYPE_FP32
  dims: [ 8448 ]              # Descriptor dimension = 64×128 + 256
}]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]       # Gom request thành batch để tăng throughput
  max_queue_delay_microseconds: 100       # Chờ tối đa 0.1ms để gom batch
}
```

> **Dynamic Batching**: Khi nhiều người dùng gửi request cùng lúc, Triton tự gom lại thành 1 batch rồi chạy 1 lần thay vì chạy từng request riêng lẻ. Điều này tăng hiệu suất GPU lên đáng kể.

---

## 3. `test_triton_client.py` — Kiểm thử Triton

### Vai trò
Script này dùng để **kiểm tra xem Triton đã hoạt động đúng chưa** sau khi deploy. Nó mô phỏng chính xác những gì FastAPI sẽ làm khi nhận ảnh từ người dùng.

### Luồng xử lý trong script
```
Ảnh JPEG  →  preprocess()  →  infer()  →  In kết quả + latency
```

**`preprocess()`** — Chuẩn bị ảnh:
```python
img = img.resize((322, 322))           # Resize về kích thước cố định
arr = (arr - mean) / std               # ImageNet normalization
arr = arr.transpose(2, 0, 1)           # HWC → CHW (PyTorch format)
arr = np.expand_dims(arr, axis=0)      # Thêm batch dim: [1, 3, H, W]
```

**`infer()`** — Gửi request đến Triton:
```python
infer_input = httpclient.InferInput(
    name="input__0",      # Phải khớp với config.pbtxt
    shape=image_array.shape,
    datatype="FP32",
)
response = client.infer(model_name="vpr_encoder", ...)
return response.as_numpy("output__0")  # Nhận descriptor về dạng numpy
```

**Benchmark latency**: Script chạy N lần (mặc định 5) và tính trung bình/tối thiểu latency.

### Cách chạy
```bash
# Triton phải đang chạy trước
python deployment/test_triton_client.py \
    --image   path/to/any_image.jpg \
    --runs    10
```

---

## 4. `Dockerfile.api` — Đóng gói FastAPI

### Vai trò
Dockerfile là **bản thiết kế** để tạo ra một Docker image chứa toàn bộ FastAPI gateway. Bất kỳ máy nào có Docker đều có thể chạy image này mà không cần cài Python, pip, hay bất cứ dependency nào.

### Giải thích từng lệnh

```dockerfile
FROM python:3.10-slim          # Dùng Python 3.10 nhẹ (bản slim không có GUI)

RUN apt-get update && apt-get install -y curl  # Cài curl để health check

WORKDIR /app                   # Tất cả file trong container sẽ nằm ở /app

# Cài dependencies TRƯỚC khi copy code
# → Nếu code thay đổi nhưng requirements không đổi, Docker dùng cache
# → Build nhanh hơn nhiều lần
COPY api/requirements.txt ./requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy source code
COPY api/ ./api/
COPY vpr_model.py ./
COPY models/ ./models/
COPY utils/ ./utils/

# Tạo user không có quyền root (bảo mật)
RUN useradd -m -u 1000 vpruser && chown -R vpruser:vpruser /app
USER vpruser                   # Chạy app với user này, không phải root

EXPOSE 8080                    # Khai báo port (metadata, không tự mở)

# Lệnh khởi động: uvicorn với 2 worker process
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

> **Tại sao copy `requirements.txt` trước rồi mới copy code?**
> Docker build theo từng "layer". Nếu bạn chỉ sửa code Python mà không sửa `requirements.txt`, Docker sẽ dùng lại layer cài pip đã cache → tiết kiệm hàng phút chờ đợi.

---

## So sánh: `Dockerfile.api` vs `docker-compose.yml`

Đây là câu hỏi rất hay vì hai file này đều liên quan đến Docker nhưng hoạt động ở **hai tầng hoàn toàn khác nhau**.

### Dockerfile.api — Bản thiết kế của 1 căn phòng

Dockerfile trả lời câu hỏi: **"Bên trong container FastAPI có gì?"**

- Nó định nghĩa **nội thất** của một container duy nhất: OS nào, Python version bao nhiêu, cài library gì, copy code nào vào, chạy lệnh gì khi start.
- Dockerfile **không biết** rằng ngoài kia còn có Triton, Prometheus hay Grafana.
- Kết quả của Dockerfile là một **image** (file tĩnh, như file ISO) — chưa chạy, chỉ là bản thiết kế.

```
Dockerfile.api  →  docker build  →  image "vpr-api"  →  (chưa chạy gì cả)
```

### docker-compose.yml — Bản thiết kế của cả tòa nhà

docker-compose.yml trả lời câu hỏi: **"Chạy những container nào, kết nối với nhau thế nào?"**

- Nó điều phối **nhiều container** cùng lúc: Triton, FastAPI, Prometheus, Grafana.
- Nó định nghĩa **quan hệ** giữa các container: FastAPI phụ thuộc Triton, Grafana phụ thuộc Prometheus.
- Nó set **biến môi trường**, **port mapping**, **volume**, **network** — những thứ chỉ tồn tại khi container đang chạy.

```
docker-compose.yml  →  docker compose up  →  4 container đang chạy và giao tiếp với nhau
```

### Bảng so sánh

| Đặc điểm | `Dockerfile.api` | `docker-compose.yml` |
|-----------|-----------------|----------------------|
| Phạm vi | 1 container (FastAPI) | Toàn bộ hệ thống (4 services) |
| Trả lời câu hỏi | "Bên trong container có gì?" | "Chạy những gì và kết nối thế nào?" |
| Kết quả | Docker image (file tĩnh) | Các container đang chạy |
| Biết về service khác | Không | Có (Triton, Prometheus, Grafana) |
| Cấu hình network | Không | Có (`vpr_net`) |
| Cấu hình volume | Không | Có (bind mount, named volume) |
| Cấu hình env var | Không | Có (`API_KEY`, `TRITON_HOST`,...) |
| Lệnh thực thi | `docker build` | `docker compose up` |

### Mối quan hệ giữa hai file

docker-compose.yml **tham chiếu** đến Dockerfile.api để biết cách build image:

```yaml
# Trong docker-compose.yml
vpr-api:
  build:
    context: ..
    dockerfile: deployment/Dockerfile.api  # ← Trỏ đến Dockerfile
```

Khi chạy `docker compose up --build`, Docker sẽ:
1. Đọc `docker-compose.yml` để biết cần những service nào.
2. Thấy service `vpr-api` cần build → đọc `Dockerfile.api` để tạo image.
3. Dùng image đó để khởi động container với đúng cấu hình (port, env, volume) từ compose.

> **Tóm lại**: Dockerfile tạo ra "nguyên liệu" (image). docker-compose.yml là "công thức nấu ăn" kết hợp tất cả nguyên liệu lại thành một bữa ăn hoàn chỉnh.

---

## 5. `docker-compose.yml` — Chạy Toàn Bộ Stack

### Vai trò
File này định nghĩa **tất cả các services** cần chạy và cách chúng kết nối với nhau. Chỉ cần 1 lệnh `docker compose up` là toàn bộ hệ thống được khởi động đúng thứ tự.

### Sơ đồ quan hệ giữa các service

```
grafana (port 3000)
    └─ phụ thuộc → prometheus (port 9090)
                       ├─ scrape → vpr-api:8080/metrics
                       └─ scrape → triton:8002/metrics

vpr-api (port 8080)
    └─ phụ thuộc (health check) → triton (port 8000, 8001, 8002)
```

### Giải thích các service

**`triton`**:
```yaml
command: tritonserver --model-repository=/models --allow-metrics=true --metrics-port=8002
```
- `--allow-metrics=true`: Bật endpoint `/metrics` tại port 8002 để Prometheus thu thập.
- `--model-repository=/models`: Trỏ đến thư mục chứa model (mount từ `./triton_models`).
- `healthcheck`: Docker Compose sẽ chỉ khởi động `vpr-api` sau khi Triton đã sẵn sàng, tránh lỗi kết nối khi start.

**`vpr-api`**:
```yaml
environment:
  - TRITON_HOST=triton   # Tên service trong Docker network, tự resolve thành IP
  - API_KEY=${VPR_API_KEY:-dev-secret-key}   # Đọc từ biến môi trường hoặc dùng default
depends_on:
  triton:
    condition: service_healthy   # Chỉ start sau khi Triton healthcheck pass
```

**`prometheus`**:
```yaml
command:
  - "--storage.tsdb.retention.time=7d"   # Giữ data metrics 7 ngày
  - "--web.enable-lifecycle"             # Cho phép reload config không cần restart
volumes:
  - prometheus_data:/prometheus          # Data metrics lưu vào named volume (persistent)
```

**`grafana`**:
```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}  # Password mặc định là "admin"
volumes:
  - ./grafana/provisioning:/etc/grafana/provisioning:ro    # Auto-setup datasource
```

**`volumes` (cuối file)**:
```yaml
volumes:
  prometheus_data:   # Named volume — data tồn tại kể cả khi container bị xóa
  grafana_data:
```
> Named volumes khác với bind mounts: dữ liệu được Docker quản lý, không bị mất khi `docker compose down`.

---

## 6. `prometheus.yml` — Cấu hình Thu thập Metrics

### Vai trò
File này nói với Prometheus: *"Đi hỏi metrics ở những địa chỉ này, mỗi 15 giây một lần."*

```yaml
global:
  scrape_interval: 15s       # Cứ 15 giây hỏi metrics 1 lần

scrape_configs:
  - job_name: "vpr-api"
    static_configs:
      - targets: ["vpr-api:8080"]   # hostname = tên service trong docker-compose
    metrics_path: "/metrics"         # FastAPI expose tại đây

  - job_name: "triton"
    static_configs:
      - targets: ["triton:8002"]    # Triton metrics port
    metrics_path: "/metrics"
```

> **Lưu ý**: Các hostname như `vpr-api` và `triton` chỉ hoạt động bên trong Docker network. Bên ngoài phải dùng `localhost:8080` và `localhost:8002`.

---

## 7. `grafana/provisioning/datasources/prometheus.yml` — Kết nối Grafana

### Vai trò
Thay vì phải vào giao diện Grafana và tự tay thêm datasource, file này **tự động cấu hình** kết nối Grafana → Prometheus ngay khi container khởi động.

```yaml
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy         # Grafana gọi Prometheus thay mặt browser (server-side)
    url: http://prometheus:9090   # Dùng tên service trong Docker network
    isDefault: true       # Đây là datasource mặc định cho mọi dashboard
```

---

## Thứ tự thực hiện

```
1. python deployment/export_model.py --ckpt_path ...
        ↓ Tạo ra deployment/triton_models/vpr_encoder/1/model.pt

2. python api/scripts/build_gallery_index.py --images_dir ...
        ↓ Tạo ra api/data/gallery_index.faiss + gallery_meta.json

3. cd deployment && docker compose up --build
        ↓ Khởi động: Triton → FastAPI → Prometheus → Grafana

4. python deployment/test_triton_client.py --image test.jpg
        ↓ Kiểm tra Triton hoạt động đúng

5. curl http://localhost:8080/api/v1/health
        ↓ Kiểm tra FastAPI sẵn sàng

6. Mở http://localhost:3000 → Grafana dashboard
```
