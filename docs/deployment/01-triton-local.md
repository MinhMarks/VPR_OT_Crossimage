# Bước 1: Triton Inference Server – Local Deployment

## Mục tiêu

Serve model VPR (DINOv2 + SALAD) bằng NVIDIA Triton Inference Server trên máy local, thông qua Docker.

## Yêu cầu

| Thứ | Yêu cầu |
|-----|---------|
| OS  | Windows 10/11 với WSL2 **hoặc** Linux |
| Docker | Docker Desktop ≥ 4.x (bật WSL2 backend) |
| GPU | NVIDIA GPU (optional, Triton cũng chạy CPU) |
| GPU Driver | ≥ 520 (nếu dùng GPU) |
| NVIDIA Container Toolkit | Bắt buộc nếu dùng GPU |
| Python | ≥ 3.10 |
| Checkpoint | File `.ckpt` trained model |

---

## Bước 1.1 – Cài đặt Docker (Windows)

1. Tải [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Trong Settings → General → **Use WSL 2 based engine** ✓
3. Cài NVIDIA Container Toolkit (trong WSL2):
   ```bash
   # Mở WSL2 terminal
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. Kiểm tra:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

---

## Bước 1.2 – Export Model sang TorchScript

> **Tại sao TorchScript chứ không phải ONNX?**  
> DINOv2 sử dụng dynamic attention với control flow phức tạp — ONNX exporter thường bị lỗi. TorchScript (trace mode) an toàn và đơn giản hơn cho use case này.

```bash
# Từ thư mục gốc project (d:/UIT/Research/VPR-OT/salad)
conda activate salad

python deployment/export_model.py --ckpt_path pretrainedWeight\results\logs\lightning_logs\version_0\checkpoints\last.ckpt --output_dir deployment/triton_models/vpr_encoder/1 --device cpu --image_size 224 224
```

**Output kỳ vọng:**
```
============================================================
  VPR Model Export — TorchScript for Triton
============================================================
[✓] Loaded checkpoint: pretrainedWeight/your_model.ckpt
[✓] Descriptor dim  : 8448
[✓] L2 norm (≈ 1.0) : 1.0000
[→] Tracing model with TorchScript...
[✓] Trace OK — output shape: torch.Size([1, 8448])
[✓] Saved TorchScript model → deployment/triton_models/vpr_encoder/1/model.pt (XXX.X MB)
```

Kiểm tra file:
```
deployment/triton_models/
└── vpr_encoder/
    ├── config.pbtxt      ← Triton config
    └── 1/
        └── model.pt      ← TorchScript model
```

---

## Bước 1.3 – Chạy Triton Server

```bash
cd deployment/

# Build và start Triton (GPU mode)
docker compose up triton

# Nếu không có GPU, chỉnh config.pbtxt trước:
# Đổi  kind: KIND_GPU  →  kind: KIND_CPU
```

Chờ log:
```
I ... Successfully loaded model 'vpr_encoder'
I ... Started GRPCInferenceService at 0.0.0.0:8001
I ... Started HTTPService at 0.0.0.0:8000
I ... Started Metrics Service at 0.0.0.0:8002
```

Kiểm tra:
```bash
# Server ready?
curl http://localhost:8000/v2/health/ready
# → {"ready": true}

# Model loaded?
curl http://localhost:8000/v2/models/vpr_encoder/ready
# → {"ready": true}

# Model info
curl http://localhost:8000/v2/models/vpr_encoder
```

---

## Bước 1.4 – Test Inference

Cài client:
```bash
pip install tritonclient[http] Pillow numpy
```

Chạy test:
```bash
python deployment/test_triton_client.py --image path/to/any_image.jpg
```

**Output kỳ vọng:**
```
Connecting to Triton @ localhost:8000 ...
[✓] Triton ready — model 'vpr_encoder' v1 loaded

Preprocessing: path/to/any_image.jpg
  Input shape : (1, 3, 224, 224)  dtype: float32

Warm-up run ...
==================================================
  Descriptor dim    : 8448
  L2 norm (≈ 1.0)  : 1.0000
  Min / Max         : -0.1234 / 0.2345
  Avg latency       : 45.23 ms  (5 runs)
  Min latency       : 42.11 ms
==================================================
[✓] Triton inference test PASSED!
```

---

## Triton Config Giải thích

```protobuf
# deployment/triton_models/vpr_encoder/config.pbtxt

name: "vpr_encoder"           # Tên model (phải trùng tên folder)
platform: "pytorch_libtorch"  # TorchScript format

max_batch_size: 0              # Cross-Attention yêu cầu batch size tĩnh truyền từ API, nên set max=0

input [{
  name: "input__0"             # Tên input tensor (TorchScript auto-naming)
  data_type: TYPE_FP32         # float32
  dims: [ -1, 3, -1, -1 ]      # [Batch, C, H, W] — phải khai báo luôn cả Batch dimension
}]

output [{
  name: "output__0"
  data_type: TYPE_FP32
  dims: [ -1, 8448 ]           # [Batch, descriptor dimension]
}]

# dynamic_batching đã bị vô hiệu hóa vì model Cross-Attention không cho phép 
# tự động gộp các ảnh ngẫu nhiên của user khác nhau vào chung một lượt tính toán.
```

---

## Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `model.pt not found` | Chưa export | Chạy `export_model.py` trước |
| `CUDA out of memory` | Model quá lớn | Giảm `max_batch_size` trong config.pbtxt |
| `Connection refused` | Triton chưa start | Đợi log "Started HTTPService" |
| `kind: KIND_GPU not available` | Không có GPU/Driver | Đổi sang `KIND_CPU` trong config.pbtxt |

---

## Bước tiếp theo

→ [Bước 2: FastAPI Gateway](./02-fastapi-gateway.md)
