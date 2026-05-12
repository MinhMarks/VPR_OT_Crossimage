# Monitoring: Prometheus + Grafana

## Tổng quan

Hệ thống VPR có 2 nguồn metrics:

| Nguồn | URL | Mô tả |
|-------|-----|-------|
| **FastAPI** | `:8080/metrics` | HTTP request metrics + custom VPR pipeline metrics |
| **Triton** | `:8002/metrics` | GPU utilization, inference latency, queue time (built-in) |

Prometheus scrape cả hai, Grafana visualize.

```
[FastAPI /metrics]  ─┐
                     ├─→ [Prometheus :9090] ─→ [Grafana :3000]
[Triton :8002]      ─┘
```

---

## Custom VPR Metrics

Định nghĩa tại `api/services/metrics.py`:

### Histograms (phân phối latency)

| Metric | Labels | Mô tả |
|--------|--------|-------|
| `vpr_triton_inference_seconds` | `model_name`, `model_version` | Thời gian forward pass DINOv2 + SALAD qua Triton |
| `vpr_faiss_retrieval_seconds`  | `top_k` | Thời gian FAISS nearest-neighbor search |
| `vpr_preprocess_seconds`       | — | Thời gian decode + resize + normalize ảnh |

> **Histogram** cho phép tính percentiles: P50, P95, P99 latency

### Gauges (giá trị tức thời)

| Metric | Mô tả |
|--------|-------|
| `vpr_gallery_size` | Số ảnh trong FAISS index |
| `vpr_triton_ready` | `1` = Triton sẵn sàng, `0` = không |
| `vpr_descriptor_dim` | Chiều descriptor (8448 với config hiện tại) |

### Counters (tổng tích lũy)

| Metric | Labels | Mô tả |
|--------|--------|-------|
| `vpr_requests_total` | `endpoint`, `status` | Tổng requests theo endpoint và kết quả |
| `vpr_index_operations_total` | — | Tổng ảnh đã index qua `/index` endpoint |
| `vpr_errors_total` | `error_type` | Tổng lỗi theo loại exception |

### HTTP metrics (tự động từ instrumentator)

| Metric | Mô tả |
|--------|-------|
| `http_requests_total` | Tổng HTTP requests |
| `http_request_duration_seconds` | Latency toàn bộ request |
| `http_requests_in_progress` | Số requests đang xử lý |

### Triton built-in metrics

| Metric | Mô tả |
|--------|-------|
| `nv_inference_request_success` | Số inference requests thành công |
| `nv_inference_request_failure` | Số inference requests thất bại |
| `nv_inference_queue_duration_us` | Thời gian request chờ trong queue (microseconds) |
| `nv_inference_compute_infer_duration_us` | Thời gian thực hiện inference |
| `nv_gpu_utilization` | GPU utilization (%) |
| `nv_gpu_memory_used_bytes` | GPU memory đã dùng |

---

## Chạy Monitoring Stack (Local)

```bash
cd deployment/

# Start toàn bộ stack (Triton + FastAPI + Prometheus + Grafana)
docker compose up --build

# Hoặc chỉ start Prometheus + Grafana (nếu Triton/FastAPI đang chạy rồi)
docker compose up prometheus grafana
```

| Service | URL |
|---------|-----|
| FastAPI Swagger | http://localhost:8080/docs |
| FastAPI Metrics | http://localhost:8080/metrics |
| Prometheus UI | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

---

## Grafana – Setup Dashboard

### Bước 1: Datasource đã tự động setup

File `deployment/grafana/provisioning/datasources/prometheus.yml` được mount vào Grafana → datasource Prometheus được tạo tự động khi start.

Kiểm tra: **Grafana → Configuration → Data Sources → Prometheus** → "Save & test" → ✅

### Bước 2: Import Dashboard

Grafana có sẵn community dashboards cho Triton và FastAPI:

**Triton Inference Server Dashboard:**
1. Vào Grafana → **Dashboards → Import**
2. Nhập ID: `12832` (NVIDIA Triton dashboard)
3. Chọn datasource: Prometheus → Import

**FastAPI HTTP Dashboard:**
1. Vào Grafana → **Dashboards → Import**
2. Nhập ID: `16110` (FastAPI Observability)
3. Chọn datasource: Prometheus → Import

### Bước 3: Tạo custom VPR dashboard

Vào **Dashboards → New Dashboard → Add Panel**. Ví dụ các query hữu ích:

**Latency P95 của Triton inference:**
```promql
histogram_quantile(0.95,
  rate(vpr_triton_inference_seconds_bucket[5m])
)
```

**Latency P50 / P95 / P99 so sánh:**
```promql
histogram_quantile(0.50, rate(vpr_triton_inference_seconds_bucket[5m]))
histogram_quantile(0.95, rate(vpr_triton_inference_seconds_bucket[5m]))
histogram_quantile(0.99, rate(vpr_triton_inference_seconds_bucket[5m]))
```

**Request rate (requests/second):**
```promql
rate(vpr_requests_total{status="success"}[1m])
```

**Error rate:**
```promql
rate(vpr_requests_total{status="error"}[1m])
```

**Gallery size theo thời gian:**
```promql
vpr_gallery_size
```

**FAISS retrieval latency theo top_k:**
```promql
histogram_quantile(0.95,
  rate(vpr_faiss_retrieval_seconds_bucket[5m])
) by (top_k)
```

**GPU memory Triton đang dùng:**
```promql
nv_gpu_memory_used_bytes / 1e9
```

---

## Prometheus UI – Quick Queries

Truy cập http://localhost:9090/graph để query trực tiếp:

```promql
# Xem tất cả VPR metrics
{job="vpr-api"}

# Kiểm tra Triton có đang nhận request không
rate(nv_inference_request_success[1m])

# Average latency Triton (seconds)
rate(vpr_triton_inference_seconds_sum[5m]) /
rate(vpr_triton_inference_seconds_count[5m])

# Triton ready status
vpr_triton_ready

# Gallery size hiện tại
vpr_gallery_size
```

---

## Kubernetes – Prometheus Operator

Nếu dùng K8s với `kube-prometheus-stack`:

```bash
# Cài Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prom prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Apply ServiceMonitor (tự động phát hiện FastAPI + Triton)
kubectl apply -f k8s/prometheus-servicemonitor.yaml

# Kiểm tra
kubectl get servicemonitor -n vpr
```

Prometheus Operator sẽ tự động scrape `/metrics` từ FastAPI và Triton mà không cần cấu hình thêm.

---

## Reload Prometheus config (không restart)

```bash
# Khi thay đổi prometheus.yml
curl -X POST http://localhost:9090/-/reload
```

---

## Checklist Monitoring

- [ ] `docker compose up` thành công, Grafana tại :3000
- [ ] http://localhost:8080/metrics trả về text Prometheus
- [ ] Prometheus UI → Status → Targets → `vpr-api` và `triton` đều **UP**
- [ ] Grafana datasource test → **OK**
- [ ] Import dashboard Triton (ID 12832)
- [ ] Tạo panel latency P95 cho Triton inference
