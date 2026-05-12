"""
Prometheus metrics cho VPR API.

Metrics được expose tại: GET /metrics

Bao gồm:
  - HTTP request metrics (tự động từ instrumentator)
  - vpr_triton_inference_seconds   – Histogram latency Triton inference
  - vpr_faiss_retrieval_seconds    – Histogram latency FAISS search
  - vpr_preprocess_seconds         – Histogram latency image preprocessing
  - vpr_gallery_size               – Gauge số ảnh trong gallery
  - vpr_triton_ready               – Gauge 1/0 Triton server sẵn sàng
  - vpr_requests_total             – Counter số request theo endpoint + status
  - vpr_descriptor_dim             – Gauge chiều descriptor
"""

from prometheus_client import Counter, Gauge, Histogram

# ──────────────────────────────────────────────────────────────────────────────
# Latency Histograms – đo thời gian xử lý từng bước
# ──────────────────────────────────────────────────────────────────────────────

TRITON_INFERENCE_LATENCY = Histogram(
    name="vpr_triton_inference_seconds",
    documentation="Thời gian Triton inference (DINOv2 + SALAD forward pass)",
    labelnames=["model_name", "model_version"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)

FAISS_RETRIEVAL_LATENCY = Histogram(
    name="vpr_faiss_retrieval_seconds",
    documentation="Thời gian FAISS nearest-neighbor search trong gallery",
    labelnames=["top_k"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

PREPROCESS_LATENCY = Histogram(
    name="vpr_preprocess_seconds",
    documentation="Thời gian preprocess ảnh (decode + resize + normalize)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# ──────────────────────────────────────────────────────────────────────────────
# Gauges – trạng thái hiện tại
# ──────────────────────────────────────────────────────────────────────────────

GALLERY_SIZE = Gauge(
    name="vpr_gallery_size",
    documentation="Số ảnh hiện có trong FAISS gallery index",
)

TRITON_READY = Gauge(
    name="vpr_triton_ready",
    documentation="Triton server sẵn sàng: 1=ready, 0=not ready",
)

DESCRIPTOR_DIM = Gauge(
    name="vpr_descriptor_dim",
    documentation="Chiều của place descriptor (= num_clusters*cluster_dim + token_dim)",
)

# ──────────────────────────────────────────────────────────────────────────────
# Counters – tổng số sự kiện
# ──────────────────────────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    name="vpr_requests_total",
    documentation="Tổng số requests theo endpoint và HTTP status",
    labelnames=["endpoint", "status"],
)

INDEX_OPERATIONS_TOTAL = Counter(
    name="vpr_index_operations_total",
    documentation="Tổng số ảnh đã thêm vào gallery qua /index endpoint",
)

ERRORS_TOTAL = Counter(
    name="vpr_errors_total",
    documentation="Tổng số lỗi theo loại",
    labelnames=["error_type"],
)
