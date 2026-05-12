# Kiến trúc hệ thống VPR Deployment

## Tổng quan

Hệ thống VPR (Visual Place Recognition) nhận ảnh đầu vào, tạo embedding, và trả về Top-K ảnh tương ứng trong gallery database.

## Sơ đồ luồng dữ liệu

```
[Client / Browser]
        │
        │  POST /api/v1/retrieve
        │  Header: X-API-Key: <key>
        │  Body: image file (JPEG/PNG)
        ▼
┌───────────────────────────────┐
│         FastAPI Gateway        │
│  api/main.py                   │
│                                │
│  1. Auth (X-API-Key)           │
│  2. Rate limit (60 req/min)    │
│  3. Preprocess image           │
│     → resize (322×322)         │
│     → normalize ImageNet       │
│     → float32 [1,3,H,W]        │
└──────────────┬────────────────┘
               │  HTTP JSON (tritonclient)
               ▼
┌───────────────────────────────┐
│    Triton Inference Server     │
│  nvcr.io/nvidia/tritonserver   │
│                                │
│  Model: vpr_encoder (v1)       │
│  Format: TorchScript           │
│  Input:  [B, 3, 322, 322]      │
│  Output: [B, 8448]             │
└──────────────┬────────────────┘
               │ descriptor [8448]
               ▼
┌───────────────────────────────┐
│  DINOv2 ViT-B/14 + SALAD      │
│  (inside TorchScript model)    │
│                                │
│  Backbone:  768-dim features   │
│  Aggregator: SALAD             │
│    64 clusters × 128 dim       │
│    + 256 token dim             │
│    = 8448 total descriptor     │
└──────────────┬────────────────┘
               │ query descriptor [8448]
               ▼
┌───────────────────────────────┐
│     FAISS Gallery Index        │
│  api/services/retrieval.py     │
│                                │
│  IndexFlatIP (cosine sim)      │
│  N gallery embeddings          │
│  → Top-K nearest neighbors     │
└──────────────┬────────────────┘
               │ List[PlaceMatch]
               ▼
[Client nhận Top-K kết quả + metadata]
```

## Cấu trúc thư mục

```
salad/
├── vpr_model.py               # Model PyTorch Lightning
├── models/
│   ├── backbones/             # DINOv2 backbone
│   └── aggregators/
│       ├── salad.py           # SALAD aggregator (chính)
│       └── salad_base.py      # Optimal transport clustering
│
├── deployment/                # Triton deployment
│   ├── export_model.py        # Export → TorchScript
│   ├── test_triton_client.py  # Test Triton
│   ├── docker-compose.yml     # Triton + FastAPI stack
│   ├── Dockerfile.api         # FastAPI container
│   └── triton_models/
│       └── vpr_encoder/
│           ├── config.pbtxt   # Triton model config
│           └── 1/
│               └── model.pt   # TorchScript (sau khi export)
│
├── api/                       # FastAPI gateway
│   ├── main.py                # App entry point
│   ├── config.py              # Settings (env vars)
│   ├── requirements.txt
│   ├── middleware/
│   │   └── auth.py            # API key auth
│   ├── routers/
│   │   ├── retrieve.py        # POST /retrieve
│   │   ├── index.py           # POST /index
│   │   └── health.py          # GET /health
│   ├── services/
│   │   ├── preprocess.py      # Image preprocessing
│   │   ├── triton_client.py   # Triton HTTP client
│   │   └── retrieval.py       # FAISS retrieval
│   ├── scripts/
│   │   └── build_gallery_index.py
│   └── data/                  # (runtime)
│       ├── gallery_index.faiss
│       └── gallery_meta.json
│
├── k8s/                       # Kubernetes manifests
│   ├── namespace.yaml
│   ├── fastapi-deployment.yaml
│   ├── fastapi-service.yaml
│   ├── triton-deployment.yaml
│   ├── triton-service.yaml
│   └── ingress.yaml
│
└── render.yaml                # Render.com deployment

```

## Descriptor Dimension

| Component          | Dim  |
|--------------------|------|
| Cluster features   | 64 × 128 = 8192 |
| Token features     | 256  |
| **Total**          | **8448** |

## Stack Versions

| Component | Version |
|-----------|---------|
| Triton Server | 24.05-py3 |
| FastAPI | ≥ 0.111 |
| FAISS | ≥ 1.7 (CPU) |
| DINOv2 | ViT-B/14 |
