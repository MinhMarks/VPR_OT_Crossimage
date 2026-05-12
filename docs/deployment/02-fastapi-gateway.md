# Bước 2: FastAPI Gateway

## Mục tiêu

Wrap Triton Inference Server bằng một REST API thân thiện:
- Nhận ảnh upload từ client
- Gọi Triton để lấy descriptor
- Tìm kiếm FAISS gallery
- Trả về Top-K kết quả

## Kiến trúc FastAPI

```
api/
├── main.py                    # Entry point, khởi tạo app + services
├── config.py                  # Settings từ env vars
├── requirements.txt
├── middleware/
│   └── auth.py                # X-API-Key authentication
├── routers/
│   ├── retrieve.py            # POST /api/v1/retrieve
│   ├── index.py               # POST /api/v1/index
│   └── health.py              # GET  /api/v1/health
├── services/
│   ├── preprocess.py          # Image → tensor
│   ├── triton_client.py       # Gọi Triton HTTP API
│   └── retrieval.py           # FAISS nearest-neighbor
└── scripts/
    └── build_gallery_index.py # Build FAISS index offline
```

---

## Bước 2.1 – Build Gallery Index

Trước khi start API, cần pre-compute embeddings cho tất cả gallery images:

```bash
# Cài dependencies
pip install faiss-cpu Pillow numpy tqdm torch torchvision

# Build index từ thư mục gallery
python api/scripts/build_gallery_index.py --images_dir datasets/gallery/ --ckpt_path  pretrainedWeight\results\logs\lightning_logs\version_0\checkpoints\last.ckpt --output_dir api/data/ --device  cuda
```

Output:
```
api/data/
├── gallery_index.faiss    # FAISS index (binary)
└── gallery_meta.json      # Metadata mỗi ảnh
```

**gallery_meta.json** format:
```json
[
  {
    "id": 0,
    "image_path": "location_001/summer.jpg",
    "lat": 10.7769,
    "lon": 106.7009,
    "place_name": "Ben Thanh Market"
  }
]
```

> **Lưu ý**: Bạn có thể thêm bất kỳ metadata nào vào dict — API sẽ trả về nguyên si.

---

## Bước 2.2 – Cấu hình Environment

Tạo file `.env` ở root project:
```bash
# .env
TRITON_HOST=localhost
TRITON_HTTP_PORT=8000
GALLERY_INDEX_PATH=api/data/gallery_index.faiss
GALLERY_META_PATH=api/data/gallery_meta.json
API_KEY=my-secret-key-123
LOG_LEVEL=info
```

---

## Bước 2.3 – Chạy FastAPI (Local Development)

```bash
# Cài API dependencies
pip install -r api/requirements.txt

# Đảm bảo Triton đang chạy (Bước 1)
# docker compose up triton -d

# Start FastAPI
uvicorn api.main:app --reload --port 8080
```

Mở trình duyệt: **http://localhost:8080/docs** → Swagger UI tự động!

---

## Bước 2.4 – Test Endpoints

### Health Check
```bash
curl http://localhost:8080/api/v1/health
```
```json
{
  "status": "ok",
  "triton_ready": true,
  "gallery_ready": true,
  "gallery_size": 2500
}
```

### Retrieve (Query)
```bash
curl -X POST http://localhost:8080/api/v1/retrieve \
     -H "X-API-Key: my-secret-key-123" \
     -F "image=@query.jpg" \
     -F "top_k=5"
```

**Response:**
```json
{
  "query_descriptor_dim": 8448,
  "gallery_size": 2500,
  "top_k": 5,
  "matches": [
    {
      "rank": 1,
      "gallery_id": 142,
      "score": 0.9823,
      "distance": 0.0354,
      "image_path": "location_001/summer.jpg",
      "metadata": {
        "lat": 10.7769,
        "lon": 106.7009,
        "place_name": "Ben Thanh Market"
      }
    }
  ]
}
```

### Add to Gallery (Index)
```bash
curl -X POST http://localhost:8080/api/v1/index \
     -H "X-API-Key: my-secret-key-123" \
     -F "image=@new_place.jpg" \
     -F "place_name=Ho Chi Minh City Hall" \
     -F "lat=10.7769" \
     -F "lon=106.7021"
```

---

## Bước 2.5 – Chạy Full Stack (Triton + FastAPI)

```bash
cd deployment/
docker compose up --build
```

- FastAPI: http://localhost:8080/docs
- Triton metrics: http://localhost:8002/metrics

---

## Authentication

API dùng **X-API-Key header** pattern:
```
Header: X-API-Key: <your-key>
```

Key được set qua env var `API_KEY`. Nếu thiếu hoặc sai → **401 Unauthorized**.

> **Nâng cấp sau**: Có thể đổi sang JWT token (code đã có `python-jose` trong requirements) cho multi-user authentication.

---

## Rate Limiting

Default: **60 requests/minute** per IP.

Khi vượt ngưỡng → **429 Too Many Requests**.

---

## Response Headers

Mỗi response tự động có:
```
X-Process-Time-Ms: 125.45
```
Để track latency từng request.

---

## Bước tiếp theo

→ [Bước 3: Cloud Deployment](./03-cloud-deployment.md)
