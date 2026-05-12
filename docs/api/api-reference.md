# VPR API – Reference

Base URL: `http://localhost:8080` (local) hoặc `https://<your-domain>` (cloud)

## Authentication

Tất cả endpoints (trừ `/health` và `/`) yêu cầu API key:

```
Header: X-API-Key: <your-api-key>
```

---

## Endpoints

### `GET /api/v1/health`

Kiểm tra trạng thái dịch vụ.

**No auth required.**

**Response 200:**
```json
{
  "status": "ok",
  "triton_ready": true,
  "gallery_ready": true,
  "gallery_size": 2500
}
```

| Field | Mô tả |
|-------|-------|
| `status` | `"ok"` hoặc `"degraded"` |
| `triton_ready` | Triton server đang chạy và model đã load |
| `gallery_ready` | FAISS index đã load và có dữ liệu |
| `gallery_size` | Số lượng ảnh trong gallery |

---

### `POST /api/v1/retrieve`

Query image → Top-K matching places.

**Auth required** (`X-API-Key`).

**Request** (`multipart/form-data`):

| Field | Type | Required | Mô tả |
|-------|------|----------|-------|
| `image` | file | ✅ | Ảnh query (JPEG/PNG, max 10 MB) |
| `top_k` | int | ❌ | Số kết quả (default: 5, max: 20) |

**Request example:**
```bash
curl -X POST https://your-api/api/v1/retrieve \
     -H "X-API-Key: your-key" \
     -F "image=@query.jpg" \
     -F "top_k=5"
```

**Response 200:**
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
        "id": 142,
        "place_name": "Ben Thanh Market",
        "lat": 10.7769,
        "lon": 106.7009
      }
    }
  ]
}
```

| Field | Mô tả |
|-------|-------|
| `rank` | Thứ hạng (1 = tốt nhất) |
| `gallery_id` | Index trong gallery |
| `score` | Cosine similarity [0, 1] — càng cao càng giống |
| `distance` | L2 distance — càng thấp càng giống |
| `image_path` | Đường dẫn ảnh trong gallery |
| `metadata` | Metadata tùy chỉnh (GPS, tên địa điểm, ...) |

**Error responses:**
| Code | Mô tả |
|------|-------|
| 400 | Ảnh không hợp lệ hoặc quá lớn |
| 401 | API key sai hoặc thiếu |
| 429 | Rate limit (> 60 req/min) |
| 503 | Triton server không sẵn sàng |

---

### `POST /api/v1/index`

Thêm ảnh mới vào gallery database.

**Auth required** (`X-API-Key`).

**Request** (`multipart/form-data`):

| Field | Type | Required | Mô tả |
|-------|------|----------|-------|
| `image` | file | ✅ | Ảnh gallery (JPEG/PNG, max 10 MB) |
| `place_name` | string | ❌ | Tên địa điểm |
| `lat` | float | ❌ | Vĩ độ (GPS) |
| `lon` | float | ❌ | Kinh độ (GPS) |
| `image_path` | string | ❌ | Đường dẫn gốc |

**Request example:**
```bash
curl -X POST https://your-api/api/v1/index \
     -H "X-API-Key: your-key" \
     -F "image=@new_place.jpg" \
     -F "place_name=Ho Chi Minh City Hall" \
     -F "lat=10.7769" \
     -F "lon=106.7021"
```

**Response 200:**
```json
{
  "gallery_id": 2501,
  "gallery_size": 2501,
  "message": "Image indexed successfully as gallery_id=2501"
}
```

---

### `GET /docs`

Swagger UI tự động — thử API trực tiếp trong browser.

### `GET /redoc`

ReDoc UI — tài liệu đẹp hơn, phù hợp để share.

---

## Response Headers

| Header | Mô tả |
|--------|-------|
| `X-Process-Time-Ms` | Thời gian xử lý request (ms) |

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:8080"
API_KEY = "your-api-key"

headers = {"X-API-Key": API_KEY}

# Health check
r = requests.get(f"{API_URL}/api/v1/health")
print(r.json())

# Retrieve
with open("query.jpg", "rb") as f:
    r = requests.post(
        f"{API_URL}/api/v1/retrieve",
        headers=headers,
        files={"image": f},
        data={"top_k": 5},
    )
results = r.json()
for match in results["matches"]:
    print(f"#{match['rank']} — score={match['score']:.4f} — {match['metadata'].get('place_name', '')}")
```

---

## cURL Quick Reference

```bash
# Health
curl http://localhost:8080/api/v1/health

# Retrieve
curl -X POST http://localhost:8080/api/v1/retrieve \
     -H "X-API-Key: dev-secret-key" \
     -F "image=@query.jpg" \
     -F "top_k=5"

# Index new image
curl -X POST http://localhost:8080/api/v1/index \
     -H "X-API-Key: dev-secret-key" \
     -F "image=@gallery.jpg" \
     -F "place_name=District 1" \
     -F "lat=10.7769" \
     -F "lon=106.7"
```
