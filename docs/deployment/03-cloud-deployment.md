# Bước 3: Cloud Deployment

## Chiến lược

| Option | GPU? | Cost | Độ khó | Dùng khi nào |
|--------|------|------|--------|-------------|
| **Render.com** | ❌ CPU | Free | ⭐ Dễ nhất | Demo, test API endpoint |
| **GCP Cloud Run** | ❌ CPU | Free tier rộng | ⭐⭐ | Production API (scale tự động) |
| **GCP GCE + GPU** | ✅ T4/A100 | ~$0.35/hr (T4) | ⭐⭐⭐ | Full stack với Triton GPU |
| **Cloud CPU VM** | ❌ CPU | **~$0.05/hr** | ⭐⭐ | Full stack Triton CPU (Tiết kiệm nhất) |
| **AWS SageMaker** | ✅ | ~$0.526/hr | ⭐⭐⭐⭐ | Enterprise / industry |

> **Khuyến nghị cho người mới:**
> 1. Start với **Render.com** để có URL public ngay (free, 5 phút setup)
> 2. Sau đó thử **GCP** khi muốn GPU và hiểu production workflow

---

## Option A: Render.com (Dễ nhất, Miễn phí)

### Tại sao Render.com?
- Deploy từ GitHub — push code là auto deploy
- HTTPS tự động (không cần config SSL)
- Free tier: 512 MB RAM, shared CPU
- Disk persistent (1 GB free) cho FAISS index

### Hạn chế free tier
- Không có GPU → Triton phải chạy riêng (hoặc dùng mode không Triton)
- Cold start: app ngủ sau 15 phút không có request (free plan)

### Deploy steps

**1. Push code lên GitHub:**
```bash
git init
git add .
git commit -m "Initial VPR API"
git remote add origin https://github.com/<your_username>/vpr-api.git
git push -u origin main
```

**2. Vào [render.com](https://render.com) → New → Blueprint:**
- Connect GitHub repo
- Render tự đọc `render.yaml` và setup service

**3. Set environment variables trên Render dashboard:**
- `API_KEY` → (auto-generated hoặc set thủ công)
- `TRITON_HOST` → IP của Triton server nếu có

**4. Upload gallery data:**
Render có Persistent Disk. Upload file FAISS qua SSH hoặc build trong Dockerfile:
```dockerfile
# Thêm vào Dockerfile.api nếu muốn bundle gallery:
COPY api/data/ /app/data/
```

**5. Kiểm tra:**
```
https://vpr-api.onrender.com/docs
https://vpr-api.onrender.com/api/v1/health
```

---

## Option B: GCP (Google Cloud Platform)

### Tại sao GCP?
- **$300 free credit** trong 90 ngày (không cần thẻ thực)
- Cloud Run: serverless, pay-per-request, auto scale to zero
- GKE (Kubernetes): managed K8s cluster miễn phí 1 zonal cluster

### B1: Setup GCP

```bash
# Cài gcloud CLI: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com
```

### B2: Build & Push Docker Image

```bash
# Build
docker build -f deployment/Dockerfile.api -t vpr-api .

# Tag
docker tag vpr-api gcr.io/YOUR_PROJECT_ID/vpr-api:latest

# Push
docker push gcr.io/YOUR_PROJECT_ID/vpr-api:latest
```

### B3: Deploy FastAPI lên Cloud Run

```bash
gcloud run deploy vpr-api \
  --image gcr.io/YOUR_PROJECT_ID/vpr-api:latest \
  --platform managed \
  --region asia-southeast1 \        # Singapore (gần Vietnam)
  --allow-unauthenticated \          # Public API
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 3 \
  --set-env-vars="TRITON_HOST=your-gpu-vm-ip,API_KEY=your-secret"
```

Sau vài phút:
```
Service URL: https://vpr-api-xxxxxxxx-as.a.run.app
```

### B4: Deploy Triton lên GCE (GPU VM)

```bash
# Tạo VM với T4 GPU (Singapore region)
gcloud compute instances create vpr-triton \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --zone=asia-southeast1-b \
  --boot-disk-size=50GB

# SSH vào VM
gcloud compute ssh vpr-triton

# Trên VM: cài Docker + NVIDIA toolkit (xem docs/deployment/01-triton-local.md)
# Copy triton_models lên VM:
gcloud compute scp -r deployment/triton_models vpr-triton:~/

# Chạy Triton
docker run -d --gpus all \
  -v ~/triton_models:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

Cập nhật Cloud Run service để trỏ về Triton VM:
```bash
gcloud run services update vpr-api \
  --update-env-vars="TRITON_HOST=<VM_EXTERNAL_IP>"
```

---

## Option C: AWS (Industry Standard)

### Free Tier
- **12 tháng**: t2.micro EC2 (CPU only, không đủ cho DINOv2)
- **Không có GPU free tier**
- Dùng khi: công ty/tổ chức có AWS credit hoặc sau 90 ngày GCP

### Deploy nhanh trên AWS EC2

```bash
# Tạo instance p3.xlarge (V100 GPU) hoặc g4dn.xlarge (T4)
# Dùng AWS Console hoặc CLI

# Instance type gợi ý:
# g4dn.xlarge: 1× T4 GPU, $0.526/hr — phù hợp nhất
# p3.xlarge  : 1× V100 GPU, $3.06/hr — mạnh hơn nhưng đắt

# Sau khi SSH vào instance:
git clone https://github.com/<your>/vpr-api.git
cd vpr-api
docker compose -f deployment/docker-compose.yml up -d
```

### AWS SageMaker + Triton (Production Grade)

```python
# Tham khảo: https://docs.aws.amazon.com/sagemaker/latest/dg/triton-inference.html
import boto3
sm_client = boto3.client("sagemaker")

# Create Triton endpoint trên SageMaker
# (Advanced topic — sau khi đã quen với GCP)
```

---

## Option D: Cloud CPU Deployment (Triton on CPU) - **KHUYÊN DÙNG**

Đây là cách tốt nhất để chạy thử nghiệm với chi phí thấp (hoặc dùng Free Tier). Triton sẽ dùng thư viện OpenVINO hoặc PyTorch CPU để inference.

### 1. Yêu cầu cấu hình Server
Do model DINOv2 + SALAD khá nặng (~1.7GB cho weights), bạn cần VM có RAM tối thiểu **8GB** (khuyên dùng **16GB**) để chạy ổn định:
- **GCP:** `e2-standard-4` (4 vCPU, 16GB RAM)
- **AWS:** `t3.xlarge` hoặc `m5.large`
- **DigitalOcean:** 8GB RAM Droplet

### 2. Cấu hình Docker Compose cho CPU
Trên Server Cloud không có GPU, bạn phải xóa bỏ phần `deploy` của NVIDIA. Sửa file `deployment/docker-compose.yml`:

```yaml
  vpr_triton:
    image: nvcr.io/nvidia/tritonserver:24.05-py3
    # ... (giữ nguyên các phần khác)
    # XÓA HOẶC COMMENT PHẦN NÀY:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

### 3. Đảm bảo config.pbtxt dùng CPU
Mở `deployment/triton_models/vpr_encoder/config.pbtxt` và kiểm tra:
```pbtxt
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

### 4. Lệnh triển khai nhanh
Sau khi SSH vào Cloud Server:
```bash
# 1. Clone code
git clone https://github.com/your-repo/vpr-salad.git && cd vpr-salad

# 2. Tạo folder data cho Milvus
mkdir -p deployment/milvus_data deployment/etcd_data deployment/minio_data

# 3. Khởi chạy toàn bộ hệ thống
# Lưu ý: Triton trên CPU có thể mất 1-2 phút để load model lần đầu
docker compose -f deployment/docker-compose.yml up -d
```

### 5. Ưu điểm & Nhược điểm
- **Ưu điểm:** Chi phí cực rẻ, dễ setup, không cần cài Driver NVIDIA phức tạp.
- **Nhược điểm:** Tốc độ inference chậm hơn (khoảng 1-3 giây/ảnh tùy CPU). Phù hợp cho hệ thống có tần suất truy vấn không quá cao.

---

## Checklist trước khi deploy cloud

- [ ] Model đã export TorchScript (`deployment/triton_models/vpr_encoder/1/model.pt`)
- [ ] Gallery index đã build (`api/data/gallery_index.faiss`)
- [ ] `.env` không có trong git (`.gitignore`)
- [ ] `API_KEY` mạnh (random string ≥ 32 chars)
- [ ] Test local với `docker compose up` trước

---

## Bước tiếp theo

→ [Bước 4: Kubernetes Guide](./04-kubernetes-guide.md)
