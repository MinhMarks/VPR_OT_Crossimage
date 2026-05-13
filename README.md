<div align="center">
  <h1>Visual Place Recognition với Optimal Transformer (VPR-OT)</h1>
  <p><strong>Phương pháp Cross-Image Enhancement ứng dụng Optimal Transport và Vision Transformer</strong></p>
</div>

---

## 📖 1. Giới thiệu

**Visual Place Recognition (VPR)** là bài toán cốt lõi trong lĩnh vực xe tự hành (Autonomous Vehicles) và định vị Robot (SLAM). Mục tiêu là xác định vị trí của một ảnh truy vấn (query) bằng cách tìm kiếm ảnh tương đồng nhất trong một kho dữ liệu khổng lồ (database/gallery). 

Tuy nhiên, VPR phải đối mặt với hai thách thức lớn:
- **Perceptual Variability:** Cùng một địa điểm có thể trông rất khác biệt do thay đổi thời tiết, ánh sáng, góc nhìn.
- **Perceptual Aliasing:** Các địa điểm khác nhau nhưng có kiến trúc đô thị giống hệt nhau.

**VPR-OT** giải quyết triệt để vấn đề này bằng cách kết hợp **DINOv2** (Vision Transformer), **Optimal Transport (Sinkhorn)** để gom cụm đặc trưng, và đặc biệt là cơ chế **Cross-Image Enhancement** giúp mô hình khai thác thông tin từ đa góc nhìn.

---

## 🧠 2. Ý tưởng cốt lõi: Cross-Image Enhancement

Thay vì xử lý từng bức ảnh một cách độc lập (Single-View), dự án nâng cấp mô hình lên khả năng **Multi-View Context Awareness**.

### 2.1. Nền tảng SALAD (Sinkhorn Algorithm for Locally Aggregated Descriptors)
- Sử dụng **DINOv2** để trích xuất *Local Features* và *Global Token*.
- Áp dụng lý thuyết **Vận tải tối ưu (Optimal Transport)** thông qua thuật toán Sinkhorn để gom cụm các local features thành một ma trận điểm số trơn tru, giải quyết triệt để lỗi "Cluster Collapse" của NetVLAD. 
- Cơ chế "thùng rác" (dustbin) giúp loại bỏ tự động các đặc trưng nhiễu (như bầu trời, mặt đường trống).

### 2.2. CrossImageEncoder
Các mô hình VPR thông thường mất đi mối liên kết giữa các ảnh chụp cùng một địa điểm dưới các góc nhìn khác nhau. **CrossImageEncoder** ra đời để giải quyết vấn đề này:
- Sử dụng mạng **Transformer Encoder** siêu nhẹ (chỉ thêm ~1.58M tham số).
- **Cơ chế hoạt động:** Chỉ áp dụng Multi-Head Self-Attention lên các *Global Tokens* của các ảnh cùng một địa điểm (ví dụ: batch 4 ảnh/địa điểm).
- **Kết quả:** Một bức ảnh bị che khuất hoặc thiếu sáng có thể "giao tiếp" và "mượn" thông tin ngữ nghĩa từ các ảnh khác cùng địa điểm, tạo ra một biểu diễn toàn cục (global representation) bền vững và phong phú hơn rất nhiều.

Dự án phát triển các biến thể:
- **CrossSalad1 (Batch-level Attention):** Cho độ chính xác cực cao (R@1 đạt **90.81%** trên MSLS validation, vượt xa baseline 89.05%).
- **MN_Salad (Increased Clusters):** Tối ưu hóa số lượng cluster cho ứng dụng thời gian thực.

---

## 🏗 3. Kiến trúc Hệ thống & Deployment

Hệ thống được thiết kế theo kiến trúc Microservices hiện đại, tối ưu hóa cho môi trường Cloud đa nền tảng, đảm bảo khả năng nội suy nhanh và khả năng mở rộng (Scalability).

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)

1. **Frontend (UI):** 
   - Xây dựng bằng Vanilla HTML/CSS/JS thuần với phong cách **Dark Glassmorphism** hiện đại.
   - Hỗ trợ **Dual Server Toggle** cho phép chuyển đổi nhanh giữa môi trường Local và Cloud.
   - Hiển thị Health Check tự động theo thời gian thực.

2. **API Gateway (Backend):**
   - Viết bằng **FastAPI** (Python), xử lý các endpoint `/index`, `/retrieve`, và `/health`.
   - Đảm nhiệm việc nhận ảnh, tiền xử lý (resize, normalize) chuẩn ImageNet, và đẩy vector sang database.
   - Tích hợp pipeline tự động upload ảnh lên ImgBB.
   - **Deploy:** Host trực tiếp trên **Render**.

3. **Inference Engine (AI Model):**
   - Chạy trên **Triton Inference Server**.
   - Model AI (DINOv2 + SALAD + CrossImageEncoder) được export ra định dạng ONNX để tối ưu tốc độ.
   - **Deploy:** Host trên **Hugging Face Spaces** (CPU tier), chịu tải tốt với luồng xử lý sequential.

4. **Vector Database:**
   - Sử dụng **Zilliz Cloud (Managed Milvus)**.
   - Indexing bằng thuật toán HNSW với độ đo `IP` (Inner Product).
   - Tốc độ truy vấn dưới `50ms` cho gallery hàng chục ngàn vector.

5. **Cloud Storage (Images):**
   - Sử dụng **ImgBB API** để host các hình ảnh của gallery, cho phép Frontend render ảnh public URL dễ dàng thay vì phải lưu trữ ảnh trên máy chủ backend.

> ⏳ **Monitoring (TODO):** Kế hoạch tích hợp Prometheus & Grafana để thu thập metrics (latency, gallery_size, triton_status) đang trong quá trình phát triển theo tài liệu `docs/deployment/05-monitoring.md`.

---

## 🚀 4. Hướng dẫn Cài đặt & Khởi chạy (Local)

### 4.1. Yêu cầu hệ thống
- Python 3.10+
- Conda (tùy chọn nhưng khuyến nghị)
- Môi trường Windows/Linux/Mac.

### 4.2. Cài đặt môi trường
Clone repository và cài đặt thư viện:
```bash
git clone https://github.com/MinhMarks/VPR_OT_Crossimage.git
cd VPR_OT_Crossimage
pip install -r requirements.txt
```

### 4.3. Cấu hình biến môi trường (`.env`)
Tạo file `.env` tại thư mục gốc với nội dung:
```ini
# Zilliz / Milvus
MILVUS_URI=in03-xxx.serverless.aws-eu-central-1.cloud.zilliz.com
MILVUS_TOKEN=your_zilliz_api_token
MILVUS_COLLECTION=vpr_gallery_cross

# API Authentication
API_KEY=your_secret_api_key

# Triton Server
TRITON_HOST=your-space-name.hf.space
TRITON_PORT=443
TRITON_SSL=true

# ImgBB
IMGBB_API_KEY=your_imgbb_api_key
```

### 4.4. Khởi chạy hệ thống cục bộ
Khởi động FastAPI server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```
Sau đó, mở file `frontend/index.html` bằng trình duyệt (hoặc dùng Live Server extension) để trải nghiệm giao diện.

### 4.5. Triển khai với Kubernetes trên Google Cloud Shell (Miễn phí)
Nếu máy tính không đủ tài nguyên (RAM) để chạy các thành phần cục bộ, bạn có thể triển khai API Gateway và Triton Server trên nền tảng **Google Cloud Shell** hoàn toàn miễn phí.
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/) và kích hoạt **Cloud Shell**.
2. Khởi tạo cụm Kubernetes siêu nhỏ: `minikube start`
3. Tải mã nguồn lên Cloud Shell và cấu hình docker-env: `eval $(minikube docker-env)`
4. Build API Gateway Image trực tiếp trên Cloud:
   ```bash
   docker build -t vpr-api-gateway:latest -f deployment/Dockerfile.api .
   ```
5. Khởi chạy các resource của Kubernetes:
   ```bash
   kubectl apply -f deployment/k8s/
   ```
6. **Mở Web Preview:** Forward port bằng lệnh `kubectl port-forward service/vpr-api-service 8080:80` và sử dụng chức năng "Web Preview" của Cloud Shell trên port 8080 để test API trực tiếp (kèm đường dẫn `/docs`).
*(Tham khảo chi tiết tại `docs/deployment/08-gcp-cloud-shell-k8s.md`)*


---

## 🗃 5. Xây dựng Kho dữ liệu (Gallery Population)

Với VPR, kho dữ liệu (Gallery) đóng vai trò quyết định. Dự án cung cấp script mạnh mẽ để nạp dữ liệu từ **MSLS Dataset** vào hệ thống Zilliz một cách hoàn toàn tự động.

### Pipeline Indexing:
Script `embed_msls_to_zilliz.py` xử lý dữ liệu theo cơ chế Cross-Image:
1. Nhóm 4 ảnh thuộc cùng 1 địa điểm (Location).
2. Tự động upload 4 ảnh lên **ImgBB** lấy public URL.
3. Tiền xử lý tensor `[4, 3, 224, 224]`.
4. Gọi qua **Triton (Hugging Face)** để nhúng qua CrossImage Attention, lấy về 4 descriptors riêng biệt `[4, 8448]`.
5. Đẩy 4 vectors + metadata (GPS, ImgBB URL, Place Name) lên **Zilliz Cloud**.

**Chạy Script:**
```bash
python scripts/embed_msls_to_zilliz.py \
  --milvus-host in03-xxx.cloud.zilliz.com \
  --milvus-token YOUR_ZILLIZ_TOKEN \
  --imgbb-key YOUR_IMGBB_KEY \
  --cities paris,tokyo \
  --max-locations 10
```

---

## 🛠 6. Duy trì hệ thống (Maintenance)

Hệ thống được host trên các dịch vụ Free Tier (Render, Hugging Face Spaces) có cơ chế "Sleep" khi không có request. Để đảm bảo hệ thống luôn sẵn sàng cho Demo:

- **UptimeRobot:** Được cấu hình ping mỗi 5 phút vào endpoint `/health` của FastAPI (Render) và endpoint của HuggingFace Space. Chi tiết xem tại `docs/deployment/09-keep-alive.md`.

---

## 📊 7. Kết quả & Hiệu năng thực nghiệm

Mô hình **CrossSalad1** (với module Multi-view Attention) thể hiện hiệu suất vượt trội trên tập validation MSLS đầy thách thức:

| Mô hình | Parameters | K@1 (%) | K@5 (%) | Đặc điểm |
|---|---|---|---|---|
| NetVLAD | >30M | 82.6 | 89.6 | CNN Baseline truyền thống |
| SALAD Base (DINOv2 Med) | 2.59M | 89.05 | 94.46 | Optimal Transport Baseline |
| **CrossSalad1 (Ours)** | **4.17M** | **90.81** | **95.41** | **Batch-level Cross-Image Attention** |
| MN_Salad (Ours) | ~2.6M | 90.81 | 95.41 | Tối ưu hóa số lượng cluster |

Việc vượt qua mốc **90% Recall@1** trên MSLS khẳng định sức mạnh của việc kết hợp Vision Transformers, Lý thuyết Vận tải Tối ưu và khai thác ngữ nghĩa Đa góc nhìn (Multi-view).

---
