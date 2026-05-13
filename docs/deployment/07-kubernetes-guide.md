# Hướng dẫn Triển khai VPR trên Kubernetes (K8s)

Tài liệu này hướng dẫn bạn cách thiết lập và triển khai hệ thống **Visual Place Recognition (VPR)** trên môi trường Kubernetes cục bộ (sử dụng **Minikube** hoặc **Docker Desktop K8s**).

Kiến trúc này là mô hình **Hybrid Cloud (Kiến trúc lai)**:
- **FastAPI Gateway** và **Triton Inference Server** chạy trong Kubernetes (Local/On-Premise).
- **Milvus Vector Database** được host trên **Zilliz Cloud** để giảm tải dung lượng ổ cứng và RAM.

---

## 1. Yêu cầu Hệ thống
- **RAM:** Tối thiểu 16GB (khuyên dùng 32GB).
- **Phần mềm:** 
  - `Docker` hoặc `Docker Desktop`.
  - `kubectl` (Công cụ giao tiếp với Kubernetes).
  - `minikube` (Máy chủ K8s thu nhỏ) — *Bỏ qua nếu bạn dùng Kubernetes tích hợp sẵn của Docker Desktop.*

---

## 2. Chuẩn bị (Build Image & Config)

### Bước 2.1: Trỏ Docker CLI vào Minikube (Chỉ dành cho Minikube)
Để Kubernetes có thể nhìn thấy image Docker mà bạn sắp build, bạn cần trỏ terminal của bạn vào môi trường Docker bên trong Minikube:

```bash
# Khởi động Minikube (nếu chưa chạy)
minikube start --memory=8192 --cpus=4

# Trỏ Docker CLI vào Minikube
eval $(minikube docker-env)
```

### Bước 2.2: Build Docker Image cho API Gateway
Vào thư mục gốc của project (nơi chứa file `Dockerfile.api`) và chạy:

```bash
docker build -t minhmarks/vpr-api-gateway:latest -f deployment/Dockerfile.api .
```

### Bước 2.3: Chỉnh sửa thư mục Model Triton
Trong file `deployment/k8s/03-triton.yaml`, hãy tìm đến dòng `path: /path/to/your/VPR-OT/salad/deployment/triton_models` và thay thế nó bằng **đường dẫn tuyệt đối** trên máy tính của bạn trỏ tới thư mục `triton_models`.
Ví dụ: `path: /d/UIT/Research/VPR-OT/salad/deployment/triton_models` (Lưu ý cú pháp mount thư mục trên Windows).

---

## 3. Triển khai lên Kubernetes

Bây giờ bạn đã sẵn sàng biến file YAML thành các ứng dụng chạy thực tế (Pods).

### Bước 3.1: Áp dụng ConfigMap và Secrets
Đây là nơi chứa các cấu hình kết nối tới Zilliz Cloud:

```bash
kubectl apply -f deployment/k8s/01-configmap.yaml
kubectl apply -f deployment/k8s/02-secret.yaml
```

### Bước 3.2: Triển khai Triton Inference Server
```bash
kubectl apply -f deployment/k8s/03-triton.yaml
```
*Lưu ý: Image của Triton nặng khoảng 7GB, lần đầu tiên triển khai K8s sẽ phải tải nó về nên có thể mất 5-10 phút.*

Bạn có thể theo dõi tiến trình tải bằng lệnh:
```bash
kubectl get pods -w
```
Hãy đợi cho đến khi Pod `vpr-triton-deployment-xxx` hiện chữ `Running`.

### Bước 3.3: Triển khai FastAPI Gateway
```bash
kubectl apply -f deployment/k8s/04-api-gateway.yaml
```

---

## 4. Kiểm tra & Giám sát

### Kiểm tra Pods & Services
Gõ lệnh sau để xem toàn cảnh hệ thống của bạn:
```bash
kubectl get all
```
Bạn sẽ thấy 2 Pod đang chạy và 2 Service.

### Đọc Logs (Gỡ lỗi)
Nếu có lỗi xảy ra, hãy copy tên của Pod (ví dụ: `vpr-api-deployment-55d8f6d899-abcde`) và đọc log:
```bash
kubectl logs vpr-api-deployment-55d8f6d899-abcde
```

### Truy cập vào API Gateway (Swagger UI)
Vì API Gateway được cấu hình là `NodePort`, Kubernetes sẽ giấu nó phía sau một cổng ngẫu nhiên. Để lấy URL truy cập, hãy chạy:

**Dành cho Minikube:**
```bash
minikube service vpr-api-service
```
Lệnh này sẽ tự động mở trình duyệt web lên trang Swagger UI của bạn!

**Dành cho Docker Desktop:**
Truy cập `http://localhost:<NodePort>` (xem cổng NodePort bằng lệnh `kubectl get svc`).

---

## 5. Dọn dẹp (Xóa toàn bộ)
Khi bạn thực hành xong và muốn tắt để giải phóng RAM:
```bash
kubectl delete -f deployment/k8s/
```
Thao tác này sẽ dọn sạch sẽ toàn bộ môi trường Kubernetes của bạn chỉ trong 3 giây.
