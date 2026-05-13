# Hướng dẫn Thực hành Kubernetes trên Google Cloud Shell (Miễn phí)

Nếu máy tính của bạn không đủ RAM để chạy Kubernetes nội bộ, **Google Cloud Shell** là giải pháp thay thế hoàn hảo. Google cung cấp cho bạn một máy ảo Linux có sẵn 5GB lưu trữ và bộ công cụ `kubectl`, `minikube`, `docker` hoàn toàn miễn phí.

---

## Bước 1: Truy cập Google Cloud Shell

1. Truy cập vào [Google Cloud Console](https://console.cloud.google.com/).
2. Đăng nhập bằng tài khoản Google của bạn.
3. Ở góc trên bên phải màn hình, bấm vào biểu tượng **Activate Cloud Shell** (hình cái máy tính có dấu `>_`).
4. Một cửa sổ dòng lệnh sẽ hiện ra ở phía dưới trình duyệt. Đợi vài giây để máy ảo khởi tạo.

---

## Bước 2: Chuẩn bị môi trường K8s

Trong cửa sổ Cloud Shell, hãy gõ lệnh sau để khởi động một cụm Kubernetes siêu nhỏ:

```bash
minikube start
```
*Lưu ý: Cloud Shell đã được Google tối ưu nên lệnh này chạy rất nhanh.*

---

## Bước 3: Tải mã nguồn lên Cloud Shell

Bạn cần đưa các file cấu hình từ máy tính của bạn lên máy ảo của Google:

1. Trong cửa sổ Cloud Shell, bấm vào biểu tượng **Ba chấm (⋮)** hoặc nút **Upload** (mũi tên đi lên).
2. Chọn **Upload Folder** và tải thư mục dự án của bạn lên (hoặc ít nhất là thư mục `deployment/k8s` và `deployment/triton_models`).
3. Sau khi upload xong, hãy dùng lệnh `cd` để đi vào thư mục đó. Ví dụ:
   ```bash
   cd salad
   ```

---

## Bước 4: Build Docker Image ngay trên Cloud

Để Kubernetes trong Minikube nhìn thấy code của bạn, chúng ta build image trực tiếp vào bộ nhớ của nó:

```bash
# Trỏ Docker vào môi trường của Minikube
eval $(minikube docker-env)

# Build API Gateway
docker build -t vpr-api-gateway:latest -f deployment/Dockerfile.api .
```

---

## Bước 5: Cấu hình lại đường dẫn Model (Quan trọng)

Vì trên Cloud Shell, đường dẫn thư mục sẽ khác máy tính của bạn, hãy chạy lệnh này để tự động cập nhật đường dẫn trong file `03-triton.yaml`:

```bash
# Lấy đường dẫn thư mục hiện tại và thay thế vào file yaml
SED_PATH=$(pwd)/deployment/triton_models
sed -i "s|path: /path/to/your/VPR-OT/salad/deployment/triton_models|path: $SED_PATH|g" deployment/k8s/03-triton.yaml
```

---

## Bước 6: Triển khai (Deploy)

Gõ các lệnh sau để khởi chạy hệ thống:

```bash
# 1. Cấu hình biến môi trường và mật khẩu
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-secret.yaml

# 2. Chạy Triton Server
kubectl apply -f k8s/03-triton.yaml

# 3. Chạy API Gateway
kubectl apply -f k8s/04-api-gateway.yaml
```

---

## Bước 7: Xem thành quả (Web Preview)

Đây là phần thú vị nhất. Để xem giao diện Swagger UI:

1. Chạy lệnh để mở cổng kết nối:
   ```bash
   kubectl port-forward service/vpr-api-service 8080:80
   ```
2. Ở phía trên cửa sổ Cloud Shell, bấm vào biểu tượng **Web Preview** (hình cái cửa sổ có mũi tên vòng cung).
3. Chọn **Preview on port 8080**.
4. Một tab mới sẽ mở ra. Hãy thêm `/docs` vào cuối URL (Ví dụ: `https://8080-dot-12345.cloudshell.dev/docs`).

**Bạn đã thành công!** Bạn đang chạy một hệ thống AI VPR trên hạ tầng Kubernetes của Google.

---

## Bước 8: Dọn dẹp
Để tránh lãng phí tài nguyên của Google (và để học cách tắt hệ thống):
```bash
kubectl delete -f deployment/k8s/
minikube stop
```
