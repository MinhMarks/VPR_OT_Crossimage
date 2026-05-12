# Bước 4: Kubernetes (K8s) – Hướng dẫn từ đầu

## K8s là gì? (Giải thích đơn giản)

Hãy tưởng tượng bạn là **quản lý nhà hàng**:

| Không có K8s | Có K8s |
|-------------|--------|
| Bạn phải tự thuê từng nhân viên (container), nếu nhân viên bỏ việc → bàn trống | K8s tự thuê nhân viên mới ngay khi có người nghỉ |
| Nếu khách đông → bạn phải tự gọi thêm nhân viên | K8s tự scale up khi traffic tăng |
| Mỗi bàn có địa chỉ riêng, phải nhớ từng bàn | K8s có "bảng hướng dẫn" (Service) — khách chỉ cần hỏi "bàn FastAPI" là được dẫn đến |

**Tóm lại**: K8s tự động hóa việc chạy, restart, scale, và kết nối các containers.

---

## 5 Khái niệm cốt lõi

### 1. Pod – Đơn vị nhỏ nhất
```
Pod = 1 hoặc nhiều containers chạy cùng nhau
```
- Mỗi pod có IP riêng trong cluster
- Nếu pod chết → K8s tạo pod mới (tự động!)
- **Ví dụ**: 1 pod chứa container FastAPI

```yaml
# Không tạo Pod trực tiếp — dùng Deployment thay thế
```

### 2. Deployment – Bản thiết kế
```
Deployment = "Tôi muốn luôn có N pods của app X chạy"
```
- Bạn khai báo số lượng replicas mong muốn
- K8s đảm bảo số đó luôn được duy trì

```yaml
# k8s/fastapi-deployment.yaml
spec:
  replicas: 2    # "Tôi muốn 2 FastAPI pods lúc nào cũng chạy"
```

### 3. Service – Địa chỉ ổn định
```
Service = Load balancer nội bộ, gộp nhiều pods thành 1 địa chỉ
```
- Pods có IP thay đổi mỗi lần restart → dùng Service để có IP cố định
- `fastapi-svc` → tự động phân tải giữa 2 FastAPI pods

```yaml
# k8s/fastapi-service.yaml
spec:
  selector:
    app: fastapi   # "Service này nhận traffic cho tất cả pods có label app=fastapi"
  ports:
    - port: 80
      targetPort: 8080
```

### 4. Ingress – Cổng từ Internet
```
Ingress = Nginx proxy từ internet vào cluster
```
- Nhận HTTP/HTTPS từ ngoài → route đến Service đúng

```yaml
# k8s/ingress.yaml
rules:
  - host: vpr.yourdomain.com
    http:
      paths:
        - path: /
          backend:
            service:
              name: fastapi-svc
```

### 5. Namespace – Phân vùng
```
Namespace = "Thư mục" để cách ly các app khác nhau
```
```bash
kubectl get pods -n vpr    # Chỉ xem pods trong namespace "vpr"
```

---

## K8s cho VPR Project

```
Internet
    │
    ▼ (port 443/80)
[Ingress: nginx]
    │  vpr.yourdomain.com →
    ▼
[Service: fastapi-svc]  ← địa chỉ DNS nội bộ ổn định
    │
    ├─→ [Pod: fastapi #1]  ╮
    └─→ [Pod: fastapi #2]  ╯ Deployment: replicas=2
          │
          │ (gọi triton-svc:8000)
          ▼
    [Service: triton-svc]
          │
          └─→ [Pod: triton #1 (GPU)]  ← Deployment: replicas=1
```

---

## Setup Local K8s với Minikube

**Minikube** = K8s cluster giả lập chạy trên máy local (học K8s mà không cần cloud).

### Cài Minikube

```bash
# Windows (dùng Chocolatey)
choco install minikube kubectl

# Hoặc download thủ công:
# https://minikube.sigs.k8s.io/docs/start/
```

### Start Cluster

```bash
minikube start --driver=docker --cpus=4 --memory=4g

# Kiểm tra
kubectl cluster-info
kubectl get nodes
```

### Build Image vào Minikube

```bash
# Để Minikube dùng local Docker image (không cần push lên registry)
eval $(minikube docker-env)   # Linux/Mac
# Hoặc trên Windows PowerShell:
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Build image
docker build -f deployment/Dockerfile.api -t vpr-api:latest .
```

### Deploy lên Minikube

```bash
# Sửa image trong fastapi-deployment.yaml:
# image: vpr-api:latest
# Thêm: imagePullPolicy: Never  (dùng local image)

# Apply toàn bộ manifests
kubectl apply -f k8s/

# Kiểm tra
kubectl get all -n vpr
```

**Output:**
```
NAME                           READY   STATUS    RESTARTS
pod/fastapi-7d9b4c5f8-abc12   1/1     Running   0
pod/fastapi-7d9b4c5f8-def34   1/1     Running   0
pod/triton-6f8c9d7b5-xyz56    1/1     Running   0

NAME                TYPE        CLUSTER-IP    PORT(S)
service/fastapi-svc ClusterIP   10.96.45.12   80/TCP
service/triton-svc  ClusterIP   10.96.78.34   8000/TCP

NAME                      READY   UP-TO-DATE
deployment.apps/fastapi   2/2     2
deployment.apps/triton    1/1     1
```

### Access FastAPI trong Minikube

```bash
# Port forward để test (không cần Ingress)
kubectl port-forward svc/fastapi-svc 8080:80 -n vpr

# Hoặc dùng Minikube tunnel
minikube service fastapi-svc -n vpr --url
```

---

## Các lệnh kubectl hay dùng

```bash
# Xem tất cả resources trong namespace vpr
kubectl get all -n vpr

# Xem logs của pod FastAPI
kubectl logs -n vpr deployment/fastapi -f

# Xem logs của pod Triton
kubectl logs -n vpr deployment/triton

# Mô tả pod (debug lỗi)
kubectl describe pod -n vpr <pod-name>

# Scale FastAPI lên 4 replicas
kubectl scale deployment fastapi --replicas=4 -n vpr

# Restart deployment (sau khi update image)
kubectl rollout restart deployment/fastapi -n vpr

# Xem lịch sử rollout
kubectl rollout history deployment/fastapi -n vpr

# Rollback về version trước
kubectl rollout undo deployment/fastapi -n vpr

# Xóa toàn bộ namespace (xóa hết resources)
kubectl delete namespace vpr
```

---

## Deploy lên GKE (Google Kubernetes Engine)

GKE cho bạn **1 zonal cluster miễn phí** (không charge cluster management fee).

```bash
# Tạo cluster
gcloud container clusters create vpr-cluster \
  --zone asia-southeast1-b \
  --num-nodes 2 \
  --machine-type e2-standard-2

# Kết nối kubectl với cluster
gcloud container clusters get-credentials vpr-cluster \
  --zone asia-southeast1-b

# Deploy
kubectl apply -f k8s/

# Tạo Ingress (cần enable nginx-ingress trên GKE):
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## Secret Management

Đừng hardcode API key trong YAML! Dùng K8s Secret:

```bash
# Tạo secret
kubectl create secret generic vpr-secrets \
  --from-literal=api-key="my-very-secret-key-xyz" \
  -n vpr

# Trong deployment.yaml đã có:
# env:
#   - name: API_KEY
#     valueFrom:
#       secretKeyRef:
#         name: vpr-secrets
#         key: api-key
```

---

## So sánh: Docker Compose vs Kubernetes

| Feature | Docker Compose | Kubernetes |
|---------|---------------|------------|
| Mục đích | Local dev / single server | Production / multi-node |
| Auto-restart | ✅ (unless-stopped) | ✅ (tự động, nhanh hơn) |
| Auto-scaling | ❌ Thủ công | ✅ HPA tự động |
| Rolling update | ❌ | ✅ zero-downtime |
| Multi-server | ❌ | ✅ |
| Learning curve | ⭐ Dễ | ⭐⭐⭐ |

**Kết luận**: Dùng Docker Compose để phát triển local, K8s khi muốn production-grade deployment.

---

## Roadmap học K8s

1. **Tuần 1**: `kubectl get/describe/logs` + Minikube local
2. **Tuần 2**: Deployment, Service, ConfigMap, Secret
3. **Tuần 3**: Ingress, PersistentVolume, Namespace
4. **Tuần 4**: HPA (auto-scaling), Rolling updates, Helm charts
5. **Sau đó**: GKE / EKS / AKS (managed K8s trên cloud)

---

**Resources:**
- [Kubernetes Official Tutorial](https://kubernetes.io/docs/tutorials/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/) — browser-based lab miễn phí
- [Minikube Docs](https://minikube.sigs.k8s.io/docs/)
