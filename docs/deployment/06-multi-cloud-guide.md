# Hướng dẫn triển khai VPR Hệ sinh thái Đa đám mây (Miễn phí)

Tài liệu này hướng dẫn bạn cách chia nhỏ hệ thống VPR thành 3 phần độc lập để chạy trên các dịch vụ Cloud Miễn phí tốt nhất hiện nay.

## Tổng quan kiến trúc
- **Database:** Zilliz Cloud (Managed Milvus) — Lưu trữ vector.
- **Inference Server:** Hugging Face Spaces (CPU 16GB RAM) — Chạy Triton Server.
- **API Gateway:** Render.com — Cửa ngõ giao tiếp với người dùng.

---

## Giai đoạn 1: Thiết lập Database (Zilliz Cloud)

1. Truy cập [cloud.zilliz.com](https://cloud.zilliz.com/) và đăng ký tài khoản.
2. Tạo một **Cluster** mới (chọn **Starter/Free Tier**).
3. Sau khi Cluster ở trạng thái "Running", hãy copy:
   - **Public Endpoint:** Dạng `https://in03-xxx.zillizcloud.com:443`
   - **API Key (Token):** Tạo một API Key mới và lưu lại.

---

## Giai đoạn 2: Triển khai Inference (Hugging Face Spaces)

Hugging Face cho bạn 16GB RAM miễn phí, rất phù hợp để chạy Triton Server trên CPU.

1. Tạo Space mới tại [huggingface.co/new-space](https://huggingface.co/new-space).
2. Chọn **Docker** -> **Blank**.
3. **Upload dữ liệu:** Bạn cần upload thư mục `deployment/triton_models/` lên Space này.
4. **Tạo file `Dockerfile`** trong Space với nội dung sau:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.05-py3
USER root
RUN apt-get update && apt-get install python3-pip -y
COPY deployment/triton_models /models
EXPOSE 7860
# Chạy Triton trên cổng 7860 (cổng mặc định của HF Spaces)
CMD ["tritonserver", "--model-repository=/models", "--allow-gpu-metrics=false", "--http-port=7860"]
```

5. Sau khi Build xong, bạn sẽ có một URL dạng: `https://<user>-<space-name>.hf.space`. 
   > **Lưu ý:** URL này chính là `TRITON_HOST` của bạn.

---

## Giai đoạn 3: Triển khai Gateway (Render.com)

1. Push code của bạn lên GitHub (bao gồm cả thư mục `api/`).
2. Trên Render, tạo **Web Service** mới và kết nối với GitHub Repo.
3. Chọn cấu hình:
   - **Runtime:** `Python 3` (hoặc dùng Docker nếu muốn).
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

---

## Giai đoạn 4: Kết nối các mắt xích (Environment Variables)

Đây là bước quan trọng nhất. Trên Dashboard của **Render.com**, hãy thêm các biến môi trường sau:

| Biến | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| `API_KEY` | `your-secure-key` | Khóa để bảo vệ API của bạn |
| `USE_LOCAL_INFERENCE` | `false` | Bắt buộc để API gọi sang Triton Cloud |
| `TRITON_HOST` | `your-hf-space-url` | Bỏ `https://` và `/` (vd: `user-vpr.hf.space`) |
| `TRITON_HTTP_PORT` | `443` | HF Spaces dùng HTTPS (cổng 443) |
| `MILVUS_HOST` | `your-zilliz-endpoint` | Lấy từ Giai đoạn 1 |
| `MILVUS_TOKEN` | `your-zilliz-token` | Lấy từ Giai đoạn 1 |

---

## Kiểm tra hệ thống

Sau khi cả 3 server đều báo "Running/Live", bạn hãy truy cập URL của Render:
`https://your-app.onrender.com/api/v1/health`

Nếu kết quả trả về `status: healthy` và `triton_ready: true`, chúc mừng bạn đã triển khai thành công hệ thống VPR Đa đám mây hoàn toàn miễn phí!
