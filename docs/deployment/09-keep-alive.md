# Giữ Server Luôn Hoạt Động (Keep-Alive)

Các dịch vụ miễn phí như **Render.com** và **Hugging Face Spaces** sẽ tự động "ngủ đông" nếu không có request nào trong vòng **15 phút** (Render) và **48 giờ** (HF Space). Hướng dẫn này giúp bạn giữ chúng luôn thức bằng dịch vụ **UptimeRobot** miễn phí.

---

## Bước 1: Đăng ký UptimeRobot

1. Truy cập [https://uptimerobot.com](https://uptimerobot.com).
2. Bấm **Register for FREE**.
3. Đăng ký bằng email và xác nhận tài khoản.

---

## Bước 2: Tạo Monitor cho Render (API Gateway)

1. Đăng nhập vào Dashboard, bấm **+ Add New Monitor**.
2. Điền thông tin:
   - **Monitor Type**: `HTTP(s)`
   - **Friendly Name**: `VPR API Gateway (Render)`
   - **URL**: `https://[tên-app-render-của-bạn].onrender.com/api/v1/health`
   - **Monitoring Interval**: `5 minutes` (gói miễn phí tối thiểu là 5 phút)
3. Bấm **Create Monitor**.

> [!TIP]
> Endpoint `/api/v1/health` nhẹ, không tốn tài nguyên và không yêu cầu API Key → hoàn hảo để ping định kỳ.

---

## Bước 3: Tạo Monitor cho Hugging Face Space (Triton)

1. Bấm **+ Add New Monitor** lần nữa.
2. Điền thông tin:
   - **Monitor Type**: `HTTP(s)`
   - **Friendly Name**: `VPR Triton Server (HF Space)`
   - **URL**: `https://minhmarks-splace.hf.space/v2/health/ready`
   - **Monitoring Interval**: `5 minutes`
3. Bấm **Create Monitor**.

---

## Bước 4: Nhận thông báo khi Server bị sập (Tùy chọn)

1. Vào mục **Alert Contacts** trong Dashboard.
2. Thêm địa chỉ **email** của bạn.
3. UptimeRobot sẽ tự động gửi email cảnh báo nếu server của bạn không phản hồi trong 2 lần ping liên tiếp.

---

## Kết quả

Sau khi cấu hình xong, bảng Dashboard của UptimeRobot sẽ hiển thị:

| Monitor | Status | Uptime (30 ngày) |
|---|---|---|
| VPR API Gateway (Render) | ✅ UP | ~99% |
| VPR Triton Server (HF Space) | ✅ UP | ~99% |

> [!NOTE]
> Gói miễn phí của UptimeRobot cho phép tối đa **50 monitors** với chu kỳ **5 phút/lần**. Hoàn toàn đủ dùng cho dự án này.
