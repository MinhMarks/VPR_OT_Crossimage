# Cập Nhật Kiến Trúc CrossImageEncoder: Global Feature Optimization

## 1. Giới Thiệu
Phiên bản mới của `CrossImageEncoder` đã được tối ưu hóa để giải quyết vấn đề bùng nổ tham số. Thay vì xử lý toàn bộ tensor đặc trưng cụm (cluster features) khổng lồ, mô hình nay chỉ tập trung vào việc tăng cường **Global Scene Token**.

## 2. Thay Đổi Cốt Lõi
### Trước Khi Tối Ưu
- **Input:** Cluster Features $s \in \mathbb{R}^{128 \times 64}$ (đã flatten thành 8192).
- **Cơ chế:** Self-Attention trên vector 8192 chiều.
- **Tham số:** ~570 Triệu.

### Sau Khi Tối Ưu (Hiện Tại)
- **Input:** Global Token $t \in \mathbb{R}^{256}$.
- **Cơ chế:** Self-Attention trên vector 256 chiều.
- **Tham số:** ~1.6 Triệu (tính cả base model thì tổng cộng chỉ tăng rất ít).

## 3. Chi Tiết Kiến Trúc Mới

### Input
- **Global Token (`t`):** Là vector đại diện toàn cục cho bức ảnh, kích thước cố định `256`.
- Vector này chứa thông tin ngữ nghĩa cấp cao của bức ảnh.

### Transformer Encoder Block
Với `input_dim = 256`:
- **Heads:** 8
- **Head Dimension:** $256 / 8 = 32$.
- **Feedforward Dimension:** $256 \times 4 = 1024$.

### Tính Toán Tham Số (Cho 1 Layer)
1. **Multi-Head Attention:**
   - $4 \times (256 \times 256)$ weights + biases.
   - $4 \times 65,536 = 262,144$ tham số.
2. **Feedforward:**
   - $2 \times (256 \times 1024) = 524,288$ tham số.
3. **Total per Layer:** ~0.8 Triệu tham số.

**Tổng cộng (2 Layers):** ~1.6 Triệu tham số.

## 4. Luồng Dữ Liệu (Data Flow) trong SALAD

1. **Backbone & Aggregator (Base):**
   - Ảnh -> CNN -> Features -> NetVLAD/Salad Base.
   - Output: Cluster Features ($s$) và Global Token ($t$).
2. **Cross-Image Enhancement:**
   - Lấy $t$ của tất cả ảnh trong cùng một địa điểm (Batch size con = `img_per_place` = 4).
   - Đưa chuỗi $t_1, t_2, t_3, t_4$ vào Transformer Encoder.
   - Output: $t'_{1}, t'_{2}, t'_{3}, t'_{4}$ (đã được làm giàu thông tin từ các ảnh khác).
3. **Final Descriptor:**
   - Kết hợp $s$ (giữ nguyên) và $t'$ (đã enhance).
   - $Descriptor = Concat(Norm(t'), Norm(s))$.

## 5. Lợi Ích
- **Giảm 99% Tham Số:** Từ 573 triệu xuống còn khoảng hơn 4 triệu (Base 2.6M + 1.6M).
- **Tốc Độ:** Nhanh hơn vượt trội do kích thước ma trận tính toán nhỏ hơn 32 lần.
- **Bộ Nhớ:** Tiêu tốn cực ít VRAM.
