# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

## 1. Thông tin chung

| Hạng mục | Thiết lập |
|---|---|
| Tên phương pháp | MFPA-FSLR, biến thể A5 (quality-gated fusion, global + part prototypes, adaptive alpha) |
| Dataset chính | WLASL|
| Số frame đầu vào | 16 frame/video |
| Tỉ lệ Train : Eval : Test | 60:20:20 |
| Seed | 42 |

## 2. Cấu hình huấn luyện

| Hạng mục | Thiết lập |
|---|---|
| Cấu hình episodic training | 5-way 1-shot và 5-way 5-shot |
| Số episode training | 1.000 episode × 100 epoch |
| Query mỗi lớp | 5 |
| Backbone và pretrained |Không có|


## 3. Kết quả đánh giá N-way K-shot
(Đánh giá trên validation set)
| Thiết lập | Accuracy validation | 95% CI half-width | Số validation episode | Best epoch |
|---|---:|---:|---:|---:|
| 5-way 1-shot | **84.65%** | ±0.68 điểm % | 1.000 | 52 |
| 5-way 5-shot | **94.72%** | ±1.11 điểm % | 100 | 73 |

CI được lấy từ metrics của từng run theo công thức xấp xỉ chuẩn `1.96 × σ / √E`.
## 4. Độ phức tạp và tốc độ suy luận

| Chỉ số | Kết quả |
|---|---|
| Tổng số tham số | Chưa được ghi trong artifacts hiện có |
| Số tham số được huấn luyện | Chưa được ghi trong artifacts hiện có |
| FLOPs | Chưa được đo |
| Inference latency | Chưa được đo theo video |
| Peak inference VRAM | Chưa có phép đo evaluation được lưu |

## 5. Ghi chú thực nghiệm


