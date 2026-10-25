# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

## 1. Thông tin chung

| Hạng mục | Thiết lập |
|---|---|
| Tên phương pháp | MFPA-FSLR A5 |
| Dataset chính | WLASL |
| Số frame đầu vào | Không được ghi trong config.yaml của run |
| Số mẫu Train / Val / Test | Từ metadata được chỉ định trong config.yaml; metadata.csv: train 7,094 video/547 lớp; val 2,242/182; test 2,370/183 |
| Seed | 42 |

## 2. Cấu hình huấn luyện

| Hạng mục | Thiết lập |
|---|---|
| Cấu hình episodic training | 5-way 1-shot, 5-way 5-shot |
| Số episode training | 1000 episode/epoch × 20 epoch |
| Query mỗi lớp | 5 |
| Backbone và pretrained | embedding 128D, normalization group; config không ghi pretrained backbone |

## 3. Kết quả đánh giá N-way K-shot

| Thiết lập | Accuracy | 95% CI half-width | Số episode | Nguồn/split |
|---|---:|---:|---:|---|
| 5-way 1-shot | 87.36% | ±1.80 điểm % | 100 | validation, epoch 9 |
| 5-way 5-shot | 95.60% | ±1.07 điểm % | 100 | validation, epoch 12 |

CI trong training log là xấp xỉ chuẩn `1.96 × σ / √E` tính trên episode. Nếu có file evaluation JSON, script ưu tiên kết quả và CI trong file đó.

## 4. Độ phức tạp và tốc độ suy luận

| Chỉ số | Kết quả |
|---|---|
| Tổng số tham số | chưa được lưu trong artifacts |
| Số tham số được huấn luyện | chưa được lưu trong artifacts |
| FLOPs | Chưa có phép đo trong artifacts |
| Inference latency | Chưa có phép đo trong artifacts |
| Peak inference VRAM | Chưa có phép đo evaluation được lưu |

## 5. Ghi chú thực nghiệm


