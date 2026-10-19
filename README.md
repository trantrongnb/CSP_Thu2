# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

## 1. Thông tin chung

| Hạng mục | Thiết lập |
|---|---|
| Tên phương pháp | MFPA-FSLR A5 |
| Dataset chính | WLASL |
| Số frame đầu vào | 16 frame/video |
| Tỉ lệ Train : Eval : Test | Theo lớp theo protocol; outputs_1shot_A5: train 12,517 / eval 4,262 / test 4,316 video |
| Seed | 42 |

## 2. Cấu hình huấn luyện

| Hạng mục | Thiết lập |
|---|---|
| Cấu hình episodic training | 5-way 1-shot, 5-way 5-shot |
| Số episode training | 1000 episode/epoch × 100 epoch |
| Query mỗi lớp | 5 |
| Backbone và pretrained | embedding 128D, normalization group; config không ghi pretrained backbone |

## 3. Kết quả đánh giá N-way K-shot

| Thiết lập | Accuracy | 95% CI half-width | Số episode | Nguồn/split |
|---|---:|---:|---:|---|
| 5-way 1-shot | 84.65% | ±0.68 điểm % | 100 | validation, epoch 52 |
| 5-way 5-shot | 94.72% | ±1.11 điểm % | 100 | validation, epoch 73 |

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

- Audit `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs_1shot_A5/wlasl_training_readiness/report_4996e1a6b2d6e73b6ad8b34567ced6a41d0f13d6d6ae836715bcf9930d35151d_8c6a2e80f50600fd2f63d91a342d923e0c308149a116dd092d0f86124b7a66ad.json`: kiểm tra 21,095 video; cache errors: 0. Lớp đủ điều kiện 1-shot Q=5: 2,000; 5-shot Q=5: 1,039.
- 5-way 1-shot: best validation ở epoch 52 là 84.65%; cuối run train 96.42% và validation 82.87% (thấp hơn peak 1.78 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs_1shot_A5/WLASL_60_20_20_A5_5w1s_seed42/`.
- 5-way 5-shot: best validation ở epoch 73 là 94.72%; cuối run train 99.37% và validation 93.52% (thấp hơn peak 1.20 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs_2shot_A5/WLASL_60_20_20_A5_5w5s_seed42/`.
- Chưa thấy evaluation JSON; accuracy ở mục 3 là validation, không phải held-out test.
- Chỉ có một training seed trong các run được quét; chưa ước lượng độ biến thiên giữa seed.
- Đây là protocol WLASL tự xây dựng của workspace; không so sánh trực tiếp với paper nếu protocol, preprocessing và test episodes khác.

---

Được tạo tự động từ 2 thư mục outputs*, config, `metrics.jsonl`, evaluation JSON và readiness audit.
