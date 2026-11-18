# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

## 1. Thông tin chung

| Hạng mục | Thiết lập |
|---|---|
| Tên phương pháp | MFPA-FSLR A5 |
| Dataset chính | WLASL |
| Số frame đầu vào | Không được ghi trong config.yaml của run |
| Số mẫu Train / Val / Test | Từ metadata được chỉ định trong config.yaml; metadata.csv: train 7,093 video/547 lớp; val 2,242/182; test 2,370/183 |
| Seed | 42 |

## 2. Cấu hình huấn luyện

| Hạng mục | Thiết lập |
|---|---|
| Cấu hình episodic training | 5-way 1-shot, 5-way 5-shot, 10-way 1-shot, 10-way 5-shot |
| Số episode training | 1000 episode/epoch × 20 epoch |
| Query mỗi lớp | 3, 5 |
| Backbone và pretrained | embedding 128D, normalization group; config không ghi pretrained backbone |

## 3. Kết quả đánh giá N-way K-shot

| Thiết lập | Accuracy | 95% CI half-width | Số episode | Nguồn/split |
|---|---:|---:|---:|---|
| 10-way 1-shot | 74.77% | ±2.24 điểm % | 100 | evaluation: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_10w1s_alpha_seed42/test_metrics.json` |
| 10-way 5-shot | 89.53% | ±1.30 điểm % | 100 | evaluation: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_10w5s_alpha_seed42/test_metrics.json` |
| 5-way 1-shot | 81.24% | ±2.30 điểm % | 100 | evaluation: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_5w1s_alpha_seed42/test_metrics.json` |
| 5-way 5-shot | 93.52% | ±1.18 điểm % | 100 | evaluation: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_5w5s_alpha_seed42/test_metrics.json` |

CI trong training log là xấp xỉ chuẩn `1.96 × σ / √E` tính trên episode. Nếu có file evaluation JSON, script ưu tiên kết quả và CI trong file đó.

## 4. Độ phức tạp và tốc độ suy luận

| Chỉ số | Kết quả |
|---|---|
| Tổng số tham số | 0.391 M |
| Số tham số được huấn luyện | 0.391 M |
| FLOPs | 21.334 GFLOPs/episode (torch profiler; không gồm preprocessing) |
| Inference latency | 177.41 ms/episode (có cache I/O; không phải ms/video) |
| Peak inference VRAM | 0.301 GB |

## 5. Ghi chú thực nghiệm

- Dataset split được đếm từ metadata.csv theo đường dẫn data.metadata trong từng config.yaml.
- 10-way 1-shot: best validation ở epoch 12 là 76.77%; cuối run train 88.54% và validation 73.17% (thấp hơn peak 3.60 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_10w1s_alpha_seed42/`.
- 10-way 5-shot: best validation ở epoch 15 là 90.17%; cuối run train 94.36% và validation 89.03% (thấp hơn peak 1.13 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_10w5s_alpha_seed42/`.
- 5-way 1-shot: best validation ở epoch 9 là 86.36%; cuối run train 89.17% và validation 82.08% (thấp hơn peak 4.28 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_5w1s_alpha_seed42/`.
- 5-way 5-shot: best validation ở epoch 16 là 94.12%; cuối run train 95.84% và validation 93.36% (thấp hơn peak 0.76 điểm %). Chọn `best.pt` theo validation.
- Artifacts: `/home/iec/disk2/cuongpd/dung36838/CSP_Thu2/mfpa_fslr/outputs/WLASL_60_20_20_A5_5w5s_alpha_seed42/`.
- Chỉ có một training seed trong các run được quét; chưa ước lượng độ biến thiên giữa seed.
- Đây là protocol WLASL tự xây dựng của workspace; không so sánh trực tiếp với paper nếu protocol, preprocessing và test episodes khác.

## 6. Bổ sung metric checkpoint (không train lại)

Đánh giá `best.pt` chỉ chạy inference. Lệnh dưới tạo accuracy/CI test, số tham số, latency, peak VRAM và FLOPs cho một episode đại diện:

```bash
bash scripts/run_conda.sh evaluate --checkpoint <run>/best.pt --episodes <test_episodes.json> \
  --output <run>/test_metrics.json --device cuda --profile-flops
```

Sau đó chạy lại `python scripts/generate_report.py` để điền các số đã lưu trong `test_metrics.json`. FLOPs là tổng operator FLOPs từ `torch.profiler`, không gồm preprocessing; latency gồm cache I/O.

---

Được tạo tự động từ 3 thư mục outputs*, config.yaml, `metrics.jsonl` và evaluation JSON.
