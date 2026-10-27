# Kiểm tra baseline — 12/10/2026

Đã chạy trên Python 3.12.3, NumPy 1.26.4, PyTorch 2.14.1+cpu,
Torchvision 0.29.1+cpu, OpenCV 4.11.0 và MediaPipe 0.10.21.

| Kiểm tra | Kết quả |
|---|---|
| `pytest -q` | **29 passed** |
| Compile các module/scripts | Pass |
| `pip check` | Không có dependency bị lỗi |
| Forward T=16, ROI=64, D=128, B=2 | Global `[2,128]`, parts `[2,3,128]`, quality `[2,3]`; finite |
| Gradient | Tới CNN, cả hai structural MLP, gate, pooling và global projection |
| Masks | Gate bằng 0 khi q=0; missing part và all-invalid không tạo NaN |
| Zero-flow/q=0 | Global embedding bằng đường landmark-only với cùng weights |
| Prototype head | 1/5-shot; lambda=0 bằng global ProtoNet; support invalid không đổi prototype |
| Sampling | Episode seed tái lập; không trùng support/query; class/gloss split được kiểm tra |
| Resume | Trọng số cuối bằng chính xác một run không gián đoạn |
| Synthetic overfit | 40 bước: CE 0.076109 → gần 0, accuracy episode 100% |
| Train → checkpoint → evaluate | Pass qua CLI và code path dùng cho dữ liệu thật |
| A0–A5 runner | Cả sáu chạy xong, cùng hash episode file, xuất CSV |
| MediaPipe backend | Xử lý clip trống 4 frames, padding tới 16; cache masks/flow bằng 0 |
| Visualization | Xuất overlay PNG từ clip/cache trên |
| Farnebäck | Synthetic translation +3 pixel X, median flow đúng hướng và biên độ trong tolerance |
| RAFT-Small | Official C_T_V2 weights, hai frames 128×128; output finite, eval mode và frozen parameters |

Các artifacts trong workspace:

- [Overfit/smoke summary](outputs/smoke/smoke_summary.json)
- [Smoke checkpoint](outputs/smoke/training/best.pt)
- [Ablation CSV tổng hợp](outputs/smoke/ablations/results.csv)
- [Preprocessing manifest](outputs/preprocess_smoke/processed/preprocessing_manifest.json)
- [Overlay clip trống](outputs/preprocess_smoke/overlay.png)

**Tất cả accuracy ở đây dùng dữ liệu tổng hợp.** Clip trống chỉ kiểm tra decoding,
khởi tạo detector và mask khi không có detection; chưa xác minh tracking LH/RH/mouth
trên người ký hiệu thật. RAFT đã được kiểm tra riêng bằng frame tổng hợp, không
phải thí nghiệm accuracy. CUDA training/AMP chưa được chạy trong môi trường CPU này.

Repo chưa có video MS-ASL/SIGNUM/AUTSL hoặc danh sách class IDs của protocol gốc.
Vì vậy chưa có accuracy thực, chưa có so sánh với paper, và chưa xác minh cải thiện
của A5 so với A4. Chạy lại trên dữ liệu và kiểm tra ROI theo [README](README.md)
trước khi training lớn hoặc diễn giải kết quả nghiên cứu.
