# MFPA-FSLR baseline

Baseline PyTorch theo đặc tả v1: dense optical flow LH/RH/mouth + landmarks,
quality-gated fusion, temporal attention theo vùng, global/part prototypes và
episode-adaptive alpha. Đây là triển khai một thiết kế nghiên cứu; chưa có kết quả
thực nghiệm trên dữ liệu ký hiệu và không phải tái hiện AIS/LBAS của Bilge et al.

## 1. Cài đặt và kiểm tra nhanh

Chạy từ thư mục `CSP/mfpa_fslr`. Workspace dùng conda `signclip_5090` có
PyTorch CUDA 12.8, NumPy 1.26 và PyYAML. Python 3.10–3.12 được khuyến nghị cho
backend MediaPipe đã pin. Môi trường virtualenv CPU cũ đã được gỡ.

```bash
bash scripts/run_conda.sh check
# Môi trường thay thế:
bash scripts/run_conda.sh --env manta check
```

Launcher gọi thẳng Python trong conda, thêm `src` vào PYTHONPATH và đặt
`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps` nếu chưa được cấu hình. Không cần cài
package vào môi trường conda dùng chung. `sign5090` là alias của `signclip_5090`
trong launcher. `check` chỉ kiểm tra môi trường/GPU, không chạy training.
`check` có timeout 30 giây (exit code 124 nếu quá thời gian); có thể đặt
`MFPA_CUDA_CHECK_TIMEOUT=90` khi ổ đĩa/driver đang chậm. Lệnh `train` và
`evaluate` trên CUDA cũng chạy bước kiểm tra này trước khi bắt đầu; nếu MPS
không phản hồi, lệnh sẽ báo lỗi và thoát. Ctrl+C dừng cả tiến trình Python
đang chờ CUDA. Import các thư viện CUDA thành công chưa bảo đảm khởi tạo CUDA
context qua MPS đã thành công.
Manta hiện có NumPy 2.2, ngoài ràng buộc NumPy<2 của package; ưu tiên signclip_5090.
Các ví dụ `mfpa-*` ở các phần còn lại áp dụng khi package đã được cài vào môi
trường của bạn; trên server này dùng launcher cho train/evaluate/episodes.

Để cài package vào một môi trường conda riêng đã có PyTorch CUDA phù hợp:

```bash
python -m pip install -e '.[dev,landmarks,raft]'
python -m pytest -q
mfpa-smoke --output outputs/smoke_new --steps 40
```

Backend landmarks hỗ trợ MediaPipe Tasks `HolisticLandmarker` (MediaPipe 0.10.30+)
và tự tải model `.task` chính thức vào `data/models/` ở lần preprocessing đầu tiên.
Có thể đặt `MFPA_HOLISTIC_MODEL` hoặc truyền `--landmark-model` để dùng model đã có.
Nếu bản MediaPipe còn cung cấp `solutions.holistic`, backend legacy vẫn được dùng.
RAFT là dependency tùy chọn.

Smoke test sinh **feature tổng hợp**, overfit một episode 3-way 1-shot, rồi chạy
train/validation/checkpoint/evaluation thật qua cùng code path. Dữ liệu smoke có
4 frames, ROI 16 và D=32 để chạy nhanh. Accuracy smoke chỉ là sanity check.
Không chạy smoke vào thư mục đã có dữ liệu; chọn một `--output` mới.

## 2. Chuẩn bị protocol và video

Tạo `data/metadata.csv`, tối thiểu:

```csv
video_id,path,gloss,class_id,language,signer_id,split,is_mirrored
v0001,/data/videos/v0001.mp4,HELLO,12,ASL,signer_01,train,false
v0002,/data/videos/v0002.mp4,THANKS,13,ASL,,val,false
v0003,/data/videos/v0003.mp4,PLEASE,14,ASL,,test,false
```

Ví dụ chỉ minh họa định dạng. Mỗi split cần ít nhất N lớp; **mọi lớp** cần ít nhất
K+Q video khác nhau. Với 5-way/Q=5: ít nhất 6 video/lớp cho 1-shot và 10 cho 5-shot.
`video_id` phải duy nhất toàn metadata và dùng được làm tên file. Đường dẫn tương
đối được tính từ thư mục chứa metadata; signer không biết để trống.

`split` chỉ nhận `train`, `val`, `test`. Loader kiểm tra class ID và gloss không
trùng split **trong cùng ngôn ngữ**; class key là `language:class_id` để hỗ trợ
meta-test cross-lingual với ID độc lập. Không tự tạo split MS-ASL200 120/30/50:
muốn so sánh paper phải cung cấp đúng danh sách của protocol đó.

```bash
python scripts/build_metadata.py --metadata data/metadata.csv --splits-dir data/splits
```

`is_mirrored=true` nghĩa là video nguồn là ảnh gương; preprocessing unmirror trước
khi detect để giữ LH/RH theo giải phẫu. Không có cột này thì mặc định false.
Phải kiểm tra convention bằng overlay trên video thực; không tự flip augmentation.
Protocol cross-signer có ràng buộc signer riêng chưa được thêm vào sampler v1.

### WLASL trong workspace này

Cache đầy đủ ở `data/processed_wlasl_farneback`; metadata dùng để preprocessing là
`data/wlasl_preprocess.csv`. Metadata training riêng ở `data/wlasl_60_20_20/metadata.csv`.
Đây là **protocol tự xây dựng**, không phải split công bố hay kết quả tái hiện WLASL.

Pool chung cho 5-way 1/5-shot với Q=5 có 912 lớp và 11.706 video:
Các lớp được sắp theo gloss alphabetically (không phân biệt hoa/thường), sau đó
chia theo thứ tự vào train/validation/test với mục tiêu 60/20/20. Classes được nối
bởi duplicate source phải ở cùng split; vì vậy ranh giới có thể lệch nhẹ để không
tách các nhóm này. Metadata gốc cũng gán split theo cùng thứ tự alphabet.
Đã loại hai mẫu không dùng được chung cho A0–A5, giữ một bản cho mỗi source SHA256
trên toàn pool và chỉ giữ lớp còn ít nhất 10 video hợp lệ, khác nội dung.
Bản canonical ưu tiên lớp có ít video hơn, rồi video ID. Các lớp liên kết bởi
video trùng trong dữ liệu gốc được giữ cùng split. Split không dùng random seed;
episode sampling vẫn có seed riêng. Việc lọc và chia lớp không dùng embedding hoặc
accuracy của model.

`data/wlasl_60_20_20/protocol.json` lưu hashes, chính sách và thống kê; `excluded_videos.csv`
ghi video bị loại cùng lý do. Signer ID chưa biết nên không coi đây là benchmark
cross-signer. Các video/cache gốc vẫn có thể dùng để xây protocol khác.

Chạy training từ thư mục `mfpa_fslr`:

```bash
bash scripts/run_conda.sh train --config configs/train_wlasl_5w1s.yaml
bash scripts/run_conda.sh train --config configs/train_wlasl_5w5s.yaml
```

For these WLASL configs, `run_conda.sh train` first runs
`scripts/prepare_dataset.py`: it creates source metadata if needed, fills missing
video caches, audits cache/source consistency, and builds a content-deduplicated
60/20/20 protocol with at least 10 usable videos per class. The raw dataset is
expected at `../WLASL` (or pass `--dataset-root` when running the preparation
script manually). Matching caches and hash-keyed audit/protocol outputs are
reused on later runs. The first run can take a long time because it processes
every raw video. Validation episodes are generated by the training entry point
and stored with the prepared protocol.

Để chạy audit sau preprocessing, dùng report path mới. Lệnh đọc lại toàn bộ cache
và kiểm tra source hashes, có thể mất vài phút tùy ổ đĩa:

```bash
python scripts/audit_wlasl.py --metadata data/wlasl_preprocess.csv \
  --cache-dir data/processed_wlasl_farneback \
  --output outputs/wlasl_training_readiness/report_reaudit.json --workers 4
```

Để tái tạo metadata từ báo cáo audit đã kiểm tra đầy đủ cache, dùng output mới:

```bash
python scripts/prepare_wlasl_training.py \
  --metadata data/wlasl_preprocess.csv \
  --audit outputs/wlasl_training_readiness/report.json \
  --cache-dir data/processed_wlasl_farneback \
  --output-dir data/wlasl_rebuilt --min-videos 10 --seed 42
```

Metadata/config hash phải khớp audit. `--min-videos 6` tạo pool chỉ đủ 1-shot/Q=5;
không dùng pool đó trực tiếp với config 5-shot. Script không ghi đè protocol cũ.

## 3. Preprocessing và kiểm tra ROI

```bash
mfpa-preprocess --metadata data/metadata.csv \
  --config configs/preprocess.yaml --output data/processed_farneback --limit 20
mfpa-visualize --video /data/videos/v0001.mp4 \
  --cache data/processed_farneback/v0001.npz --output outputs/debug/v0001.mp4
# Sau khi kiểm tra overlay trên >=20 clips, xử lý toàn bộ:
mfpa-preprocess --metadata data/metadata.csv \
  --config configs/preprocess.yaml --output data/processed_farneback
```

Preprocessing gồm decode, letterbox 256×256, uniform sample 16 frames, Holistic,
flow giữa 15 cặp frame, union ROI t/t+1 và crop 64×64. Flow luôn tính trên **full
aligned frames** trước khi crop. Clip dưới 16 frames lặp frame cuối với mask 0.
Video một frame được cache nhưng sẽ bị Dataset từ chối vì không có transition thật.

Mặc định flow theo pixel ROI: `u *= 64/roi_width`, `v *= 64/roi_height`; chia scale
cố định 20, clip [-5,5], lưu float16. `flow_scale=20` là giá trị khởi tạo, chưa tune.
Có tùy chọn `canvas_pixel_displacement` nếu muốn giữ đơn vị canvas. Quality là
valid detections của cả hai frame nhân tỉ lệ diện tích ROI trong canvas; không có
confidence giả hay per-video min/max normalization. Không có head compensation.

Cache lưu landmarks, centers, masks, sampled IDs, ROI, transform letterbox, mirror
flag, FPS/kích thước gốc, timing, config hash và source-video hash. Manifest lưu
estimator/version/cấu hình và 40 mouth indices theo thứ tự cố định. Nếu đổi config,
thư viện estimator, video hoặc mirror flag, dùng cache directory mới. Cache được
ghi atomically và có thể tiếp tục chạy khi đã xử lý một phần.

Để dùng frozen RAFT-Small, sao chép `configs/preprocess.yaml`, đặt
`flow_estimator: raft_small`, chạy `--device cuda` và `--output data/processed_raft`.
Torchvision tải weights `C_T_V2` vào `.torch_hub` trong cache output ở lần đầu.
Đổi `data.cache_dir` trong cấu hình train sang cache này. Không backprop qua RAFT.

Toàn video thiếu vùng hợp lệ sẽ bị báo lỗi rõ ở Dataset, không tự loại video để
thay đổi protocol. Báo số failed samples và sửa metadata/protocol một cách công
khai nếu cần; sử dụng cùng tập video đủ điều kiện khi so sánh A0–A5.

## 4. Training và đánh giá

Các đường dẫn trong YAML được tính từ **working directory**, không từ file YAML.
Sửa `data.metadata`, `data.cache_dir` và `output_dir` cho dữ liệu của bạn.

```bash
mfpa-train --config configs/train_5w1s.yaml --device auto
# Hoặc huấn luyện 5-shot:
mfpa-train --config configs/train_5w5s.yaml --device auto
```

Mỗi episode encode support/query chung một batch, CE loss, AdamW, gradient clipping.
Mặc định GroupNorm để tránh phụ thuộc batch trong few-shot; có thể đặt
`model.normalization: batch`. Validation dùng episode JSON cố định, tự tạo lần đầu
ở `evaluation.val_episodes_file` và giữ nguyên cho các lần train/ablation sau.
Model tốt nhất được chọn theo **validation accuracy**; test không tham gia lựa chọn.

Output: `best.pt`, `last.pt`, `config.yaml`, `metrics.jsonl`, `splits.json`,
`provenance.json`. Checkpoint gồm optimizer/scaler/RNG, preprocessing hash,
dataset fingerprint và hash validation episodes. Resume một run bị gián đoạn với
cùng config và output:

```bash
mfpa-train --config configs/train_5w1s.yaml \
  --resume outputs/A5_5w1s_seed42/last.pt --device auto
```

Tạo test episodes một lần cho mỗi N/K/seed, dùng lại cho **mọi phương pháp**:

```bash
mfpa-episodes --metadata data/metadata.csv --cache-dir data/processed_farneback \
  --split test --n-way 5 --k-shot 1 --queries-per-class 5 \
  --count 1000 --seed 42 --output data/episodes/test_5w1s_seed42.json
mfpa-evaluate --checkpoint outputs/A5_5w1s_seed42/best.pt \
  --episodes data/episodes/test_5w1s_seed42.json --output outputs/A5_5w1s_seed42/test.json
```

Tạo thêm file bằng `--k-shot 5` cho 5-shot và `--n-way 20` khi đủ lớp/video.
Một checkpoint có thể đánh giá nhiều K mà không fine-tune. Với nhiều training seeds,
sao chép YAML, đổi `seed` và `output_dir`; giữ nguyên validation/test episode files.
Để chạy các evaluation seeds 42/123/2026, tạo ba episode files riêng và đánh giá
từng file. Script không ghi đè file episode đã tồn tại.

Kết quả JSON gồm mean episodic accuracy (%), CI95 normal approximation theo
episode (`1.96 * sample_std / sqrt(n)`), loss, alpha, confusion theo class key,
chi tiết từng episode, hashes, parameter counts, throughput, GPU peak memory.
Thời gian model có cache I/O và không bao gồm tiền xử lý; chi phí preprocessing
được đo riêng từ video thật trong cache. Không coi tổng hai phép đo này là benchmark
latency streaming end-to-end; chưa có FLOPs/hard-class/occlusion benchmark.

Meta-test cross-lingual dùng checkpoint ASL với `--metadata`, `--cache-dir`,
`--split test` và episode file của ngôn ngữ đích. Cấu hình preprocessing phải khớp
checkpoint. Code không tải MS-ASL/SIGNUM/AUTSL hay tự đặt subset của paper.

### Chọn metric matching

`head.matcher` chọn cách chấm điểm: `euclidean` (mặc định, squared Euclidean trên
embedding chuẩn hóa), `relation`, hoặc `hausdorff` (metric cố định). Relation mode
giữ Relation Network học được và thêm ba thống kê khoảng cách từ query tới tập
support observations bên cạnh các pair features: Hausdorff (shot xa nhất), trung
bình, và shot gần nhất. Điều này cho phép scorer học cách cân bằng Hausdorff với
độ phân tán của support khi có nhiều shot. Khoảng cách được tính cho global
feature và từng part, không gộp support shots trước đó. Chọn
`head.matcher: relation` trong YAML và dùng `output_dir` mới để train. Không đổi
matcher khi evaluate/resume checkpoint: config matcher phải khớp run đã train.
Để so sánh công bằng, giữ nguyên seed, episode files, encoder config và lịch train.

## 5. Ablation A0–A5

| Variant | Inputs/fusion | Prototype score |
|---|---|---|
| A0 | Landmark-only | Global |
| A1 | Flow-only | Global |
| A2 | Flow + landmark, sum | Global |
| A3 | Quality-gated fusion | Global |
| A4 | Quality-gated fusion | Global + parts, uniform alpha |
| A5 | Quality-gated fusion | Global + parts, adaptive alpha |

```bash
mfpa-train --config configs/train_5w1s.yaml --variant A0 --output outputs/A0_5w1s
python scripts/run_ablations.py --config configs/train_5w1s.yaml \
  --episodes data/episodes/test_5w1s_seed42.json --output outputs/ablations_5w1s
```

Script train tuần tự A0–A5, evaluate cùng test episodes và ghi `results.csv`.
Script không resume tự động hay ghi đè run đã có checkpoint. Để chạy hand-only hoặc
mouth-only, đặt `model.regions: [0,1]` hoặc `[2]`. Nếu toàn bộ vùng được chọn thiếu
ở video nào đó, phải xử lý failed sample theo protocol trước khi so sánh.

Các lựa chọn triển khai được xác định rõ:

- Structural hand vector 67D: wrist-relative xyz / max XY bounding extent,
  center xy và center delta. Mouth 124D: mean-centered xyz / max XY extent,
  center xy và delta. Cùng một scale chia cả xyz; z không được coi là metric depth.
  Không dùng thống kê val/test. Velocity bằng 0 nếu một trong hai detection thiếu.
- Landmark ở t vẫn hợp lệ nếu thiếu landmark t+1, nhưng flow/velocity cặp đó bị mask.
  Padding bị mask ở mọi nhánh.
- `q=0` triệt tiêu motion gate; structural features vẫn có thể dùng. Temporal
  reliability của A0 là detector mask, A1 là flow quality, A2–A5 là trung bình hai
  độ tin cậy. Part quality là tổng reliability / số transition thật, nên missing
  coverage làm giảm độ tin cậy. Đây là quy tắc bổ sung cho phần còn mở trong spec.
- Global projection nhân **presence mask** trước concat. Missing part/global trả
  vector 0; global-invalid sample bị từ chối trước khi tính score.
- A2 bỏ quality gate nhưng vẫn loại flow invalid. A4 dùng uniform trên các part
  prototypes hợp lệ; A5 cần ít nhất một class đối chứng hợp lệ khi tính alpha.
  Prototype vùng vẫn là quality-weighted mean ở A4/A5.
- Head renormalize alpha trên giao các vùng query/class hợp lệ, fallback global
  nếu không còn vùng. Alpha differentiable; `head.detach_alpha: true` là ablation.
- Các module không dùng trong A0–A2 vẫn được lưu để chia sẻ cùng cấu trúc state dict,
  được freeze và bỏ qua forward. Báo cả tổng parameter count và trainable count.

## 6. Cấu trúc và nguồn API

```text
configs/             preprocess và train 5-way 1/5-shot
src/mfpa/
  preprocessing/     frames, landmarks, ROI, Farnebäck/RAFT, cache, visualization
  datasets/          kiểm tra metadata/cache, structural vectors, episode sampler/collator
  models/            CNN/MLP/fusion/pooling/global projection/prototype head
  engine.py          train, resume, evaluate, checkpoint/provenance
  smoke.py           dataset tổng hợp và end-to-end sanity experiment
scripts/             validate metadata và chạy ablations
tests/               flow units, masks, no leakage, prototypes và backward
outputs/             artifacts do các lệnh tạo, được gitignore
```

API backend tham khảo: [MediaPipe Holistic](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md),
[Torchvision RAFT-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html),
[RAFT preprocessing](https://docs.pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html).
