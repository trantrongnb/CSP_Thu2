# CSP_Thu2 — MFPA-FSLR baseline

Code baseline theo [MFPA_FSLR_Implementation_Spec.md](MFPA_FSLR_Implementation_Spec.md)
nằm trong [mfpa_fslr](mfpa_fslr/README.md).

```bash
cd mfpa_fslr
bash scripts/run_conda.sh check
```

Workspace dùng conda `signclip_5090` với PyTorch CUDA 12.8 và MPS.
Launcher gọi trực tiếp Python của conda; chọn `--env manta` nếu cần dùng manta.
Metadata WLASL hiện dùng split theo lớp 60:20:20.
Xem README của package để chuẩn bị dữ liệu, preprocessing, training và đánh giá A0–A5.
