# CSP_Thu2 — MFPA-FSLR baseline

Code baseline theo [MFPA_FSLR_Implementation_Spec.md](MFPA_FSLR_Implementation_Spec.md)
nằm trong [mfpa_fslr](mfpa_fslr/README.md).

```bash
source .venv/bin/activate
cd mfpa_fslr
pytest -q
mfpa-smoke --output outputs/smoke_new
```

Môi trường `.venv` hiện có PyTorch CPU và MediaPipe để kiểm tra pipeline.
Xem README của package để chuẩn bị dữ liệu, preprocessing, training và đánh giá A0–A5.
