import cv2
import numpy as np


class FarnebackFlow:
    def __init__(self, **options):
        self.options = {"pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3,
                        "poly_n": 5, "poly_sigma": 1.2, "flags": 0, **options}

    def __call__(self, first, second):
        return cv2.calcOpticalFlowFarneback(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY),
                                          cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), None, **self.options)


class DisFlow:
    def __init__(self, preset="medium", scale=0.5):
        presets = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                   "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        if preset not in presets or not 0 < scale <= 1:
            raise ValueError("DIS preset must be fast/medium and scale in (0, 1]")
        self.dis, self.scale = cv2.DISOpticalFlow_create(presets[preset]), float(scale)

    def __call__(self, first, second):
        h, w = first.shape[:2]
        size = (max(8, int(round(w * self.scale / 8)) * 8),
                max(8, int(round(h * self.scale / 8)) * 8))
        a = cv2.resize(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), size)
        b = cv2.resize(cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), size)
        field = cv2.resize(self.dis.calc(a, b, None), (w, h), interpolation=cv2.INTER_LINEAR)
        field[..., 0] *= w / size[0]
        field[..., 1] *= h / size[1]
        return field


class RaftSmallFlow:
    """Frozen torchvision RAFT-Small; weights downloaded once by torchvision."""

    def __init__(self, device="cpu", num_flow_updates=12):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.device, self.updates = torch.device(device), num_flow_updates
        weights = Raft_Small_Weights.C_T_V2
        self.transforms = weights.transforms()
        self.model = raft_small(weights=weights).to(self.device).eval().requires_grad_(False)

    def __call__(self, first, second):
        import torch
        if min(first.shape[:2]) < 128 or any(x % 8 for x in first.shape[:2]):
            raise ValueError("RAFT requires canvas dimensions >=128 and divisible by 8")
        a, b = [torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
                for x in (first, second)]
        a, b = self.transforms(a, b)
        with torch.inference_mode():
            field = self.model(a, b, num_flow_updates=self.updates)[-1]
        return field[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
