import torch
from torch import nn
from torch.nn import functional as F

from .motion_encoder import MotionEncoder
from .structural_encoder import StructuralEncoder
from .fusion import ReliabilityFusion
from .temporal_pool import PartTemporalPool


class MFPAFSLR(nn.Module):
    """A0: landmarks, A1: flow, A2: sum, A3-A5: reliability fusion."""

    def __init__(self, dimension=128, variant="A5", normalization="group", regions=(0, 1, 2)):
        super().__init__()
        if variant not in {f"A{i}" for i in range(6)}:
            raise ValueError("variant must be A0..A5")
        if not regions or len(set(regions)) != len(regions) or not set(regions) <= {0, 1, 2}:
            raise ValueError("regions must contain unique LH=0, RH=1, M=2 indices")
        self.variant, self.regions = variant, tuple(regions)
        self.motion = MotionEncoder(dimension, normalization)
        self.structure = StructuralEncoder(dimension)
        self.fusion = ReliabilityFusion(dimension)
        self.pool = PartTemporalPool(dimension)
        self.global_projection = nn.Sequential(nn.Linear(3 * dimension, dimension), nn.GELU(),
                                              nn.Linear(dimension, dimension))
        if variant == "A0":
            self.motion.requires_grad_(False)
        if variant == "A1":
            self.structure.requires_grad_(False)
        if variant in {"A0", "A1", "A2"}:
            self.fusion.gate.requires_grad_(False)

    def encode(self, flow, hand_struct, mouth_struct, quality, valid_mask,
               struct_valid=None, transition_valid=None):
        quality = quality.clamp(0, 1)
        structural_mask = (valid_mask if struct_valid is None else struct_valid).bool()
        transitions = (torch.ones_like(quality[:, 0], dtype=torch.bool)
                       if transition_valid is None else transition_valid.bool())
        active = torch.zeros(3, dtype=torch.bool, device=quality.device)
        active[list(self.regions)] = True
        allowed = transitions[:, None, :] & active[None, :, None]
        structural_mask = structural_mask & allowed
        quality = quality * allowed
        motion_mask = quality > 0
        motion = self.motion(flow) if self.variant != "A0" else None
        structure = self.structure(hand_struct, mouth_struct) if self.variant != "A1" else None
        if structure is not None:
            structure = torch.where(structural_mask.unsqueeze(-1), structure, 0.0)
        if self.variant == "A0":
            features = self.fusion.norm(structure)
            mask, step_quality = structural_mask, structural_mask.float()
        elif self.variant == "A1":
            features = self.fusion.norm(torch.where(motion_mask.unsqueeze(-1), motion, 0.0))
            mask, step_quality = motion_mask, quality
        else:
            features, _ = self.fusion(motion, structure, quality, gated=self.variant != "A2")
            mask = (structural_mask | motion_mask) & valid_mask.bool()
            # Average reliability of the two modalities; structure remains usable if q=0.
            step_quality = (structural_mask.float() + quality) / 2
        mask = mask & allowed
        features = torch.where(mask.unsqueeze(-1), features, 0.0)
        parts, attention = self.pool(features, mask)
        present = mask.any(dim=-1)
        part_quality = (step_quality * mask).sum(-1) / transitions.sum(-1, keepdim=True).clamp_min(1)
        global_valid = present.any(-1)
        global_vector = self.global_projection((parts * present.unsqueeze(-1)).flatten(1))
        global_vector = F.normalize(global_vector, dim=-1, eps=1e-6)
        global_vector = torch.where(global_valid.unsqueeze(-1), global_vector, 0.0)
        return {"global": global_vector, "parts": parts, "part_quality": part_quality,
                "global_valid": global_valid, "attention": attention}

    def forward(self, **inputs):
        return self.encode(**inputs)
