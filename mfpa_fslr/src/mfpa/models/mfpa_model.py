import torch
from torch import nn
from torch.nn import functional as F

from .motion_encoder import MotionEncoder
from .structural_encoder import StructuralEncoder
from .fusion import ReliabilityFusion
from .temporal_pool import PartTemporalPool


class MFPAFSLR(nn.Module):
    """A0: landmarks, A1: flow, A2: sum, A3-A5: reliability fusion."""

    def __init__(self, dimension=128, variant="A5", normalization="group", regions=(0, 1, 2),
                 flow_dropout=0.15, structure_dropout=0.1, auxiliary_classes=0):
        super().__init__()
        if variant not in {f"A{i}" for i in range(6)}:
            raise ValueError("variant must be A0..A5")
        if not regions or len(set(regions)) != len(regions) or not set(regions) <= {0, 1, 2}:
            raise ValueError("regions must contain unique LH=0, RH=1, M=2 indices")
        self.variant, self.regions = variant, tuple(regions)
        self.flow_dropout, self.structure_dropout = flow_dropout, structure_dropout
        self.motion = MotionEncoder(dimension, normalization)
        self.structure = StructuralEncoder(dimension)
        self.fusion = ReliabilityFusion(dimension)
        self.pool = PartTemporalPool(dimension)
        self.global_projection = nn.Sequential(nn.Linear(3 * dimension, dimension), nn.GELU(),
                                              nn.Linear(dimension, dimension))
        self.aux_classifier = nn.Linear(dimension, auxiliary_classes) if auxiliary_classes else None
        if variant == "A0":
            self.motion.requires_grad_(False)
        if variant == "A1":
            self.structure.requires_grad_(False)
        if variant in {"A0", "A1", "A2"}:
            self.fusion.gate.requires_grad_(False)

    def encode(self, flow, hand_struct, mouth_struct, quality, valid_mask,
               struct_valid=None, transition_valid=None, quality_vector=None):
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
        motion = (self.motion(flow.to(dtype=next(self.motion.parameters()).dtype))
                  if self.variant != "A0" else None)
        if self.variant != "A1" and self.training:
            hand_struct, mouth_struct = self._jitter_structure(hand_struct, mouth_struct)
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
            if self.training:
                if self.flow_dropout:
                    drop = torch.rand_like(quality) < self.flow_dropout
                    quality = quality.masked_fill(drop, 0)
                    motion_mask = quality > 0
                if self.structure_dropout:
                    structural_mask = structural_mask & (torch.rand_like(quality) >= self.structure_dropout)
                    structure = torch.where(structural_mask.unsqueeze(-1), structure, 0.0)
            features, _ = self.fusion(motion, structure, quality, quality_vector,
                                      gated=self.variant != "A2")
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
                "global_valid": global_valid, "attention": attention,
                "aux_logits": self.aux_classifier(global_vector) if self.aux_classifier is not None else None}

    def forward(self, **inputs):
        return self.encode(**inputs)

    @staticmethod
    def _jitter_structure(hand, mouth):
        """Small shared scale/rotation/translation perturbation on structural inputs."""
        hand, mouth = hand.clone(), mouth.clone()
        batch = hand.shape[0]
        device, dtype = hand.device, hand.dtype
        scale = 1.0 + (torch.rand(batch, device=device, dtype=dtype) - 0.5) * 0.08
        angle = (torch.rand(batch, device=device, dtype=dtype) - 0.5) * 0.12

        def rotate_xy(values):
            theta = angle.reshape((batch,) + (1,) * (values.ndim - 1))
            cos, sin = theta.cos(), theta.sin()
            x, y = values[..., 0:1], values[..., 1:2]
            return torch.cat([cos * x - sin * y, sin * x + cos * y], dim=-1)

        def scale_to(values):
            factor = scale.reshape((batch,) + (1,) * (values.ndim - 1))
            return values * factor

        hand_xyz = hand[..., :63].reshape(batch, 2, hand.shape[2], 21, 3)
        hand_xyz[..., :2] = scale_to(rotate_xy(hand_xyz[..., :2]))
        hand[..., :63] = hand_xyz.reshape(batch, 2, hand.shape[2], 63)
        # Rotate and lightly scale the body-relative positions and relational vectors.
        for start, end in ((123, 125), (125, 127), (127, 131), (131, 135)):
            hand[..., start:end] = scale_to(rotate_xy(hand[..., start:end]))
        mouth_xyz = mouth[..., :120].reshape(batch, mouth.shape[1], 40, 3)
        mouth_xyz[..., :2] = scale_to(rotate_xy(mouth_xyz[..., :2]))
        mouth[..., :120] = mouth_xyz.reshape(batch, mouth.shape[1], 120)
        mouth[..., 120:124] = scale_to(rotate_xy(mouth[..., 120:124]))
        shift = (torch.rand(batch, 1, 1, 2, device=device, dtype=dtype) - 0.5) * 0.04
        hand[..., 123:125] += shift
        mouth[..., 120:122] += shift.squeeze(1)
        return hand, mouth
