import torch
from torch import nn


class ReliabilityFusion(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * dimension + 6, dimension), nn.GELU(),
                                  nn.Linear(dimension, 1))
        self.norm = nn.LayerNorm(dimension)

    def forward(self, motion, structure, quality, quality_vector=None, gated=True):
        # Missing flow must be removed even for the simple-sum ablation.
        motion = torch.where((quality > 0).unsqueeze(-1), motion, 0.0)
        gate = quality_vector if quality_vector is not None else quality.unsqueeze(-1).expand(*quality.shape, 6)
        if gated:
            gate = (quality > 0).unsqueeze(-1) * torch.sigmoid(self.gate(torch.cat([motion, structure, gate], dim=-1)))
        else:
            gate = (quality > 0).unsqueeze(-1).to(motion.dtype)
        return self.norm(structure + gate * motion), gate
