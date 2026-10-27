import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(scores, mask, dim=-1):
    """All-masked rows return exactly zero, including under mixed precision."""
    mask = mask.bool()
    work = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    weights = torch.softmax(work, dim=dim) * mask
    return (weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)).to(scores.dtype)


class PartTemporalPool(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.attention = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, 1))
            for _ in range(3)
        ])

    def forward(self, features, mask):
        scores = torch.stack([net(features[:, r]).squeeze(-1)
                              for r, net in enumerate(self.attention)], dim=1)
        attention = masked_softmax(scores, mask)
        parts = F.normalize((attention.unsqueeze(-1) * features).sum(dim=2), dim=-1, eps=1e-6)
        return parts, attention
