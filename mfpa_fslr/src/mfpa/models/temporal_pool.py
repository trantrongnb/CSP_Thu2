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
        self.position = nn.Parameter(torch.zeros(1, 1, 32, dimension))
        self.temporal = nn.Sequential(nn.Conv1d(dimension, dimension, 3, padding=1, groups=dimension),
                                      nn.Conv1d(dimension, dimension, 1), nn.GELU())

    def forward(self, features, mask):
        t = features.shape[2]
        if t > self.position.shape[2]:
            raise ValueError(f"Temporal length {t} exceeds configured maximum")
        x = features + self.position[:, :, :t]
        b, r, _, d = x.shape
        x = x.reshape(b * r, t, d)
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        features = x.reshape(b, r, t, d)
        scores = torch.stack([net(features[:, r]).squeeze(-1)
                              for r, net in enumerate(self.attention)], dim=1)
        attention = masked_softmax(scores, mask)
        parts = F.normalize((attention.unsqueeze(-1) * features).sum(dim=2), dim=-1, eps=1e-6)
        return parts, attention
