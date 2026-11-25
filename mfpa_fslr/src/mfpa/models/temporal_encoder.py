import torch
from torch import nn
from torch.nn import functional as F


class _TemporalResidualBlock(nn.Module):
    """Lightweight depthwise temporal convolution with a masked residual path."""

    def __init__(self, dimension, dilation=1, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            dimension, dimension, kernel_size=3, padding=dilation,
            dilation=dilation, groups=dimension, bias=False,
        )
        self.expand = nn.Conv1d(dimension, 2 * dimension, kernel_size=1)
        self.project = nn.Conv1d(dimension, dimension, kernel_size=1)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, mask):
        batch, regions, steps, dimension = features.shape
        x = features.reshape(batch * regions, steps, dimension)
        valid = mask.reshape(batch * regions, steps).to(x.dtype)
        h = (x * valid.unsqueeze(-1)).transpose(1, 2)
        h = self.depthwise(h)
        h = F.glu(self.expand(h), dim=1)
        h = self.project(F.gelu(h)).transpose(1, 2)
        h = self.norm(x + self.dropout(h))
        h = h * valid.unsqueeze(-1)
        return h.reshape(batch, regions, steps, dimension)


class PartTemporalEncoder(nn.Module):
    """Encodes short-range gesture order separately for each anatomical region.

    Input/output are [B, R, T, D]. A shared encoder is used for both hands and
    mouth to keep parameter count low. Dilations 1 and 2 yield a seven-step
    receptive field while preserving the temporal length.
    """

    def __init__(self, dimension=128, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TemporalResidualBlock(dimension, dilation=1, dropout=dropout),
            _TemporalResidualBlock(dimension, dilation=2, dropout=dropout),
        ])

    def forward(self, features, mask):
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError("Expected features [B,R,T,D] and mask [B,R,T]")
        x = features * mask.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return x
