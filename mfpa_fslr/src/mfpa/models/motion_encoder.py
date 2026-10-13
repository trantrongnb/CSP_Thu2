from torch import nn


class MotionEncoder(nn.Module):
    """One shared CNN for LH/RH/M; GroupNorm avoids episode batch dependence."""

    def __init__(self, dimension=128, normalization="group"):
        super().__init__()
        if normalization not in {"group", "batch"}:
            raise ValueError("normalization must be group or batch")
        layers, channels = [], 2
        for width in (32, 64, 128):
            norm = nn.GroupNorm(8, width) if normalization == "group" else nn.BatchNorm2d(width)
            layers += [nn.Conv2d(channels, width, 3, 2, 1), norm, nn.GELU()]
            channels = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, dimension))

    def forward(self, flow):
        b, r, t, c, h, w = flow.shape
        return self.net(flow.reshape(b * r * t, c, h, w)).reshape(b, r, t, -1)
