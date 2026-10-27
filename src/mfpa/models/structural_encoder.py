import torch
from torch import nn


class StructuralEncoder(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.hand = nn.Sequential(nn.Linear(67, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.mouth = nn.Sequential(nn.Linear(124, dimension), nn.GELU(), nn.Linear(dimension, dimension))

    def forward(self, hand_struct, mouth_struct):
        return torch.cat([self.hand(hand_struct), self.mouth(mouth_struct).unsqueeze(1)], dim=1)
