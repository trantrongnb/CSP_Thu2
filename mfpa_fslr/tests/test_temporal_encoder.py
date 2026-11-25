import torch

from mfpa.models.temporal_encoder import PartTemporalEncoder


def test_temporal_encoder_shapes_masks_and_gradients():
    torch.manual_seed(7)
    encoder = PartTemporalEncoder(dimension=32, dropout=0.0)
    features = torch.randn(2, 3, 15, 32, requires_grad=True)
    mask = torch.ones(2, 3, 15, dtype=torch.bool)
    mask[0, 1, -3:] = False
    output = encoder(features, mask)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0, 1, -3:]) == 0
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_temporal_encoder_uses_order_not_just_framewise_pooling():
    torch.manual_seed(11)
    encoder = PartTemporalEncoder(dimension=16, dropout=0.0).eval()
    features = torch.randn(1, 3, 9, 16)
    mask = torch.ones(1, 3, 9, dtype=torch.bool)
    forward = encoder(features, mask)
    reversed_order = encoder(features.flip(2), mask)
    assert not torch.allclose(forward, reversed_order.flip(2), atol=1e-5, rtol=1e-5)
