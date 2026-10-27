import pytest
import torch
from torch.nn import functional as F

from mfpa.models import part_proto_logits


def embeddings(k=1):
    generator = torch.Generator().manual_seed(42)
    return (torch.randn(5, k, 128, generator=generator), torch.randn(5, k, 3, 128, generator=generator),
            torch.ones(5, k, 3), torch.randn(25, 128, generator=generator),
            torch.randn(25, 3, 128, generator=generator), torch.ones(25, 3))


@pytest.mark.parametrize("k", [1, 5])
def test_shape_alpha_and_exact_global_reduction(k):
    values = embeddings(k)
    logits, alpha = part_proto_logits(*values)
    assert logits.shape == (25, 5)
    assert alpha.shape == (5, 3)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(alpha.sum(-1), torch.ones(5))
    global_only, _ = part_proto_logits(*values, lam=0)
    pg = F.normalize(F.normalize(values[0], dim=-1).mean(1), dim=-1)
    qg = F.normalize(values[3], dim=-1)
    expected = -((qg[:, None] - pg[None]) ** 2).sum(-1) / .2
    torch.testing.assert_close(global_only, expected, atol=0, rtol=0)


def test_invalid_prototypes_query_mask_and_support_only_alpha():
    sg, sp, sq, qg, qp, qq = embeddings()
    sq[:, :, 2] = 0
    qq[0] = 0
    logits, alpha = part_proto_logits(sg, sp, sq, qg, qp, qq)
    assert not alpha[:, 2].any()
    torch.testing.assert_close(alpha.sum(-1), torch.ones(5))
    global_only, _ = part_proto_logits(sg, sp, sq, qg, qp, qq, lam=0)
    torch.testing.assert_close(logits[0], global_only[0], rtol=0, atol=0)
    _, changed_alpha = part_proto_logits(sg, sp, sq, qg * 10, qp.flip(0), torch.zeros_like(qq))
    torch.testing.assert_close(alpha, changed_alpha, rtol=0, atol=0)
    sq.zero_()
    logits, alpha = part_proto_logits(sg, sp, sq, qg, qp, qq)
    assert not alpha.any()
    assert torch.isfinite(logits).all()


def test_uniform_weights_and_missing_counterclass():
    values = list(embeddings())
    _, alpha = part_proto_logits(*values, adaptive=False)
    torch.testing.assert_close(alpha, torch.full((5, 3), 1 / 3))
    values[2].zero_()
    values[2][0, :, 0] = 1
    _, adaptive = part_proto_logits(*values)
    assert not adaptive.any()  # No valid comparison class for this region.
    _, uniform = part_proto_logits(*values, adaptive=False)
    assert uniform[0, 0] == 1


def test_alpha_is_differentiable_and_optional_detach():
    values = list(embeddings())
    values[1].requires_grad_()
    _, alpha = part_proto_logits(*values)
    (alpha * torch.tensor([1., 2., 3.])).sum().backward()
    assert values[1].grad.abs().sum() > 0
    _, detached = part_proto_logits(*values, detach_alpha=True)
    assert not detached.requires_grad


def test_zero_reliability_support_part_cannot_change_prototypes():
    sg, sp, sq, qg, qp, qq = embeddings(k=5)
    sq[:, 1, 2] = 0
    before, alpha = part_proto_logits(sg, sp, sq, qg, qp, qq)
    sp[:, 1, 2] = torch.randn_like(sp[:, 1, 2]) * 100
    after, changed_alpha = part_proto_logits(sg, sp, sq, qg, qp, qq)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(alpha, changed_alpha, rtol=0, atol=0)
