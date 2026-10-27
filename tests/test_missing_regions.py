import pytest
import torch

from mfpa.models import MFPAFSLR
from mfpa.models.fusion import ReliabilityFusion
from mfpa.models.temporal_pool import masked_softmax


@pytest.mark.parametrize("variant", [f"A{i}" for i in range(6)])
def test_encoder_shapes_and_missing_mouth(batch, variant):
    batch["quality"][:, 2] = 0
    batch["struct_valid"][:, 2] = False
    batch["valid_mask"][:, 2] = False
    model = MFPAFSLR(dimension=128, variant=variant)
    output = model.encode(**batch)
    assert output["parts"].shape == (2, 3, 128)
    assert output["global"].shape == (2, 128)
    assert output["part_quality"].shape == (2, 3)
    assert output["global_valid"].all()
    assert not output["parts"][:, 2].any()
    assert not output["attention"][:, 2].any()
    assert not output["part_quality"][:, 2].any()
    for value in output.values():
        assert torch.isfinite(value).all()


def test_zero_quality_removes_flow_and_matches_structural_encoder(batch):
    full = MFPAFSLR(dimension=32)
    structural = MFPAFSLR(dimension=32, variant="A0")
    structural.load_state_dict(full.state_dict())
    batch["quality"].zero_()
    output = full.encode(**batch)
    batch["flow"] = torch.randn_like(batch["flow"]) * 100
    changed = full.encode(**batch)
    only_landmarks = structural.encode(**batch)
    torch.testing.assert_close(output["global"], changed["global"], rtol=0, atol=0)
    torch.testing.assert_close(output["global"], only_landmarks["global"], rtol=0, atol=0)
    assert (output["part_quality"] > 0).all()  # Landmark evidence survives missing flow.


def test_all_invalid_no_nan_no_biased_global(batch):
    batch["quality"].zero_()
    batch["struct_valid"].zero_()
    batch["valid_mask"].zero_()
    output = MFPAFSLR(dimension=32).encode(**batch)
    assert not output["global_valid"].any()
    for key in ("global", "parts", "part_quality", "attention"):
        assert torch.isfinite(output[key]).all()
        assert not output[key].any()


def test_gate_zero_and_masked_softmax_backward():
    fusion = ReliabilityFusion(8)
    motion = torch.randn(2, 3, 4, 8, requires_grad=True)
    structure = torch.randn_like(motion, requires_grad=True)
    fused, gate = fusion(motion, structure, torch.zeros(2, 3, 4))
    assert not gate.any()
    torch.testing.assert_close(fused, fusion.norm(structure))
    scores = torch.randn(3, 4, requires_grad=True)
    mask = torch.tensor([[False] * 4, [True, False, True, False], [True] * 4])
    weights = masked_softmax(scores, mask)
    assert not weights[0].any()
    weights.square().sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_disabled_region_cannot_affect_encoder(batch):
    model = MFPAFSLR(dimension=32, regions=(0, 1))
    before = model.encode(**batch)
    batch["mouth_struct"] += 100
    batch["flow"][:, 2] += 100
    after = model.encode(**batch)
    torch.testing.assert_close(before["global"], after["global"], rtol=0, atol=0)
    assert not before["parts"][:, 2].any()
