import torch

from mfpa.datasets.feature_dataset import FeatureDataset
from mfpa.datasets.episode_sampler import EpisodeSampler
from mfpa.engine import score_episode
from mfpa.models import MFPAFSLR


def test_episode_gradients_reach_all_trainable_branches(synthetic_data):
    metadata, cache = synthetic_data
    dataset = FeatureDataset(metadata, cache, "train")
    episode = EpisodeSampler(dataset.class_to_videos, 3, 1, 2, 42).sample()
    model = MFPAFSLR(dimension=32)
    logits, _, loss, _ = score_episode(model, dataset, episode, "cpu", {})
    assert torch.isfinite(logits).all()
    loss.backward()
    for module in (model.motion, model.structure.hand, model.structure.mouth, model.fusion.gate,
                   model.pool, model.global_projection):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
