import csv
import json

import pytest

from mfpa.datasets.feature_dataset import FeatureDataset, read_metadata
from mfpa.datasets.episode_sampler import EpisodeSampler, load_episodes


def test_deterministic_episodes_no_leakage_and_remap(synthetic_data, tmp_path):
    metadata, cache = synthetic_data
    dataset = FeatureDataset(metadata, cache, "test")
    first = EpisodeSampler(dataset.class_to_videos, 3, 1, 2, 42)
    second = EpisodeSampler(dataset.class_to_videos, 3, 1, 2, 42)
    for _ in range(10):
        episode = first.sample()
        assert episode == second.sample()
        assert not set(episode["support"]) & set(episode["query"])
        for vid, y in zip(episode["query"], episode["query_labels"]):
            assert dataset.by_id[vid]["class_key"] == episode["classes"][y]
    path = tmp_path / "episodes.json"
    first.save(path, 2, dataset)
    document = load_episodes(path, dataset)
    document["episodes"][0]["query"][0] = document["episodes"][0]["support"][0]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        load_episodes(path, dataset)


def test_insufficient_samples_are_not_replaced():
    with pytest.raises(ValueError, match="distinct videos"):
        EpisodeSampler({"a": ["1", "2"], "b": ["3", "4"]}, 2, 1, 2)


def test_metadata_class_and_gloss_leakage(synthetic_data):
    metadata, _ = synthetic_data
    with metadata.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["split"] = "test"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        read_metadata(metadata)


def test_all_invalid_cache_is_rejected(synthetic_data):
    import numpy as np
    metadata, cache = synthetic_data
    path = cache / "train_0_0.npz"
    with np.load(path) as handle:
        values = {key: handle[key] for key in handle.files}
    values["hand_valid"][:] = False
    values["mouth_valid"][:] = False
    values["flow_quality"][:] = 0
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="all selected regions invalid"):
        FeatureDataset(metadata, cache, "train")
