import pytest
import torch

from mfpa.datasets.feature_dataset import structural_features
from mfpa.smoke import make_synthetic_dataset, synthetic_cache


torch.set_num_threads(2)


@pytest.fixture
def synthetic_data(tmp_path):
    return make_synthetic_dataset(tmp_path / "data")


@pytest.fixture
def batch():
    features = structural_features(synthetic_cache())
    return {key: torch.from_numpy(value.copy()).unsqueeze(0).repeat(2, *([1] * value.ndim))
            for key, value in features.items()}
