import numpy as np

from mfpa.datasets.feature_dataset import structural_features, validate_cache
from mfpa.preprocessing.landmarks import MOUTH_INDICES
from mfpa.smoke import synthetic_cache


def test_structural_alignment_and_shape():
    cache = synthetic_cache(frames=16, roi_size=64)
    validate_cache(cache, "fixture")
    inputs = structural_features(cache)
    assert inputs["hand_struct"].shape == (2, 15, 67)
    assert inputs["mouth_struct"].shape == (15, 124)
    assert len(MOUTH_INDICES) == len(set(MOUTH_INDICES)) == 40
    np.testing.assert_allclose(inputs["hand_struct"][0, :, 63:65], cache["hand_center"][:-1, 0])
    np.testing.assert_allclose(inputs["hand_struct"][0, :, 65:], np.diff(cache["hand_center"][:, 0], axis=0))
    np.testing.assert_allclose(inputs["hand_struct"][:, :, :3], 0)


def test_missing_next_landmark_removes_velocity_only():
    cache = synthetic_cache()
    cache["hand_valid"][1, 0] = False
    cache["hand_lm"][1, 0] = 0
    cache["hand_center"][1, 0] = 0
    cache["flow_quality"][0, :2] = 0
    inputs = structural_features(cache)
    np.testing.assert_array_equal(inputs["hand_struct"][0, 0, 65:], 0)
    assert inputs["struct_valid"][0, 0]
    assert not inputs["hand_struct"][0, 1].any()
