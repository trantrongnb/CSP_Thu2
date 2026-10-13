import numpy as np
import pytest

from mfpa.preprocessing.roi import crop_flow, valid_fraction
from mfpa.preprocessing.video_frames import sample_frame_ids, letterbox
from mfpa.preprocessing.optical_flow import FarnebackFlow
from mfpa.preprocessing.pipeline import pack_features, DEFAULTS
from mfpa.smoke import synthetic_cache


def test_roi_displacement_scale_and_canvas_units():
    field = np.full((64, 64, 2), [4., -2.], np.float32)
    box = np.array([16., 8., 48., 24.], np.float32)
    roi = crop_flow(field, box, 16)
    np.testing.assert_allclose(roi[0], 2.)
    np.testing.assert_allclose(roi[1], -2.)
    canvas = crop_flow(field, box, 16, "canvas_pixel_displacement")
    np.testing.assert_allclose(canvas[0], 4.)
    np.testing.assert_allclose(canvas[1], -2.)


def test_roi_fraction_and_outside_canvas():
    assert valid_fraction([-10, 0, 10, 20], 64, 64) == .5
    assert valid_fraction([80, 0, 100, 20], 64, 64) == 0
    field = np.ones((64, 64, 2), np.float32)
    assert not crop_flow(field, [80, 0, 100, 20]).any()


def test_short_clip_padding_and_letterbox():
    ids, valid = sample_frame_ids(3, 16)
    assert ids.tolist() == [0, 1, 2] + [2] * 13
    assert valid.sum() == 2
    image = np.full((50, 100, 3), 255, np.uint8)
    canvas, transform = letterbox(image, 256)
    assert canvas.shape == (256, 256, 3)
    np.testing.assert_allclose(transform, [2.56, 2.56, 0, 64])
    assert not canvas[:64].any()
    with pytest.raises(ValueError):
        sample_frame_ids(0)


def test_farneback_translation_direction():
    import cv2
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 255, (128, 128), np.uint8)
    first = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    second = cv2.warpAffine(first, np.float32([[1, 0, 3], [0, 1, 0]]), (128, 128))
    flow = FarnebackFlow()(first, second)
    np.testing.assert_allclose(np.median(flow[20:-20, 20:-20], axis=(0, 1)), [3, 0], atol=.25)


def test_pipeline_flow_on_full_frames_and_padded_masks():
    cache = synthetic_cache(frames=4)
    cache["sampled_frame_ids"] = np.array([0, 1, 2, 2])
    cache["transition_valid"][-1] = False
    video = {"rgb": np.zeros((4, 64, 64, 3), np.uint8),
             "transition_valid": cache["transition_valid"], "sampled_frame_ids": cache["sampled_frame_ids"]}
    landmarks = {key: cache[key] for key in ("hand_lm", "mouth_lm", "hand_valid", "mouth_valid", "hand_center", "mouth_center")}
    calls = []
    def estimator(a, b):
        calls.append((a.shape, b.shape))
        return np.ones((64, 64, 2), np.float32)
    output = pack_features(video, landmarks, estimator, {**DEFAULTS, "canvas_size": 64, "roi_size": 16})
    assert calls == [((64, 64, 3), (64, 64, 3))] * 2
    assert output["flow"].shape == (3, 3, 2, 16, 16)
    assert not output["flow"][:, -1].any()
    assert not output["flow_quality"][:, -1].any()
