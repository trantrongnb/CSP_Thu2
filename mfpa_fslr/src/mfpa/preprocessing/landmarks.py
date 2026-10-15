import contextlib
import numpy as np
import sys
import threading
from importlib.metadata import version
from pathlib import Path
import urllib.request


# Outer contour (20) followed by inner contour (20), fixed order.
MOUTH_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)

# Official MediaPipe Holistic Landmarker bundle. Override with
# MFPA_HOLISTIC_MODEL or the pipeline's --landmark-model option.
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
_MODEL_LOCK = threading.Lock()


@contextlib.contextmanager
def _hide_feedback_manager_warning():
    """Filter one benign native MediaPipe warning while preserving other stderr."""
    import os
    import threading

    class FilteredFd:
        def __init__(self, fd, target):
            self.fd, self.target = fd, target
            self.buffer = b""
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self._pump, daemon=True)

        def _pump(self):
            while not self.stop.is_set():
                try:
                    chunk = os.read(self.fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self.buffer += chunk
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    if b"inference_feedback_manager.cc" not in line:
                        os.write(self.target, line + b"\n")
            if self.buffer and b"inference_feedback_manager.cc" not in self.buffer:
                os.write(self.target, self.buffer)

    original = os.dup(2)
    read_fd, write_fd = os.pipe()
    filtered = FilteredFd(read_fd, original)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    filtered.thread.start()
    try:
        yield
    finally:
        os.dup2(original, 2)
        # Closing the saved target copy after draining is safe; stderr now points
        # to the original process descriptor again.
        filtered.thread.join(timeout=2)
        os.close(read_fd)
        os.close(original)


def resolve_holistic_model(model_path=None):
    """Return a HolisticLandmarker task file, downloading the official bundle if needed."""
    with _MODEL_LOCK:
        configured = model_path or __import__("os").environ.get("MFPA_HOLISTIC_MODEL")
        target = (Path(configured).expanduser() if configured else
                  Path(__file__).resolve().parents[3] / "data/models/holistic_landmarker.task")
        if target.is_file() and target.stat().st_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            print(f"Downloading MediaPipe Holistic model to {target}", flush=True)
            urllib.request.urlretrieve(HOLISTIC_MODEL_URL, temporary)
            if not temporary.is_file() or temporary.stat().st_size < 1_000_000:
                raise RuntimeError("Downloaded Holistic model is missing or unexpectedly small")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def extract_landmarks(rgb, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5,
                      model_path=None):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Install mfpa-fslr[landmarks] (MediaPipe 0.10.30 or newer)") from exc
    t = len(rgb)
    hands, mouth = np.zeros((t, 2, 21, 3), np.float32), np.zeros((t, 40, 3), np.float32)
    hv, mv = np.zeros((t, 2), bool), np.zeros(t, bool)

    legacy = getattr(getattr(mp, "solutions", None), "holistic", None)
    if hasattr(legacy, "Holistic"):
        with _hide_feedback_manager_warning(), legacy.Holistic(
            static_image_mode=False, model_complexity=model_complexity, refine_face_landmarks=False,
            min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence,
        ) as detector:
            for frame, image in enumerate(rgb):
                result = detector.process(image)
                _copy_result(result, frame, hands, mouth, hv, mv, legacy=True)
    else:
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolve_holistic_model(model_path))),
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=detection_confidence,
            min_face_landmarks_confidence=detection_confidence,
            min_hand_landmarks_confidence=detection_confidence,
        )
        with _hide_feedback_manager_warning(), vision.HolisticLandmarker.create_from_options(options) as detector:
            for frame, image in enumerate(rgb):
                task_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
                result = detector.detect(task_image)
                _copy_result(result, frame, hands, mouth, hv, mv, legacy=False)
    hands[~hv], mouth[~mv] = 0, 0
    return {"hand_lm": hands, "mouth_lm": mouth, "hand_valid": hv, "mouth_valid": mv,
            "hand_center": hands[..., :2].mean(axis=2), "mouth_center": mouth[..., :2].mean(axis=1),
            "mediapipe_version": np.array(getattr(mp, "__version__", version("mediapipe")))}


def _copy_result(result, frame, hands, mouth, hand_valid, mouth_valid, *, legacy):
    hand_points = (result.left_hand_landmarks, result.right_hand_landmarks)
    for part, points in enumerate(hand_points):
        if points:
            if legacy:
                values = [(point.x, point.y, point.z) for point in points.landmark]
            else:
                values = [(point.x, point.y, point.z) for point in points]
            hands[frame, part] = values
            hand_valid[frame, part] = np.isfinite(hands[frame, part]).all()
    face_points = result.face_landmarks
    if face_points:
        if legacy:
            mouth[frame] = [(face_points.landmark[i].x, face_points.landmark[i].y,
                             face_points.landmark[i].z) for i in MOUTH_INDICES]
        else:
            mouth[frame] = [(face_points[i].x, face_points[i].y, face_points[i].z)
                            for i in MOUTH_INDICES]
        mouth_valid[frame] = np.isfinite(mouth[frame]).all()
