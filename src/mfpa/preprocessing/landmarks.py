import numpy as np


# Outer contour (20) followed by inner contour (20), fixed order.
MOUTH_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)


def extract_landmarks(rgb, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Install mfpa-fslr[landmarks] (MediaPipe 0.10.21)") from exc
    if not hasattr(mp, "solutions"):
        raise RuntimeError("This backend requires mediapipe==0.10.21 with solutions.holistic")
    t = len(rgb)
    hands, mouth = np.zeros((t, 2, 21, 3), np.float32), np.zeros((t, 40, 3), np.float32)
    hv, mv = np.zeros((t, 2), bool), np.zeros(t, bool)
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=model_complexity, refine_face_landmarks=False,
        min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence,
    ) as detector:
        for frame, image in enumerate(rgb):
            result = detector.process(image)
            for part, points in enumerate((result.left_hand_landmarks, result.right_hand_landmarks)):
                if points is not None:
                    hands[frame, part] = [(p.x, p.y, p.z) for p in points.landmark]
                    hv[frame, part] = np.isfinite(hands[frame, part]).all()
            if result.face_landmarks is not None:
                mouth[frame] = [(result.face_landmarks.landmark[i].x,
                                 result.face_landmarks.landmark[i].y,
                                 result.face_landmarks.landmark[i].z) for i in MOUTH_INDICES]
                mv[frame] = np.isfinite(mouth[frame]).all()
    hands[~hv], mouth[~mv] = 0, 0
    return {"hand_lm": hands, "mouth_lm": mouth, "hand_valid": hv, "mouth_valid": mv,
            "hand_center": hands[..., :2].mean(axis=2), "mouth_center": mouth[..., :2].mean(axis=1),
            "mediapipe_version": np.array(mp.__version__)}
