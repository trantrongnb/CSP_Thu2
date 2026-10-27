import cv2
import numpy as np


def sample_frame_ids(count, target_frames=16):
    if count < 1 or target_frames < 2:
        raise ValueError("Need a nonempty video and at least two sampled frames")
    if count >= target_frames:
        ids = np.linspace(0, count - 1, target_frames).astype(np.int64)
    else:
        ids = np.concatenate([np.arange(count), np.full(target_frames - count, count - 1)])
    return ids, (np.diff(ids) > 0)


def letterbox(rgb, size=256):
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh))
    # Actual x/y scales include integer resize rounding.
    return canvas, np.array([nw / w, nh / h, left, top], dtype=np.float32)


def read_video(path, target_frames=16, canvas_size=256, is_mirrored=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps, frames = cap.get(cv2.CAP_PROP_FPS), []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    ids, valid = sample_frame_ids(len(frames), target_frames)
    rgb, transforms, sizes = [], [], []
    for index in ids:
        frame = frames[index]
        if is_mirrored:
            frame = frame[:, ::-1].copy()  # Unmirror before anatomical LH/RH detection.
        canvas, transform = letterbox(frame, canvas_size)
        rgb.append(canvas)
        transforms.append(transform)
        sizes.append([frame.shape[1], frame.shape[0]])
    return {"rgb": np.stack(rgb), "sampled_frame_ids": ids,
            "transition_valid": valid, "original_fps": np.array(fps),
            "original_size": np.asarray(sizes), "letterbox_transform": np.stack(transforms),
            "is_mirrored": np.array(is_mirrored), "decoded_frame_count": np.array(len(frames))}
