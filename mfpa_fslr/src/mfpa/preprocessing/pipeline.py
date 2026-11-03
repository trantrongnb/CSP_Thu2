import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .landmarks import MOUTH_INDICES, extract_landmarks
from .optical_flow import DisFlow, FarnebackFlow, RaftSmallFlow
from .roi import landmark_box, union_box, valid_fraction, crop_flow
from .video_frames import read_video
from ..datasets.feature_dataset import read_metadata


def progress(index, total, label, started):
    width = 28
    filled = int(width * index / max(total, 1))
    elapsed = time.perf_counter() - started
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    sys.stdout.write(f"\r[{bar}] {index:>5}/{total:<5} {100 * index / max(total, 1):5.1f}% "
                     f"{label[:28]:<28} ETA {eta / 60:5.1f}m")
    sys.stdout.flush()
    if index >= total:
        sys.stdout.write("\n")


DEFAULTS = {"target_frames": 8, "canvas_size": 256, "roi_size": 64,
            "flow_estimator": "dis", "flow_scale_resolution": 0.5, "flow_gap": 1,
            "flow_pairing": "midpoint_raw", "flow_temporal_normalization": "none",
            "flow_representation": "roi_pixel_displacement",
            "flow_scale": 20.0, "flow_clip": 5.0, "cache_dtype": "float16",
            "hand_margin": 1.5, "mouth_margin": 1.8, "model_complexity": 1,
            "detection_confidence": 0.5, "tracking_confidence": 0.5,
            "farneback": {}, "num_flow_updates": 12}


def preprocessing_signature(config, landmark_model=None):
    from importlib.metadata import version
    import os
    import mediapipe as mp

    model_path = landmark_model or os.environ.get("MFPA_HOLISTIC_MODEL")
    legacy_holistic = getattr(getattr(mp, "solutions", None), "holistic", None)
    if not hasattr(legacy_holistic, "Holistic"):
        if model_path is None:
            model_path = Path(__file__).resolve().parents[3] / "data/models/holistic_landmarker.task"
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            # The model is downloaded by extraction. Keep the stable source URL
            # in the signature before it exists locally.
            from .landmarks import HOLISTIC_MODEL_URL
            model_identity = HOLISTIC_MODEL_URL
        else:
            model_identity = hashlib.sha256(model_path.read_bytes()).hexdigest()
    else:
        model_identity = "legacy-solutions-holistic"
    payload = {"schema_version": 2, "config": config, "mouth_indices": MOUTH_INDICES,
               "opencv_version": cv2.__version__, "numpy_version": np.__version__,
               "mediapipe_version": version("mediapipe"), "landmark_model": model_identity}
    if config["flow_estimator"] == "raft_small":
        payload.update(torch_version=version("torch"), torchvision_version=version("torchvision"),
                       raft_weights="Raft_Small_Weights.C_T_V2")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def progress(index, total, label, started):
    width = 28
    filled = int(width * index / max(total, 1))
    elapsed = time.perf_counter() - started
    rate = index / elapsed if elapsed > 0 else 0.0
    eta = (total - index) / rate if rate > 0 else 0.0
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    sys.stdout.write(f"\r[{bar}] {index:>5}/{total:<5} {100 * index / max(total, 1):5.1f}% "
                     f"{label[:26]:<26} ETA {eta / 60:5.1f}m")
    sys.stdout.flush()
    if index >= total:
        sys.stdout.write("\n")


def pack_features(video, landmarks, estimator, config):
    rgb, size = video["rgb"], config["canvas_size"]
    t, out = len(rgb), config["roi_size"]
    valid = np.concatenate([landmarks["hand_valid"], landmarks["mouth_valid"][:, None]], axis=1)
    boxes = np.zeros((t, 3, 4), np.float32)
    for frame in range(t):
        for part in range(3):
            if valid[frame, part]:
                points = landmarks["hand_lm"][frame, part] if part < 2 else landmarks["mouth_lm"][frame]
                boxes[frame, part] = landmark_box(points, size, config["hand_margin"] if part < 2 else config["mouth_margin"])
    flow = np.zeros((3, t - 1, 2, out, out), np.float32)
    quality = np.zeros((3, t - 1), np.float32)
    quality_vector = np.zeros((3, t - 1, 6), np.float32)
    transition_boxes = np.zeros((t - 1, 3, 4), np.float32)
    for step in range(t - 1):
        if not video["transition_valid"][step]:
            continue
        pair_valid = valid[step] & valid[step + 1]
        if not pair_valid.any():
            continue
        pair = video.get("flow_rgb")
        first, second = (pair[step] if pair is not None and config.get("flow_pairing") == "midpoint_raw"
                         else (rgb[step], rgb[step + 1]))
        field = estimator(first, second)
        backward = estimator(second, first)
        if config.get("flow_temporal_normalization", "none") == "sampled_gap":
            raw_gap = max(int(video["sampled_frame_ids"][step + 1] - video["sampled_frame_ids"][step]), 1)
            factor = raw_gap / max(config.get("flow_gap", 1), 1)
            field *= factor
            backward *= factor
        if field.shape != (size, size, 2) or not np.isfinite(field).all():
            raise ValueError("Flow estimator returned invalid field")
        for part in np.flatnonzero(pair_valid):
            box = union_box(boxes[step, part], boxes[step + 1, part])
            transition_boxes[step, part] = box
            quality[part, step] = valid_fraction(box, size, size)
            if quality[part, step] > 0:
                flow[part, step] = crop_flow(field, box, out, config["flow_representation"])
                x0, y0, x1, y1 = np.clip(np.rint(box).astype(int), 0, size)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                if xx.size:
                    f = field[y0:y1, x0:x1]
                    bx = np.clip(np.rint(xx + f[..., 0]).astype(int), 0, size - 1)
                    by = np.clip(np.rint(yy + f[..., 1]).astype(int), 0, size - 1)
                    err = np.linalg.norm(f + backward[by, bx], axis=-1)
                    quality_vector[part, step, 0] = np.mean(err < 1.5 + 0.05 * np.linalg.norm(f, axis=-1))
                    p0 = landmarks["hand_lm"][step, part, :, :2] if part < 2 else landmarks["mouth_lm"][step, :, :2]
                    p1 = landmarks["hand_lm"][step + 1, part, :, :2] if part < 2 else landmarks["mouth_lm"][step + 1, :, :2]
                    raw_gap = max(int(video["sampled_frame_ids"][step + 1] - video["sampled_frame_ids"][step]), 1)
                    speed = size * np.median(np.linalg.norm(p1 - p0, axis=-1))
                    if config.get("flow_temporal_normalization", "none") == "none":
                        speed *= config.get("flow_gap", 1) / raw_gap
                    quality_vector[part, step, 1] = np.exp(-abs(np.median(np.linalg.norm(f, axis=-1)) - speed) / 12.0)
                    roi_image = cv2.cvtColor(first[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                    if roi_image.size:
                        gx = cv2.Sobel(roi_image, cv2.CV_32F, 1, 0)
                        gy = cv2.Sobel(roi_image, cv2.CV_32F, 0, 1)
                        quality_vector[part, step, 2] = min(1.0, float(np.mean(gx*gx + gy*gy)) / 3000)
                        quality_vector[part, step, 3] = min(1.0, float(np.mean(cv2.Laplacian(roi_image, cv2.CV_32F)**2)) / 800)
                    quality_vector[part, step, 4] = min(1.0, float((x1-x0)*(y1-y0)) / (size*size))
                    if part < 2:
                        s0 = np.linalg.norm(landmarks["hand_lm"][step, part, 9] - landmarks["hand_lm"][step, part, 0])
                        s1 = np.linalg.norm(landmarks["hand_lm"][step + 1, part, 9] - landmarks["hand_lm"][step + 1, part, 0])
                        ratio = max(s0, s1, 1e-5) / max(min(s0, s1), 1e-5)
                        quality_vector[part, step, 5] = float(np.exp(-abs(np.log(ratio)) / 0.25))
                    else:
                        quality_vector[part, step, 5] = 1.0
    flow = np.clip(flow / config["flow_scale"], -config["flow_clip"], config["flow_clip"])
    result = {k: v for k, v in video.items() if k not in {"rgb", "flow_rgb"}}
    result.update(landmarks)
    result.update(flow=flow.astype(config["cache_dtype"]), flow_quality=quality,
                  flow_quality_vector=quality_vector,
                  flow_valid=quality > 0, roi_box=boxes, transition_roi_box=transition_boxes,
                  canvas_size=np.array(size))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Precompute frozen flow and anatomical landmarks")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--landmark-model", type=Path,
                        help="Holistic .task model used when the installed MediaPipe lacks solutions.holistic")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of videos to preprocess concurrently")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = {**DEFAULTS, **(yaml.safe_load(args.config.read_text()) or {})}
    if config["target_frames"] < 2 or config["flow_scale"] <= 0 or config["flow_clip"] <= 0:
        raise ValueError("Invalid sampling or flow normalization config")
    if config["cache_dtype"] not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be float16 or float32")
    if config["flow_estimator"] not in {"dis", "farneback", "raft_small"}:
        raise ValueError("flow_estimator must be dis, farneback or raft_small")
    if config["flow_clip"] > np.finfo(config["cache_dtype"]).max:
        raise ValueError("flow_clip exceeds cache dtype range")
    signature, provenance = preprocessing_signature(config, args.landmark_model)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "preprocessing_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["config_hash"] != signature:
        raise ValueError("Output cache uses a different configuration; choose a new --output")
    manifest.write_text(json.dumps({"config_hash": signature, **provenance}, indent=2))
    (args.output / "mouth_indices.json").write_text(json.dumps(MOUTH_INDICES))
    if config["flow_estimator"] == "raft_small":
        import torch
        torch.hub.set_dir(str(args.output / ".torch_hub"))
    rows = read_metadata(args.metadata)
    rows = rows[:args.limit] if args.limit is not None else rows
    progress_start = time.perf_counter()
    progress_lock = threading.Lock()

    def process_row(item):
        index, row = item
        path = Path(row["path"])
        if not path.is_absolute():
            path = args.metadata.parent / path
        mirror_value = row.get("is_mirrored", "false").strip().lower()
        if mirror_value not in {"", "0", "false", "1", "true"}:
            raise ValueError(f"Invalid is_mirrored for {row['video_id']}")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_path = args.output / f"{row['video_id']}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if (str(cached["config_hash"].item()) != signature or
                    str(cached["source_hash"].item()) != source_hash or
                    bool(cached["is_mirrored"].item()) != (mirror_value in {"1", "true"})):
                    raise ValueError(f"Stale cache: {cache_path}; use a new output directory")
            with progress_lock:
                progress(index, len(rows), f"cached {row['video_id']}", progress_start)
            return
        start = time.perf_counter()
        estimator = (DisFlow(config.get("dis_preset", "medium"), config["flow_scale_resolution"])
                     if config["flow_estimator"] == "dis" else
                     FarnebackFlow(**config["farneback"]) if config["flow_estimator"] == "farneback" else
                     RaftSmallFlow(args.device, config["num_flow_updates"]))
        video = read_video(path, config["target_frames"], config["canvas_size"],
                           mirror_value in {"1", "true"}, config["flow_gap"],
                           config.get("flow_pairing", "midpoint_raw"))
        landmarks = extract_landmarks(video["rgb"], config["model_complexity"],
                                      config["detection_confidence"], config["tracking_confidence"],
                                      args.landmark_model)
        features = pack_features(video, landmarks, estimator, config)
        features.update(config_hash=np.array(signature), source_hash=np.array(source_hash),
                        preprocessing_seconds=np.array(time.perf_counter() - start))
        # Atomic publish, so interrupted preprocessing never leaves a partial cache.
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **features)
        temporary.replace(cache_path)
        with progress_lock:
            progress(index, len(rows),
                     f"{row['video_id']} q={features['flow_quality'].mean():.2f}", progress_start)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_row, item) for item in enumerate(rows, 1)]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
