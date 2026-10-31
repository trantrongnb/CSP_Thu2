#!/usr/bin/env python3
"""Visualize cached optical flow as HSV frames or vector-field arrows."""

import argparse
from pathlib import Path

import cv2
import numpy as np


PART_NAMES = ("LH", "RH", "M")


def flow_hsv(flow, clip):
    """Convert normalized [u, v] flow to a direction/magnitude HSV image."""
    u, v = flow.astype(np.float32)
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(magnitude / clip, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_arrows(flow, clip, stride, scale):
    """Draw a sparse vector field; arrow lengths use normalized flow units."""
    height, width = flow.shape[-2:]
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    flow = flow.astype(np.float32)
    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            u, v = flow[:, y, x]
            magnitude = float(np.hypot(u, v))
            if magnitude < 1e-4:
                continue
            length = min(magnitude, clip) * scale
            end = (round(x + u / magnitude * length), round(y + v / magnitude * length))
            cv2.arrowedLine(image, (x, y), end, (20, 40, 220), 1, tipLength=0.3)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Preprocessed video .npz cache")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 or .png contact sheet")
    parser.add_argument("--codec", choices=("avc1", "mp4v", "MJPG"), default="mp4v",
                        help="Video fourcc: avc1 (H.264, if available), mp4v, or MJPG (AVI)")
    parser.add_argument("--mode", choices=("hsv", "arrows"), default="hsv")
    parser.add_argument("--part", choices=("all", "LH", "RH", "M"), default="all")
    parser.add_argument("--flow-clip", type=float, default=5.0,
                        help="Normalized magnitude mapped to full HSV brightness (default: 5)")
    parser.add_argument("--stride", type=int, default=8, help="Arrow spacing in pixels")
    parser.add_argument("--arrow-scale", type=float, default=8.0,
                        help="Arrow pixels per normalized flow unit")
    parser.add_argument("--fps", type=float, default=5.0, help="MP4 playback frame rate")
    args = parser.parse_args(argv)
    if args.flow_clip <= 0 or args.stride <= 0 or args.arrow_scale <= 0 or args.fps <= 0:
        parser.error("flow-clip, stride, arrow-scale and fps must be positive")
    if not args.cache.is_file():
        parser.error(f"Cache not found: {args.cache}")
    suffix = args.output.suffix.lower()
    if suffix not in {".png", ".mp4", ".avi"}:
        parser.error("Output extension must be .png, .mp4, or .avi")
    if suffix == ".avi" and args.codec != "MJPG":
        parser.error("Use --codec MJPG with .avi output")
    if suffix == ".mp4" and args.codec == "MJPG":
        parser.error("MJPG is intended for .avi output; use avc1 or mp4v with .mp4")

    with np.load(args.cache, allow_pickle=False) as data:
        if "flow" not in data:
            parser.error("Cache is missing the 'flow' array")
        flow = data["flow"].astype(np.float32)
        quality = data["flow_quality"].astype(np.float32) if "flow_quality" in data else None
        sampled_ids = data["sampled_frame_ids"] if "sampled_frame_ids" in data else None

    if flow.ndim != 5 or flow.shape[0] != 3 or flow.shape[2] != 2:
        parser.error(f"Expected flow shape [3, transitions, 2, H, W], got {flow.shape}")
    if not np.isfinite(flow).all():
        parser.error("Flow array contains NaN or infinity")
    part_ids = range(3) if args.part == "all" else (PART_NAMES.index(args.part),)
    height, width = flow.shape[-2:]
    frames = []
    for step in range(flow.shape[1]):
        tiles = []
        for part in part_ids:
            tile = (flow_hsv(flow[part, step], args.flow_clip) if args.mode == "hsv"
                    else flow_arrows(flow[part, step], args.flow_clip, args.stride, args.arrow_scale))
            q = float(quality[part, step]) if quality is not None else 1.0
            frame_id = int(sampled_ids[step]) if sampled_ids is not None and step < len(sampled_ids) else step
            cv2.putText(tile, f"{PART_NAMES[part]}  t={step} frame={frame_id} q={q:.2f}",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        frame = np.concatenate(tiles, axis=1)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError(f"Expected uint8 BGR frame with 3 channels, got {frame.shape} {frame.dtype}")
        frames.append(np.ascontiguousarray(frame))

    if not frames:
        parser.error("Cache contains no flow transitions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        sheet = np.concatenate(frames, axis=0)
        if not cv2.imwrite(str(args.output), sheet):
            raise RuntimeError(f"Could not write {args.output}")
    else:
        canvas = frames[0]
        output_size = (canvas.shape[1], canvas.shape[0])
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*args.codec), args.fps,
                                 output_size)
        if not writer.isOpened():
            args.output.unlink(missing_ok=True)
            raise RuntimeError(f"Could not open output video {args.output}")
        written = 0
        try:
            for frame in frames:
                if frame.shape[:2] != (output_size[1], output_size[0]):
                    raise RuntimeError("Generated frame dimensions changed while writing video")
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise RuntimeError(f"Expected uint8 BGR frame with 3 channels, got {frame.shape} {frame.dtype}")
                writer.write(frame)
                written += 1
        finally:
            writer.release()
        if written == 0 or not args.output.is_file() or args.output.stat().st_size == 0:
            args.output.unlink(missing_ok=True)
            raise RuntimeError("Video writer produced no usable output")
    print(f"Saved {len(frames)} flow frames to {args.output} ({width}x{height} per part)")


if __name__ == "__main__":
    main()
