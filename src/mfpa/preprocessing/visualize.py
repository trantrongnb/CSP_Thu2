import argparse
from pathlib import Path

import cv2
import numpy as np

from .video_frames import read_video


def flow_image(flow, clip=5.0):
    u, v = flow.astype(np.float32)
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((*u.shape, 3), np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    # Fixed scale across videos, never per-video min/max.
    hsv[..., 2] = (np.clip(magnitude / clip, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Overlay anatomical LH/RH/M ROI and cached vector flow")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow-clip", type=float, default=5.)
    args = parser.parse_args(argv)
    with np.load(args.cache, allow_pickle=False) as handle:
        cache = {k: handle[k] for k in handle.files}
    t = len(cache["sampled_frame_ids"])
    # letterbox offsets/scales encode the canvas; generated baseline uses square canvases.
    original = cache["original_size"][0]
    sx, sy, left, top = cache["letterbox_transform"][0]
    size = max(round(original[0] * sx) + 2 * int(left), round(original[1] * sy) + 2 * int(top))
    # Odd padding can be one pixel larger; persist exact size in new caches.
    size = int(cache.get("canvas_size", size))
    video = read_video(args.video, t, size, bool(cache["is_mirrored"]))
    if not np.array_equal(video["sampled_frame_ids"], cache["sampled_frame_ids"]):
        raise ValueError("Video sampling differs from this cache")
    panels = []
    colors = [(0, 255, 0), (0, 128, 255), (255, 255, 0)]
    for step in range(t - 1):
        frame = cv2.cvtColor(video["rgb"][step], cv2.COLOR_RGB2BGR)
        tiles = []
        for part, name in enumerate(("LH", "RH", "M")):
            box = cache["transition_roi_box"][step, part].astype(int)
            quality = float(cache["flow_quality"][part, step])
            if quality > 0:
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), colors[part], 2)
                cv2.putText(frame, name, tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[part], 1)
            tile = cv2.resize(flow_image(cache["flow"][part, step], args.flow_clip), (size, size))
            cv2.putText(tile, f"{name} q={quality:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            tiles.append(tile)
        cv2.putText(frame, f"t={step}, frame={cache['sampled_frame_ids'][step]}", (8, size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        panels.append(np.concatenate([frame, *tiles], axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        if not cv2.imwrite(str(args.output), np.concatenate(panels, axis=0)):
            raise RuntimeError("Could not write visualization")
    else:
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 5, (4 * size, size))
        if not writer.isOpened():
            raise RuntimeError("Could not open video output")
        try:
            for panel in panels:
                writer.write(panel)
        finally:
            writer.release()
    print(f"Visualization: {args.output}")


if __name__ == "__main__":
    main()
