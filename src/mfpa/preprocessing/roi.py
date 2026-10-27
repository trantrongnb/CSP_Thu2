import cv2
import numpy as np


def landmark_box(points, canvas_size, margin=1.5, minimum_size=4):
    xy = points[..., :2] * canvas_size
    low, high = xy.min(axis=0), xy.max(axis=0)
    center = (low + high) / 2
    extent = np.maximum((high - low) * margin, minimum_size)
    return np.concatenate([center - extent / 2, center + extent / 2]).astype(np.float32)


def union_box(first, second):
    return np.concatenate([np.minimum(first[:2], second[:2]), np.maximum(first[2:], second[2:])])


def valid_fraction(box, width, height):
    x1, y1, x2, y2 = box
    area = max(0., x2 - x1) * max(0., y2 - y1)
    intersection = max(0., min(x2, width) - max(x1, 0.)) * max(0., min(y2, height) - max(y1, 0.))
    return float(np.clip(intersection / max(area, 1e-8), 0, 1))


def crop_flow(field, box, roi_size=64, representation="roi_pixel_displacement"):
    """Resample an already computed full-frame field; zero outside the canvas."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("ROI must have positive extent")
    x = x1 + (np.arange(roi_size, dtype=np.float32) + 0.5) * width / roi_size - 0.5
    y = y1 + (np.arange(roi_size, dtype=np.float32) + 0.5) * height / roi_size - 0.5
    grid_x, grid_y = np.meshgrid(x, y)
    cropped = cv2.remap(field, grid_x, grid_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if representation == "roi_pixel_displacement":
        cropped = cropped * np.array([roi_size / width, roi_size / height], dtype=np.float32)
    elif representation != "canvas_pixel_displacement":
        raise ValueError(f"Unknown flow representation: {representation}")
    return cropped.transpose(2, 0, 1)
