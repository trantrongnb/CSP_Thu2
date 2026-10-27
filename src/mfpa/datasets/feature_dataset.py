import csv
import hashlib
from pathlib import Path

import numpy as np
import torch


REQUIRED_COLUMNS = {"video_id", "path", "gloss", "class_id", "language", "signer_id", "split"}


def read_metadata(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError(f"Metadata requires columns {sorted(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Metadata is empty")
    ids, paths, class_splits, gloss_splits = set(), set(), {}, {}
    for row in rows:
        vid = row["video_id"]
        if not vid or vid in ids or Path(vid).name != vid or vid in {".", ".."}:
            raise ValueError(f"Duplicate/unsafe video_id: {vid}")
        ids.add(vid)
        video_path = Path(row["path"])
        resolved = str((video_path if video_path.is_absolute() else path.parent / video_path).resolve())
        if not row["path"] or resolved in paths:
            raise ValueError(f"Repeated or empty source video path: {row['path']}")
        paths.add(resolved)
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {row['split']}")
        if not row["class_id"] or not row["gloss"].strip() or not row["language"].strip():
            raise ValueError(f"Empty class/gloss/language for {vid}")
        row["class_key"] = f"{row['language']}:{row['class_id']}"
        for table, key in ((class_splits, row["class_key"]),
                           (gloss_splits, (row["language"], row["gloss"].strip().casefold()))):
            previous = table.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Class/gloss leakage across splits: {key}")
    return rows


def structural_features(cache):
    hands, mouth = cache["hand_lm"].astype(np.float32), cache["mouth_lm"].astype(np.float32)
    hv, mv = cache["hand_valid"].astype(bool), cache["mouth_valid"].astype(bool)
    hc, mc = cache["hand_center"].astype(np.float32), cache["mouth_center"].astype(np.float32)
    transition = cache["transition_valid"].astype(bool)
    # A fixed coordinate rule; no validation/test-derived normalization statistics.
    hand_scale = np.maximum(np.ptp(hands[..., :2], axis=2).max(axis=-1), 1e-6)
    hand_relative = (hands - hands[:, :, :1]) / hand_scale[..., None, None]
    mouth_scale = np.maximum(np.ptp(mouth[..., :2], axis=1).max(axis=-1), 1e-6)
    mouth_relative = (mouth - mouth.mean(axis=1, keepdims=True)) / mouth_scale[:, None, None]
    hand_delta = np.diff(hc, axis=0) * (hv[:-1] & hv[1:])[..., None]
    mouth_delta = np.diff(mc, axis=0) * (mv[:-1] & mv[1:])[:, None]
    hand_struct = np.concatenate([hand_relative[:-1].reshape(len(transition), 2, 63), hc[:-1], hand_delta], axis=-1)
    mouth_struct = np.concatenate([mouth_relative[:-1].reshape(len(transition), 120), mc[:-1], mouth_delta], axis=-1)
    hand_struct *= (hv[:-1] & transition[:, None])[..., None]
    mouth_struct *= (mv[:-1] & transition)[:, None]
    structural_valid = np.concatenate([hv[:-1], mv[:-1, None]], axis=1).T & transition[None, :]
    quality = cache["flow_quality"].astype(np.float32) * transition[None, :]
    flow = cache["flow"].astype(np.float32)
    flow = np.where((quality > 0)[..., None, None, None], flow, 0.0)
    return {"flow": flow, "hand_struct": hand_struct.transpose(1, 0, 2),
            "mouth_struct": mouth_struct, "quality": quality,
            "valid_mask": structural_valid | (quality > 0), "struct_valid": structural_valid,
            "transition_valid": transition}


def validate_cache(cache, video_id):
    required = {"flow", "hand_lm", "mouth_lm", "hand_center", "mouth_center", "flow_quality",
                "hand_valid", "mouth_valid", "transition_valid", "sampled_frame_ids", "config_hash"}
    if not required <= set(cache):
        raise ValueError(f"{video_id}: missing cache keys {sorted(required - set(cache))}")
    t = len(cache["sampled_frame_ids"])
    flow_shape = cache["flow"].shape
    if t < 2 or len(flow_shape) != 5 or flow_shape[:3] != (3, t - 1, 2) or flow_shape[3] != flow_shape[4]:
        raise ValueError(f"{video_id}: invalid flow shape {flow_shape}")
    expected = {"hand_lm": (t, 2, 21, 3), "mouth_lm": (t, 40, 3), "hand_center": (t, 2, 2),
                "mouth_center": (t, 2), "flow_quality": (3, t - 1), "hand_valid": (t, 2),
                "mouth_valid": (t,), "transition_valid": (t - 1,)}
    for key, shape in expected.items():
        if cache[key].shape != shape:
            raise ValueError(f"{video_id}: {key} expected {shape}, got {cache[key].shape}")
    for key in ("flow", "hand_lm", "mouth_lm", "hand_center", "mouth_center", "flow_quality"):
        if not np.isfinite(cache[key]).all():
            raise ValueError(f"{video_id}: nonfinite {key}")
    for key in ("hand_valid", "mouth_valid", "transition_valid"):
        if not np.isin(cache[key], [0, 1]).all():
            raise ValueError(f"{video_id}: {key} must be binary")
    if not ((cache["flow_quality"] >= 0) & (cache["flow_quality"] <= 1)).all():
        raise ValueError(f"{video_id}: quality outside [0,1]")
    ids = cache["sampled_frame_ids"]
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (np.diff(ids) < 0).any():
        raise ValueError(f"{video_id}: invalid chronological sampled_frame_ids")
    if not np.array_equal(cache["transition_valid"].astype(bool), np.diff(ids) > 0):
        raise ValueError(f"{video_id}: transition mask disagrees with frame IDs")
    if (cache["flow_quality"][:, ~cache["transition_valid"].astype(bool)] != 0).any():
        raise ValueError(f"{video_id}: padded flow transition has nonzero quality")


class FeatureDataset:
    """Class-disjoint metadata and NPZ cache, with explicit invalid-sample failure."""

    def __init__(self, metadata, cache_dir, split, variant="A5", regions=(0, 1, 2)):
        self.metadata, self.cache_dir, self.split = Path(metadata), Path(cache_dir), split
        self.rows = [r for r in read_metadata(self.metadata) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No samples for split {split}")
        self.by_id = {r["video_id"]: r for r in self.rows}
        self.class_to_videos = {}
        self.config_hash = None
        shape = None
        for row in self.rows:
            vid = row["video_id"]
            cache = self._load(vid)
            validate_cache(cache, vid)
            signature = str(cache["config_hash"].item())
            if self.config_hash is None:
                self.config_hash = signature
            if signature != self.config_hash:
                raise ValueError(f"Mixed preprocessing configurations in cache: {vid}")
            if shape is None:
                shape = cache["flow"].shape
            if cache["flow"].shape != shape:
                raise ValueError(f"Mixed flow dimensions in cache: {vid}")
            inputs = structural_features(cache)
            usable = (inputs["struct_valid"] if variant == "A0" else
                      inputs["quality"] > 0 if variant == "A1" else inputs["valid_mask"])
            if not usable[list(regions)].any():
                raise ValueError(f"{vid}: all selected regions invalid for {variant}; "
                                 "report this failed sample and explicitly revise metadata/protocol")
            self.class_to_videos.setdefault(row["class_key"], []).append(vid)
        self.fingerprint = hashlib.sha256(self.metadata.read_bytes() + self.config_hash.encode()).hexdigest()

    def _load(self, video_id):
        with np.load(self.cache_dir / f"{video_id}.npz", allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        vid = self.rows[index]["video_id"] if isinstance(index, int) else index
        if vid not in self.by_id:
            raise KeyError(f"Video {vid} is outside split {self.split}")
        return {key: torch.from_numpy(np.ascontiguousarray(value))
                for key, value in structural_features(self._load(vid)).items()}
