"""Prepare the local WLASL dataset required by a training configuration.

The script is safe to rerun: preprocessing reuses matching caches, while the
audit and protocol outputs are keyed by their input hashes and never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import os
import time
from pathlib import Path

import yaml

# Also support direct invocation outside run_conda.sh.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfpa.datasets.feature_dataset import read_metadata


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_metadata_rows(dataset_root: Path, seed: int = 42) -> list[dict[str, str]]:
    """Create alphabetically split, class-disjoint metadata from gloss folders."""
    dataset_root = dataset_root.resolve()
    gloss_dirs = []
    for gloss_dir in sorted(dataset_root.iterdir(), key=lambda path: path.name):
        if not gloss_dir.is_dir():
            continue
        videos = sorted((path for path in gloss_dir.iterdir()
                         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
                        key=lambda path: path.name)
        if videos:
            gloss_dirs.append((gloss_dir.name, videos))
    if not gloss_dirs:
        raise ValueError(f"No video files found under {dataset_root}")

    classes = sorted(gloss for gloss, _ in gloss_dirs)
    class_ids = {gloss: str(index) for index, gloss in enumerate(classes)}
    train_end, val_end = int(len(classes) * 0.6), int(len(classes) * 0.8)
    split_by_gloss = {gloss: ("train" if index < train_end else "val" if index < val_end else "test")
                      for index, gloss in enumerate(classes)}

    rows = []
    for gloss, videos in gloss_dirs:
        safe_gloss = re.sub(r"[^A-Za-z0-9_.-]+", "_", gloss.strip()).strip("._") or "video"
        for video in videos:
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem.strip()).strip("._") or "video"
            rows.append({"video_id": f"{safe_gloss}_{safe_stem}", "path": str(video.resolve()),
                         "gloss": gloss, "class_id": class_ids[gloss], "language": "ASL",
                         "signer_id": "", "split": split_by_gloss[gloss], "is_mirrored": "false"})
    return rows


def write_metadata_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["video_id", "path", "gloss", "class_id", "language", "signer_id", "split", "is_mirrored"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage(index: int, label: str) -> None:
    print(f"\n[{index}/4] {label}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "WLASL")
    parser.add_argument("--preprocess-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/preprocess.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    overall_start = time.perf_counter()
    package_dir = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    cache_dir = (package_dir / config["data"]["cache_dir"]).resolve()
    prep_metadata = package_dir / "data/wlasl_preprocess.csv"
    prepared_dir = package_dir / "data/wlasl_prepared_60_20_20"
    if args.workers < 1:
        parser.error("--workers must be positive")

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Raw WLASL dataset not found at {args.dataset_root}. "
            "Pass --dataset-root or place the gloss/video folders there."
        )
    prep_metadata.parent.mkdir(parents=True, exist_ok=True)
    stage(1, "Build or validate WLASL metadata")
    if not prep_metadata.exists():
        started = time.perf_counter()
        rows = build_metadata_rows(args.dataset_root)
        write_metadata_csv(rows, prep_metadata)
        print(f"Created metadata for {len(rows)} videos in {time.perf_counter() - started:.1f}s", flush=True)
    else:
        # Refuse silently using metadata created from a different raw dataset.
        rows = read_metadata(prep_metadata)
        listed = {Path(row["path"]).resolve() for row in rows}
        absent = [path for path in listed if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"Metadata references missing source videos, e.g. {absent[0]}")

    read_metadata(prep_metadata)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage(2, "Preprocess video frames and landmarks")
    preprocess_command = [sys.executable, "-m", "mfpa.preprocessing.pipeline",
                          "--metadata", str(prep_metadata), "--output", str(cache_dir),
                          "--config", str(args.preprocess_config.resolve()),
                          "--workers", str(args.workers)]
    if model_path := os.environ.get("MFPA_HOLISTIC_MODEL"):
        preprocess_command.extend(["--landmark-model", model_path])
    run(preprocess_command)

    metadata_hash = sha256(prep_metadata)
    manifest_path = cache_dir / "preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit_key = f"{metadata_hash}_{manifest['config_hash']}"
    audit_path = package_dir / "outputs/wlasl_training_readiness" / f"report_{audit_key}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    stage(3, "Audit prepared video caches")
    if not audit_path.exists():
        run([sys.executable, str(package_dir / "scripts/audit_wlasl.py"),
             "--metadata", str(prep_metadata), "--cache-dir", str(cache_dir),
             "--output", str(audit_path), "--workers", str(args.workers)])

    # Include the split-policy version so a changed assignment rule produces a
    # new immutable protocol instead of reusing the prior randomized split.
    protocol_key = f"{metadata_hash}_{sha256(audit_path)[:16]}_alphabetical_v1"
    protocol_dir = prepared_dir / protocol_key
    stage(4, "Build the content-deduplicated train/validation/test protocol")
    if not protocol_dir.exists():
        run([sys.executable, str(package_dir / "scripts/prepare_wlasl_training.py"),
             "--metadata", str(prep_metadata), "--audit", str(audit_path),
             "--cache-dir", str(cache_dir), "--output-dir", str(protocol_dir),
             "--min-videos", "10", "--seed", str(config.get("seed", 42))])
    else:
        protocol_file = protocol_dir / "protocol.json"
        if not protocol_file.is_file():
            raise RuntimeError(f"Incomplete prepared dataset at {protocol_dir}; move it aside and rerun")
        protocol = json.loads(protocol_file.read_text())
        if protocol.get("audit_report_sha256") != sha256(audit_path):
            raise RuntimeError(f"Prepared dataset provenance mismatch: {protocol_dir}")

    # Point only this training invocation at its prepared protocol. The input
    # YAML remains unchanged and the engine will create fixed validation episodes.
    config["data"]["metadata"] = str(protocol_dir / "metadata.csv")
    config["data"]["cache_dir"] = str(cache_dir)
    config["evaluation"]["val_episodes_file"] = str(
        protocol_dir / "episodes" / f"val_{config['training']['n_way']}w{config['training']['k_shot']}s_seed"
        f"{config['evaluation'].get('seed', 123)}.json"
    )
    generated_config = package_dir / "outputs" / f".prepared_{config_path.stem}.yaml"
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Prepared dataset: {protocol_dir}", flush=True)
    print(f"Training config: {generated_config}", flush=True)
    print(f"Dataset preparation finished in {(time.perf_counter() - overall_start) / 60:.1f} minutes",
          flush=True)


if __name__ == "__main__":
    main()
