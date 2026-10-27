"""Validate frozen caches and source hashes before selecting a WLASL protocol."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mfpa.datasets.feature_dataset import read_metadata, structural_features, validate_cache
from mfpa.preprocessing.pipeline import preprocessing_signature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.output.exists():
        raise FileExistsError("Choose a new report path; existing audits are never overwritten")
    metadata_bytes = args.metadata.read_bytes()
    rows = read_metadata(args.metadata)
    manifest = json.loads((args.cache_dir / "preprocessing_manifest.json").read_text())
    config = manifest["config"]
    signature, _ = preprocessing_signature(config)
    expected_shape = (3, config["target_frames"] - 1, 2, config["roi_size"], config["roi_size"])
    start = time.monotonic()

    def inspect(row):
        vid = row["video_id"]
        try:
            with np.load(args.cache_dir / f"{vid}.npz", allow_pickle=False) as handle:
                cache = {key: handle[key] for key in handle.files}
            validate_cache(cache, vid)
            if str(cache["config_hash"].item()) != manifest["config_hash"]:
                raise ValueError("Cache configuration hash mismatch")
            if cache["flow"].shape != expected_shape or cache["flow"].dtype != np.dtype(config["cache_dtype"]):
                raise ValueError("Flow dimensions/dtype disagree with preprocessing config")
            flow = cache["flow"].astype(np.float32)
            if np.max(np.abs(flow)) > config["flow_clip"]:
                raise ValueError("Flow exceeds configured clipping range")
            if np.any(flow[cache["flow_quality"] <= 0] != 0):
                raise ValueError("Invalid flow is nonzero")
            source = Path(row["path"])
            if not source.is_absolute():
                source = args.metadata.parent / source
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_hash != str(cache["source_hash"].item()):
                raise ValueError("Source video hash mismatch")
            mirror = row.get("is_mirrored", "false").strip().lower()
            if mirror not in {"", "0", "false", "1", "true"}:
                raise ValueError("Invalid is_mirrored metadata")
            if bool(cache["is_mirrored"].item()) != (mirror in {"1", "true"}):
                raise ValueError("Mirror convention differs from cache")
            inputs = structural_features(cache)
            if not all(np.isfinite(value).all() for value in inputs.values()):
                raise ValueError("Derived features contain NaN/Inf")
            usable = {"A0": bool(inputs["struct_valid"].any()),
                      "A1": bool((inputs["quality"] > 0).any()),
                      "A5": bool(inputs["valid_mask"].any())}
            return {"video_id": vid, "class_key": row["class_key"], "gloss": row["gloss"],
                    "source_hash": source_hash, "usable": usable}
        except Exception as exc:
            return {"video_id": vid, "class_key": row["class_key"], "gloss": row["gloss"], "error": str(exc)}

    errors, by_hash = [], defaultdict(list)
    invalid = {key: [] for key in ("A0", "A1", "A5", "common")}
    counts = {key: Counter() for key in invalid}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, item in enumerate(pool.map(inspect, rows), 1):
            if "error" in item:
                errors.append(item)
            else:
                identity = {key: item[key] for key in ("video_id", "class_key", "gloss")}
                by_hash[item["source_hash"]].append(identity)
                for variant, ok in {**item["usable"], "common": all(item["usable"].values())}.items():
                    if ok:
                        counts[variant][item["class_key"]] += 1
                    else:
                        invalid[variant].append(identity)
            if index % 2000 == 0:
                print(f"Checked {index}/{len(rows)} in {time.monotonic() - start:.1f}s", flush=True)
    if args.metadata.read_bytes() != metadata_bytes:
        raise ValueError("Metadata changed during audit; repeat with stable inputs")
    groups = [{"source_hash": key, "videos": items} for key, items in by_hash.items() if len(items) > 1]
    report = {"checked_videos": len(rows), "cache_errors": errors,
              "all_invalid_by_variant": invalid, "duplicate_source_groups": groups,
              "cross_class_duplicate_source_groups": [g for g in groups if len({v["class_key"] for v in g["videos"]}) > 1],
              "input_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "preprocessing_config_hash": manifest["config_hash"],
              "manifest_matches_current_config_environment": signature == manifest["config_hash"],
              "metadata_split_counts": dict(Counter(row["split"] for row in rows)),
              "eligibility": {variant: {"usable_videos": sum(pool.values()),
                                        "classes_eligible_1shot_Q5": sum(n >= 6 for n in pool.values()),
                                        "classes_eligible_5shot_Q5": sum(n >= 10 for n in pool.values())}
                              for variant, pool in counts.items()},
              "audit_seconds": time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.replace(args.output)
    print(f"Audited {len(rows)} videos; errors={len(errors)}; duplicates={len(groups)}; report={args.output}")


if __name__ == "__main__":
    main()
