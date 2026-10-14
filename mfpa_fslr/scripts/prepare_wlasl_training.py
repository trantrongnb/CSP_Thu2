"""Build a local, audited WLASL protocol; this is not a published WLASL split."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from mfpa.datasets.episode_sampler import EpisodeSampler
from mfpa.datasets.feature_dataset import read_metadata


def select_rows(rows, audit, min_videos):
    if min_videos < 2:
        raise ValueError("min_videos must be at least 2")
    if audit["cache_errors"]:
        raise ValueError("Resolve cache errors before preparing a training protocol")
    if audit["checked_videos"] != len(rows):
        raise ValueError("Audit does not cover the supplied metadata")
    by_id = {row["video_id"]: row for row in rows}
    rejected = {item["video_id"]: "invalid_common_A0_A5"
                for item in audit["all_invalid_by_variant"]["common"]}
    if not set(rejected) <= set(by_id):
        raise ValueError("Audit references unknown invalid video IDs")
    counts = Counter(row["class_key"] for row in rows if row["video_id"] not in rejected)
    group_members = set()
    for group in audit["duplicate_source_groups"]:
        videos = group["videos"]
        for item in videos:
            vid = item["video_id"]
            if vid not in by_id or by_id[vid]["class_key"] != item["class_key"] or vid in group_members:
                raise ValueError("Audit duplicate groups disagree with metadata")
            group_members.add(vid)
        candidates = [by_id[item["video_id"]] for item in videos if item["video_id"] not in rejected]
        if candidates:
            # Preserve a copy in the least populated class; ties use a stable ID.
            canonical = min(candidates, key=lambda row: (counts[row["class_key"]], row["video_id"]))
            for row in candidates:
                if row["video_id"] != canonical["video_id"]:
                    rejected[row["video_id"]] = f"duplicate_content_of:{canonical['video_id']}"
    kept = [dict(row) for row in rows if row["video_id"] not in rejected]
    counts = Counter(row["class_key"] for row in kept)
    for row in kept:
        if counts[row["class_key"]] < min_videos:
            rejected[row["video_id"]] = f"class_below_{min_videos}_unique_valid_videos"
    return sorted((row for row in kept if row["video_id"] not in rejected),
                  key=lambda row: (row["class_key"], row["video_id"])), rejected


def split_classes(rows, duplicate_groups, seed=42):
    classes = sorted({row["class_key"] for row in rows})
    parents = {key: key for key in classes}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for group in duplicate_groups:
        keys = sorted({item["class_key"] for item in group["videos"]} & set(classes))
        for key in keys[1:]:
            parents[find(key)] = find(keys[0])
    components = defaultdict(list)
    for key in classes:
        components[find(key)].append(key)
    groups = sorted(components.values())
    random.Random(seed).shuffle(groups)

    def take_exact(pool, target):
        reachable = {0: ()}
        for index, group in enumerate(pool):
            for total in sorted(reachable, reverse=True):
                new_total = total + len(group)
                if new_total <= target and new_total not in reachable:
                    reachable[new_total] = (*reachable[total], index)
            if target in reachable:
                selected = set(reachable[target])
                return ([key for i, group in enumerate(pool) if i in selected for key in group],
                        [group for i, group in enumerate(pool) if i not in selected])
        raise ValueError("Cannot obtain the target class counts while preserving duplicate-linked groups")

    train_target = int(len(classes) * .6)
    val_target = int(len(classes) * .2)
    test_target = len(classes) - train_target - val_target
    if min(train_target, val_target, test_target) < 1:
        raise ValueError("Need enough classes for train/val/test")
    val, remaining = take_exact(groups, val_target)
    test, remaining = take_exact(remaining, test_target)
    assignments = {key: "train" for group in remaining for key in group}
    assignments.update({key: "val" for key in val})
    assignments.update({key: "test" for key in test})
    return assignments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wlasl_60_20_20"))
    parser.add_argument("--min-videos", type=int, choices=(6, 10), default=10,
                        help="10 for a shared 1/5-shot pool with Q=5; 6 for 1-shot only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing protocols are never overwritten")
    rows = read_metadata(args.metadata)
    audit = json.loads(args.audit.read_text())
    if hashlib.sha256(args.metadata.read_bytes()).hexdigest() != audit.get("input_metadata_sha256"):
        raise ValueError("Metadata changed since the readiness audit")
    manifest = json.loads((args.cache_dir / "preprocessing_manifest.json").read_text())
    if manifest["config_hash"] != audit.get("preprocessing_config_hash"):
        raise ValueError("Cache configuration changed since the readiness audit")
    if not audit["manifest_matches_current_config_environment"]:
        raise ValueError("Audit reported a preprocessing configuration/environment mismatch")
    kept, rejected = select_rows(rows, audit, args.min_videos)
    assignments = split_classes(kept, audit["duplicate_source_groups"], args.seed)
    pools = {split: defaultdict(list) for split in ("train", "val", "test")}
    for row in kept:
        row["split"] = assignments[row["class_key"]]
        source = Path(row["path"])
        row["path"] = str((source if source.is_absolute() else args.metadata.parent / source).resolve())
        if not source.is_absolute():
            source = args.metadata.parent / source
        if not source.is_file() or not (args.cache_dir / f"{row['video_id']}.npz").is_file():
            raise FileNotFoundError(f"Missing source/cache: {row['video_id']}")
        pools[row["split"]][row["class_key"]].append(row["video_id"])
    for pool in pools.values():
        EpisodeSampler(pool, 5, args.min_videos - 5, 5, args.seed)
    # Verify no duplicate-content group retains more than one copy.
    ids = {row["video_id"] for row in kept}
    if any(sum(item["video_id"] in ids for item in group["videos"]) > 1
           for group in audit["duplicate_source_groups"]):
        raise ValueError("Duplicate video content remains in the selected pool")

    with args.metadata.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    args.output_dir.mkdir(parents=True)
    output_metadata = args.output_dir / "metadata.csv"
    with output_metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    read_metadata(output_metadata)  # Validate novel-class/gloss separation.
    with (args.output_dir / "excluded_videos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "gloss", "class_id", "reason"])
        writer.writeheader()
        writer.writerows({"video_id": row["video_id"], "gloss": row["gloss"],
                          "class_id": row["class_id"], "reason": rejected[row["video_id"]]}
                         for row in rows if row["video_id"] in rejected)
    splits_dir = args.output_dir / "splits"
    splits_dir.mkdir()
    for split, pool in pools.items():
        (splits_dir / f"{split}_classes.json").write_text(json.dumps(sorted(pool), indent=2))
    protocol = {"name": "local_WLASL_global_content_dedup_class_disjoint",
                "published_protocol": False, "seed": args.seed, "target_class_ratios": [0.6, 0.2, 0.2],
                "rounding": "floor(0.6*N) train, floor(0.2*N) val; remaining classes test",
                "min_unique_valid_videos_per_class": args.min_videos,
                "deduplication": "global exact source SHA256; prefer least populated class, then video ID",
                "split_groups": "connected classes sharing source content in the original audited pool",
                "signer_protocol": "signer IDs unknown; not a cross-signer benchmark",
                "class_ids": "local IDs inherited from preprocessing metadata",
                "input_metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                "audit_report_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(output_metadata.read_bytes()).hexdigest(),
                "preprocessing_config_hash": manifest["config_hash"],
                "cache_dir": str(args.cache_dir.resolve()),
                "excluded_counts": dict(Counter(reason.split(":", 1)[0] for reason in rejected.values())),
                "splits": {split: {"classes": len(pool), "videos": sum(map(len, pool.values()))}
                           for split, pool in pools.items()}}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(json.dumps(protocol["splits"], indent=2))
    print(f"Metadata: {output_metadata}; excluded videos: {len(rejected)}")


if __name__ == "__main__":
    main()
