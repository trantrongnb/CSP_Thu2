"""Validate author-supplied protocol and export class lists; never invent MS-ASL splits."""
import argparse
import json
from pathlib import Path

from mfpa.datasets.feature_dataset import read_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    rows = read_metadata(args.metadata)
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        classes = sorted({row["class_key"] for row in rows if row["split"] == split})
        (args.splits_dir / f"{split}_classes.json").write_text(json.dumps(classes, indent=2))
        print(f"{split}: {len(classes)} classes, {sum(row['split'] == split for row in rows)} videos")


if __name__ == "__main__":
    main()
