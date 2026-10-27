"""Train A0..A5 sequentially and produce a CSV on the same fixed test episodes."""
import argparse
import csv
import json
from pathlib import Path

from mfpa.engine import train_main, evaluate_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    rows = []
    for variant in (f"A{i}" for i in range(6)):
        output = args.output / variant
        train_main(["--config", str(args.config), "--variant", variant, "--output", str(output),
                    "--device", args.device])
        result_path = output / "test_metrics.json"
        evaluate_main(["--checkpoint", str(output / "best.pt"), "--episodes", str(args.episodes),
                       "--output", str(result_path), "--device", args.device])
        result = json.loads(result_path.read_text())
        rows.append({key: result[key] for key in ("variant", "mean_accuracy_pct", "ci95_half_width_pct",
                                                  "n_episodes", "n_way", "k_shot", "training_seed", "episodes_sha256")})
        with (args.output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
