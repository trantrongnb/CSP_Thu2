#!/usr/bin/env python3
"""Evaluate every N-way/K-shot checkpoint under outputs via run_conda.sh."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
LAUNCHER = ROOT / "scripts" / "run_conda.sh"


def find_runs(outputs: Path):
    """Return all run directories with a config and best checkpoint."""
    return [(path.parent, path) for path in sorted(outputs.rglob("config.yaml"))
            if (path.parent / "best.pt").is_file()]


def episode_path(config: dict, episode_dir: Path, count: int) -> Path:
    """Use one fixed test episode file per dataset/protocol/seed combination."""
    data, training = config["data"], config["training"]
    seed = config.get("evaluation", {}).get("seed", config.get("seed", 42))
    identity = "\0".join((str(Path(data["metadata"]).resolve()),
                           str(Path(data["cache_dir"]).resolve()), str(training["n_way"]),
                           str(training["k_shot"]), str(training["queries_per_class"]),
                           str(seed), str(count)))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return episode_dir / (f"test_{training['n_way']}w{training['k_shot']}s_"
                          f"q{training['queries_per_class']}_seed{seed}_{digest}.json")


def launcher(*args: str):
    subprocess.run(["bash", str(LAUNCHER), *args], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--episodes-dir", type=Path, default=ROOT / "data/evaluation_episodes")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Replace existing test_metrics.json")
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")

    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        parser.error(f"No outputs directory: {outputs}")
    runs = find_runs(outputs)
    if not runs:
        parser.error(f"No config.yaml + best.pt run found below {outputs}")
    args.episodes_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for run_dir, config_path in runs:
        label = run_dir.relative_to(outputs)
        result = run_dir / "test_metrics.json"
        if result.exists() and not args.force:
            print(f"SKIP {label}: {result.name} exists", flush=True)
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data, training = config["data"], config["training"]
            fixed_episodes = episode_path(config, args.episodes_dir, args.episodes)
            if not fixed_episodes.exists():
                launcher("episodes", "--metadata", str(data["metadata"]), "--cache-dir", str(data["cache_dir"]),
                         "--split", "test", "--n-way", str(training["n_way"]),
                         "--k-shot", str(training["k_shot"]), "--queries-per-class",
                         str(training["queries_per_class"]), "--count", str(args.episodes),
                         "--seed", str(config.get("evaluation", {}).get("seed", config.get("seed", 42))),
                         "--output", str(fixed_episodes))
            command = ["evaluate", "--checkpoint", str(run_dir / "best.pt"), "--episodes", str(fixed_episodes),
                       "--output", str(result), "--device", args.device]
            if not args.skip_flops:
                command.append("--profile-flops")
            print(f"EVALUATE {label}: {training['n_way']}-way {training['k_shot']}-shot", flush=True)
            launcher(*command)
        except (KeyError, OSError, subprocess.CalledProcessError, yaml.YAMLError) as error:
            failed.append(f"{label}: {error}")
            print(f"FAILED {label}: {error}", file=sys.stderr, flush=True)

    if failed:
        raise SystemExit("Some evaluations failed:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
