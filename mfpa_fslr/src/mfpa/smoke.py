import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .datasets.feature_dataset import FeatureDataset
from .datasets.episode_sampler import EpisodeSampler
from .engine import score_episode, train_main, evaluate_main
from .models import MFPAFSLR
from .utils.seed import seed_everything


def synthetic_cache(class_index=0, sample_index=0, frames=4, roi_size=16):
    """Artificial feature fixture; contains no measured sign-language evidence."""
    rng = np.random.default_rng(1000 + class_index * 100 + sample_index)
    hands = rng.normal(0, .025, (frames, 2, 21, 3)).astype(np.float32)
    mouth = rng.normal(0, .01, (frames, 40, 3)).astype(np.float32)
    position = np.array([.2 + (class_index % 3) * .25, .3 + (class_index % 2) * .3], np.float32)
    hands[..., :2] += position
    mouth[..., :2] += position
    # Class-dependent articulation and directed flow, shared across samples.
    hands[:, :, 4:9, 1] += .01 * (class_index + 1)
    flow = rng.normal(0, .01, (3, frames - 1, 2, roi_size, roi_size)).astype(np.float32)
    flow[:, :, 0] += np.cos(class_index * 2.) * .5
    flow[:, :, 1] += np.sin(class_index * 2.) * .5
    return {"flow": flow, "hand_lm": hands, "mouth_lm": mouth,
            "hand_center": hands[..., :2].mean(2), "mouth_center": mouth[..., :2].mean(1),
            "hand_valid": np.ones((frames, 2), bool), "mouth_valid": np.ones(frames, bool),
            "flow_quality": np.ones((3, frames - 1), np.float32),
            "transition_valid": np.ones(frames - 1, bool), "sampled_frame_ids": np.arange(frames),
            "config_hash": np.array("synthetic-fixture-v1")}


def make_synthetic_dataset(root):
    root = Path(root)
    if (root / "metadata.csv").exists():
        raise FileExistsError("Use an empty directory for synthetic smoke data")
    cache = root / "processed"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for cls in range(3):
            class_id = split_index * 3 + cls
            for sample in range(4):
                vid = f"{split}_{cls}_{sample}"
                np.savez_compressed(cache / f"{vid}.npz", **synthetic_cache(cls, sample + split_index * 10))
                rows.append({"video_id": vid, "path": f"synthetic/{vid}.mp4", "gloss": f"SYNTHETIC_{class_id}",
                             "class_id": str(class_id), "language": "SYNTHETIC", "signer_id": "", "split": split})
    metadata = root / "metadata.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata, cache


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic backward/overfit/checkpoint/evaluation smoke test")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args(argv)
    if args.steps < 2:
        raise ValueError("At least two overfit steps are required")
    torch.set_num_threads(2)
    seed_everything(42)
    metadata, cache = make_synthetic_dataset(args.output / "data")
    dataset = FeatureDataset(metadata, cache, "train")
    episode = EpisodeSampler(dataset.class_to_videos, 3, 1, 2, seed=42).sample()
    model = MFPAFSLR(dimension=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    losses = []
    for _ in range(args.steps):
        logits, labels, loss, _ = score_episode(model, dataset, episode, "cpu", {})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite smoke loss")
        optimizer.step()
        losses.append(loss.item())
    logits, labels, _, _ = score_episode(model, dataset, episode, "cpu", {})
    acc = (logits.argmax(-1) == labels).float().mean().item()
    if losses[-1] >= losses[0] or acc < .99:
        raise AssertionError(f"Synthetic overfit failed: loss {losses[0]} -> {losses[-1]}, accuracy {acc}")
    config = {"seed": 42, "output_dir": str(args.output / "training"),
              "data": {"metadata": str(metadata), "cache_dir": str(cache)},
              "model": {"dimension": 32, "variant": "A5", "normalization": "group", "regions": [0, 1, 2]},
              "head": {"lambda_part": 1., "tau_s": .2},
              "training": {"n_way": 3, "k_shot": 1, "queries_per_class": 2, "episodes_per_epoch": 2,
                           "max_epochs": 1, "lr": .001, "weight_decay": .0001, "grad_clip": 1., "amp": False},
              "evaluation": {"val_episodes_file": str(args.output / "episodes/val.json"), "val_episodes": 2, "seed": 123}}
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_main(["--config", str(config_path), "--device", "cpu"])
    test = FeatureDataset(metadata, cache, "test")
    episodes = args.output / "episodes/test.json"
    EpisodeSampler(test.class_to_videos, 3, 1, 2, seed=2026).save(episodes, 3, test)
    evaluate_main(["--checkpoint", str(args.output / "training/best.pt"), "--episodes", str(episodes),
                   "--output", str(args.output / "test_metrics.json"), "--device", "cpu"])
    summary = {"synthetic_only": True, "overfit_initial_loss": losses[0], "overfit_final_loss": losses[-1],
               "overfit_accuracy": acc, "steps": args.steps, "end_to_end_checkpoint_evaluation": "passed"}
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
