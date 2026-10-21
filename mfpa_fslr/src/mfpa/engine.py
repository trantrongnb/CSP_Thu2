import argparse
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from .datasets.feature_dataset import FeatureDataset
from .datasets.episode_sampler import EpisodeSampler, load_episodes
from .datasets.collate import encode_episode
from .models import MFPAFSLR, part_proto_logits
from .models.prototype_head import RelationNetwork
from .utils.seed import seed_everything
from .utils.metrics import episodic_accuracy_summary


def build_model(config):
    model = MFPAFSLR(**config["model"])
    matcher = config.get("head", {}).get("matcher", "euclidean")
    if matcher not in {"euclidean", "relation"}:
        raise ValueError("head.matcher must be 'euclidean' or 'relation'")
    if matcher == "relation":
        model.relation_network = RelationNetwork(config["model"].get("dimension", 128))
    return model


def score_episode(model, dataset, episode, device, head):
    support, query, labels = encode_episode(model, dataset, episode, device)
    options = {**head}
    lambda_part = options.pop("lambda_part", 1.0)
    options["lam"] = lambda_part if model.variant in {"A4", "A5"} else 0.0
    options["tau_a"] = options.pop("tau_alpha", 1.0)
    options["adaptive"] = model.variant == "A5"
    matcher = options.pop("matcher", "euclidean")
    logits, alpha = part_proto_logits(support["global"], support["parts"], support["part_quality"],
                                      query["global"], query["parts"], query["part_quality"],
                                      matcher=matcher,
                                      relation_network=getattr(model, "relation_network", None),
                                      **options)
    loss = F.cross_entropy(logits, labels)
    return logits, labels, loss, alpha


def dataset_for(config, split, variant=None):
    return FeatureDataset(config["data"]["metadata"], config["data"]["cache_dir"], split,
                          variant=variant or config["model"].get("variant", "A5"),
                          regions=config["model"].get("regions", [0, 1, 2]))


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, dataset, document, device, head):
    model.eval()
    accuracies, losses, per_episode, confusion, alphas = [], [], [], {}, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    for episode in document["episodes"]:
        logits, labels, loss, alpha = score_episode(model, dataset, episode, device, head)
        predicted = logits.argmax(-1)
        acc = (predicted == labels).float().mean().item()
        accuracies.append(acc)
        losses.append(loss.item())
        alphas.append(alpha.cpu())
        per_episode.append({"accuracy": acc, "loss": loss.item(), "alpha": alpha.cpu().tolist()})
        for actual, pred in zip(labels.tolist(), predicted.tolist()):
            actual, pred = episode["classes"][actual], episode["classes"][pred]
            row = confusion.setdefault(actual, {})
            row[pred] = row.get(pred, 0) + 1
    sync(device)
    elapsed = time.perf_counter() - start
    result = episodic_accuracy_summary(accuracies)
    result.update(mean_loss=sum(losses) / len(losses),
                  seconds_per_episode_including_cache_io=elapsed / len(accuracies),
                  query_videos_per_second=len(accuracies) * document["n_way"] * document["queries_per_class"] / elapsed,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  mean_alpha=torch.cat(alphas).mean(0).tolist(), confusion=confusion, episodes=per_episode)
    return result


def provenance():
    dependencies = {}
    for package in ("torch", "torchvision", "numpy", "mediapipe", "opencv-python-headless", "mfpa-fslr"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"dependencies": dependencies, "git_commit": commit}


def atomic_checkpoint(state, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Episodic training and validation for A0..A5")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=[f"A{i}" for i in range(6)])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.variant:
        config["model"]["variant"] = args.variant
    output = args.output or Path(config["output_dir"])
    config["output_dir"] = str(output)
    settings = config["training"]
    if settings["max_epochs"] < 1 or settings["episodes_per_epoch"] < 1 or settings["grad_clip"] <= 0:
        raise ValueError("Require positive epochs, episodes and grad_clip")
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output already has a checkpoint; use --resume or a new --output")
    seed_everything(config["seed"])
    device = choose_device(args.device)
    train, val = dataset_for(config, "train"), dataset_for(config, "val")
    if train.config_hash != val.config_hash:
        raise ValueError("Train/validation preprocessing mismatch")
    # Fail before training if any class has too few samples.
    EpisodeSampler(train.class_to_videos, settings["n_way"], settings["k_shot"],
                   settings["queries_per_class"], config["seed"])
    val_path = Path(config["evaluation"]["val_episodes_file"])
    if not val_path.exists():
        EpisodeSampler(val.class_to_videos, settings["n_way"], settings["k_shot"],
                       settings["queries_per_class"], config["evaluation"].get("seed", 123)).save(
                           val_path, config["evaluation"]["val_episodes"], val)
    val_document = load_episodes(val_path, val)
    if (val_document["n_way"], val_document["k_shot"], val_document["queries_per_class"]) != (
        settings["n_way"], settings["k_shot"], settings["queries_per_class"]):
        raise ValueError("Validation episode protocol disagrees with training config")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    amp = settings.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    first_epoch, best = 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (state["config"] != config or state["train_fingerprint"] != train.fingerprint
            or state["val_episodes_sha256"] != file_hash(val_path)):
            raise ValueError("Resume requires the same config, dataset and validation episodes")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_epoch, best = state["epoch"] + 1, state["best_val_accuracy_pct"]
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "provenance.json").write_text(json.dumps(provenance(), indent=2))
    (output / "splits.json").write_text(json.dumps({"train": sorted(train.class_to_videos),
                                                  "val": sorted(val.class_to_videos)}, indent=2))
    for epoch in range(first_epoch, settings["max_epochs"]):
        model.train()
        sampler = EpisodeSampler(train.class_to_videos, settings["n_way"], settings["k_shot"],
                                 settings["queries_per_class"], config["seed"] + epoch)
        loss_sum, accuracy_sum = 0., 0.
        for index in range(settings["episodes_per_epoch"]):
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                logits, labels, loss, _ = score_episode(model, train, sampler.sample(), device, config.get("head", {}))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, episode {index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not gradients_finite:
                # AMP can overflow gradients while the forward loss remains finite.
                # Skip this update; GradScaler will reduce its scale on update().
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if (index + 1) % settings.get("log_every", 100) == 0 or index == 0:
                    print(f"epoch {epoch + 1}, episode {index + 1}: non-finite gradients; "
                          "skipped optimizer step", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            accuracy_sum += (logits.argmax(-1) == labels).float().mean().item()
            if (index + 1) % settings.get("log_every", 100) == 0 or index == 0:
                print(f"epoch {epoch + 1}, episode {index + 1}: loss={loss.item():.4f}", flush=True)
        validation = evaluate(model, val, val_document, device, config.get("head", {}))
        metric = validation["mean_accuracy_pct"]
        improved = metric > best
        best = max(best, metric)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "config": config, "epoch": epoch, "best_val_accuracy_pct": best,
                 "train_fingerprint": train.fingerprint, "preprocessing_hash": train.config_hash,
                 "train_classes": sorted(train.class_to_videos), "val_episodes_sha256": file_hash(val_path),
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_checkpoint(state, output / "last.pt")
        if improved:
            atomic_checkpoint(state, output / "best.pt")
        log = {"epoch": epoch + 1, "train_loss": loss_sum / settings["episodes_per_epoch"],
               "train_accuracy_pct": 100 * accuracy_sum / settings["episodes_per_epoch"],
               "val_accuracy_pct": metric, "val_ci95_half_width_pct": validation["ci95_half_width_pct"]}
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
    print(f"Best validation accuracy: {best:.2f}%; checkpoint: {output / 'best.pt'}", flush=True)


def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on fixed novel-class episodes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args(argv)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if args.metadata:
        config["data"]["metadata"] = str(args.metadata)
    if args.cache_dir:
        config["data"]["cache_dir"] = str(args.cache_dir)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    dataset = dataset_for(config, args.split)
    if dataset.config_hash != state["preprocessing_hash"]:
        raise ValueError("Checkpoint and evaluation preprocessing differ")
    if set(dataset.class_to_videos) & set(state["train_classes"]):
        raise ValueError("Evaluation contains meta-training classes")
    document = load_episodes(args.episodes, dataset)
    result = evaluate(model, dataset, document, device, config.get("head", {}))
    # Cached model timing above excludes optical flow/detection; report measured offline cost separately.
    preprocessing_cost = []
    for row in dataset.rows:
        with np.load(dataset.cache_dir / f"{row['video_id']}.npz", allow_pickle=False) as cache:
            if "preprocessing_seconds" in cache:
                preprocessing_cost.append(float(cache["preprocessing_seconds"]))
    result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                  variant=model.variant, split=args.split, n_way=document["n_way"], k_shot=document["k_shot"],
                  queries_per_class=document["queries_per_class"], episode_seed=document["seed"],
                  training_seed=config["seed"], episodes_sha256=file_hash(args.episodes),
                  preprocessing_hash=dataset.config_hash, provenance=provenance(),
                  offline_preprocessing_seconds_per_video=(sum(preprocessing_cost) / len(preprocessing_cost)
                                                           if preprocessing_cost else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"{model.variant} {document['n_way']}-way {document['k_shot']}-shot: "
          f"{result['mean_accuracy_pct']:.2f} +/- {result['ci95_half_width_pct']:.2f}% (95% CI)")
    print(f"Results: {args.output}")
