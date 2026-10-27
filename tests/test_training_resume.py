import torch
import yaml

from mfpa.engine import train_main


def test_resume_reproduces_uninterrupted_training(synthetic_data, tmp_path):
    metadata, cache = synthetic_data
    output = tmp_path / "training"
    config = {"seed": 42, "output_dir": str(output),
              "data": {"metadata": str(metadata), "cache_dir": str(cache)},
              "model": {"dimension": 16, "variant": "A5"},
              "head": {"lambda_part": 1., "tau_s": .2},
              "training": {"n_way": 3, "k_shot": 1, "queries_per_class": 1, "episodes_per_epoch": 1,
                           "max_epochs": 2, "lr": .001, "weight_decay": .0001, "grad_clip": 1., "amp": False},
              "evaluation": {"val_episodes_file": str(tmp_path / "val.json"), "val_episodes": 2}}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    train_main(["--config", str(path), "--device", "cpu"])
    expected = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    # Construct the exact completed first-epoch state by rerunning a one-epoch config.
    single = {**config, "training": {**config["training"], "max_epochs": 1},
              "output_dir": str(tmp_path / "one_epoch")}
    single_path = tmp_path / "one_epoch.yaml"
    single_path.write_text(yaml.safe_dump(single))
    train_main(["--config", str(single_path), "--device", "cpu"])
    interrupted = torch.load(tmp_path / "one_epoch/last.pt", map_location="cpu", weights_only=True)
    interrupted["config"] = config
    resume_path = tmp_path / "interrupted.pt"
    torch.save(interrupted, resume_path)
    train_main(["--config", str(path), "--resume", str(resume_path), "--device", "cpu"])
    actual = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert actual["epoch"] == expected["epoch"] == 1
    for key, tensor in actual["model"].items():
        torch.testing.assert_close(tensor, expected["model"][key], rtol=0, atol=0)
    assert actual["best_val_accuracy_pct"] == expected["best_val_accuracy_pct"]
