import torch


def encode_episode(model, dataset, episode, device):
    ids = episode["support"] + episode["query"]
    samples = [dataset[video_id] for video_id in ids]
    inputs = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}
    features = model.encode(**inputs)  # One combined batch, also for optional BatchNorm.
    if not features["global_valid"].all():
        bad = [vid for vid, ok in zip(ids, features["global_valid"].tolist()) if not ok]
        raise ValueError(f"All-invalid videos reached encoder: {bad}")
    n, support_count = len(episode["classes"]), len(episode["support"])
    k = support_count // n
    support = {key: value[:support_count].reshape(n, k, *value.shape[1:])
               for key, value in features.items() if key in {"global", "parts", "part_quality"}}
    query = {key: value[support_count:] for key, value in features.items()
             if key in {"global", "parts", "part_quality"}}
    labels = torch.tensor(episode["query_labels"], dtype=torch.long, device=device)
    return support, query, labels
