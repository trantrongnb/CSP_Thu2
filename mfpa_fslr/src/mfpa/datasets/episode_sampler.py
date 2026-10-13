import argparse
import json
import random
from pathlib import Path


class EpisodeSampler:
    def __init__(self, class_to_videos, n_way=5, k_shot=1, queries_per_class=5, seed=42):
        if n_way < 2 or k_shot < 1 or queries_per_class < 1:
            raise ValueError("Require N>=2, K>=1, Q>=1")
        self.pool = {str(k): sorted(v) for k, v in class_to_videos.items()}
        self.n, self.k, self.q, self.seed = n_way, k_shot, queries_per_class, seed
        self.rng = random.Random(seed)
        if len(self.pool) < n_way:
            raise ValueError(f"Need {n_way} classes, have {len(self.pool)}")
        videos = [v for values in self.pool.values() for v in values]
        if len(videos) != len(set(videos)):
            raise ValueError("Each video ID must occur exactly once in the class pool")
        small = {c: len(v) for c, v in self.pool.items() if len(v) < k_shot + queries_per_class}
        if small:
            raise ValueError(f"Classes need K+Q={k_shot + queries_per_class} distinct videos: {small}")

    def sample(self):
        classes = self.rng.sample(sorted(self.pool), self.n)
        support, query, labels = [], [], []
        for label, cls in enumerate(classes):
            selected = self.rng.sample(self.pool[cls], self.k + self.q)
            support.extend(selected[:self.k])
            query.extend(selected[self.k:])
            labels.extend([label] * self.q)
        return {"classes": classes, "support": support, "query": query, "query_labels": labels}

    def save(self, path, count, dataset):
        if count < 2:
            raise ValueError("Use at least two evaluation episodes to estimate uncertainty")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to replace fixed episodes: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "split": dataset.split, "dataset_fingerprint": dataset.fingerprint,
                    "config_hash": dataset.config_hash, "n_way": self.n, "k_shot": self.k,
                    "queries_per_class": self.q, "seed": self.seed,
                    "episodes": [self.sample() for _ in range(count)]}
        path.write_text(json.dumps(document, indent=2))
        return document


def load_episodes(path, dataset):
    document = json.loads(Path(path).read_text())
    if document["dataset_fingerprint"] != dataset.fingerprint or document["split"] != dataset.split:
        raise ValueError("Fixed episodes belong to a different dataset/cache/split")
    n, k, q = document["n_way"], document["k_shot"], document["queries_per_class"]
    if n < 2 or k < 1 or q < 1 or len(document["episodes"]) < 2:
        raise ValueError("Invalid episode protocol")
    for episode in document["episodes"]:
        s, query, classes, labels = (episode[key] for key in ("support", "query", "classes", "query_labels"))
        if (len(s) != n * k or len(query) != n * q or len(classes) != n or len(set(classes)) != n
            or len(labels) != len(query) or any(not isinstance(y, int) or not 0 <= y < n for y in labels)):
            raise ValueError("Malformed episode sizes/labels")
        if len(set(s + query)) != len(s + query):
            raise ValueError("Support/query leakage or repeated samples in episode")
        if any(labels.count(y) != q for y in range(n)):
            raise ValueError("Unequal query counts")
        for index, vid in enumerate(s):
            if vid not in dataset.by_id or dataset.by_id[vid]["class_key"] != classes[index // k]:
                raise ValueError("Support is not class-major or references wrong split/class")
        for vid, y in zip(query, labels):
            if vid not in dataset.by_id or dataset.by_id[vid]["class_key"] != classes[y]:
                raise ValueError("Query label/split mismatch")
    return document


def main(argv=None):
    from .feature_dataset import FeatureDataset
    parser = argparse.ArgumentParser(description="Save evaluation episodes shared by every ablation")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--queries-per-class", type=int, default=5)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    dataset = FeatureDataset(args.metadata, args.cache_dir, args.split)
    sampler = EpisodeSampler(dataset.class_to_videos, args.n_way, args.k_shot, args.queries_per_class, args.seed)
    sampler.save(args.output, args.count, dataset)
    print(f"Saved {args.count} fixed {args.n_way}-way {args.k_shot}-shot episodes to {args.output}")


if __name__ == "__main__":
    main()
