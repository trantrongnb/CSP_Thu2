import numpy as np


def episodic_accuracy_summary(accuracies):
    values = np.asarray(accuracies, dtype=np.float64) * 100
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("CI requires at least two finite episode accuracies")
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean_accuracy_pct": mean, "ci95_half_width_pct": half,
            "ci95_pct": [mean - half, mean + half], "n_episodes": len(values),
            "ci_method": "normal approximation: mean +/- 1.96 * sample std / sqrt(episodes)"}
