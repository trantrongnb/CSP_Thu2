#!/usr/bin/env python3
"""Generate a Markdown experiment report from MFPA-FSLR output artifacts."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_metrics(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "val_accuracy_pct" in row:
            rows.append(row)
    return rows


def load_config(path):
    """Read the nested scalar/list subset used by project YAML configs."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    root = {}
    stack = [(-1, root)]
    for raw in lines:
        content = raw.split("#", 1)[0].rstrip()
        if not content.strip() or ":" not in content:
            continue
        indent = len(content) - len(content.lstrip())
        key, raw_value = content.strip().split(":", 1)
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        raw_value = raw_value.strip()
        if not raw_value:
            value = {}
            parent[key] = value
            stack.append((indent, value))
        elif raw_value.startswith("- "):
            parent.setdefault(key, []).append(parse_scalar(raw_value[2:]))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def parse_scalar(value):
    value = value.strip().strip("\"'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def find_audit(outputs_roots, explicit=None):
    if explicit:
        return read_json(explicit), Path(explicit)
    candidates = sorted({
        path for outputs in outputs_roots
        for pattern in ("wlasl_training_readiness/report_*.json",
                        "*/wlasl_training_readiness/report_*.json")
        for path in outputs.glob(pattern)
    })
    for path in candidates:
        data = read_json(path)
        if data:
            return data, path
    return {}, None


def find_evaluations(run_dir):
    results = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in {"splits.json", "provenance.json"}:
            continue
        data = read_json(path)
        if isinstance(data, dict) and "mean_accuracy_pct" in data:
            results.append((path, data))
    return results


def find_runs(outputs_roots):
    found = {}
    for outputs in outputs_roots:
        for config_path in sorted(outputs.glob("*/config.yaml")):
            run_dir = config_path.parent
            config = load_config(config_path)
            metrics = load_metrics(run_dir / "metrics.jsonl")
            if config and metrics:
                run = {
                    "dir": run_dir,
                    "config": config,
                    "metrics": metrics,
                    "evaluations": find_evaluations(run_dir),
                }
                identity = (run_dir.name, config.get("seed"), episode_setup(config))
                previous = found.get(identity)
                # Prefer a result directory with eval artifacts or richer metrics if duplicate copies exist.
                score = (bool(run["evaluations"]), len(metrics), len(list(run_dir.glob("*.pt"))))
                previous_score = ((bool(previous["evaluations"]), len(previous["metrics"]),
                                   len(list(previous["dir"].glob("*.pt")))) if previous else None)
                if previous is None or score > previous_score:
                    found[identity] = run
    return sorted(found.values(), key=lambda run: str(run["dir"]))


def clean(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt_num(value, digits=2):
    if value is None:
        return "chưa có"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "chưa có"


def episode_setup(config):
    training = config.get("training", {})
    return (training.get("n_way"), training.get("k_shot"),
            training.get("queries_per_class"))


def eval_episode_count(config, result):
    if result and result.get("n_episodes") is not None:
        return result["n_episodes"]
    return config.get("evaluation", {}).get("val_episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, action="append",
                        help="Output directory to scan; repeat to specify multiple directories")
    parser.add_argument("--outputs-root", type=Path, default=ROOT,
                        help="Directory under which all outputs* directories are discovered")
    parser.add_argument("--audit", type=Path,
                        help="Optional readiness audit JSON (auto-discovered by default)")
    parser.add_argument("--preprocess-config", type=Path, default=ROOT / "configs/preprocess.yaml")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "README.md",
                        help="Markdown destination (default: repository-root REPORT.md)")
    args = parser.parse_args()

    if args.outputs:
        outputs_roots = [path.resolve() for path in args.outputs]
    else:
        outputs_roots = sorted(path.resolve() for path in args.outputs_root.glob("outputs*") if path.is_dir())
    if not outputs_roots:
        parser.error(f"No outputs* directories found under {args.outputs_root}")
    runs = find_runs(outputs_roots)
    audit_records = []
    if args.audit:
        audit_data = read_json(args.audit)
        if audit_data:
            audit_records.append((Path(args.audit), audit_data))
    else:
        for outputs_root in outputs_roots:
            for pattern in ("wlasl_training_readiness/report_*.json",
                            "*/wlasl_training_readiness/report_*.json"):
                for path in sorted(outputs_root.glob(pattern)):
                    data = read_json(path)
                    if data and all(data.get("input_metadata_sha256") != existing.get("input_metadata_sha256")
                                    for _, existing in audit_records):
                        audit_records.append((path, data))
    audit_records.sort(key=lambda pair: str(pair[0]))
    audit, audit_path = audit_records[0][1], audit_records[0][0] if audit_records else ({}, None)
    preprocess = load_config(args.preprocess_config)

    lines = [
        "# BÁO CÁO KẾT QUẢ THỰC NGHIỆM",
        "",
        "## 1. Thông tin chung",
        "",
        "| Hạng mục | Thiết lập |",
        "|---|---|",
    ]

    datasets = sorted({
        ("WLASL" if run["dir"].name.startswith("WLASL_") else
         Path(str(run["config"].get("data", {}).get("metadata", ""))).parent.name)
        for run in runs
    })
    variants = sorted({
        str(run["config"].get("model", {}).get("variant", "không rõ"))
        for run in runs
    })
    seeds = sorted({run["config"].get("seed") for run in runs if run["config"].get("seed") is not None})
    episode_specs = sorted({episode_setup(run["config"]) for run in runs})
    train_setups = ", ".join(
        f"{n}-way {k}-shot" for n, k, _ in episode_specs if n is not None and k is not None
    ) or "chưa có run artifacts"
    lines.extend([
        f"| Tên phương pháp | MFPA-FSLR {', '.join(variants) if variants else ''} |",
        f"| Dataset chính | {', '.join(datasets) if datasets else 'chưa có trong config'} |",
        f"| Số frame đầu vào | {preprocess.get('target_frames', 'chưa có')} frame/video |",
    ])
    split_summaries = []
    for audit_path_i, audit_i in audit_records:
        counts = audit_i.get("metadata_split_counts", {})
        if counts:
            split_summaries.append(
                f"{audit_path_i.parent.parent.name}: train {counts.get('train', 'chưa có'):,} / "
                f"eval {counts.get('val', 'chưa có'):,} / test {counts.get('test', 'chưa có'):,} video"
            )
    if split_summaries:
        train_eval_test = "; ".join(split_summaries)
    else:
        train_eval_test = "chưa có trong audit JSON"
    lines.extend([
        f"| Tỉ lệ Train : Eval : Test | Theo lớp theo protocol; {train_eval_test} |",
        f"| Seed | {', '.join(map(str, seeds)) if seeds else 'chưa có trong config'} |",
        "",
        "## 2. Cấu hình huấn luyện",
        "",
        "| Hạng mục | Thiết lập |",
        "|---|---|",
        f"| Cấu hình episodic training | {train_setups} |",
    ])
    epoch_descriptions = []
    query_values = set()
    encoder_descriptions = set()
    for run in runs:
        training = run["config"].get("training", {})
        epoch_descriptions.append(
            f"{training.get('episodes_per_epoch', '?')} episode/epoch × "
            f"{training.get('max_epochs', len(run['metrics']))} epoch"
        )
        if training.get("queries_per_class") is not None:
            query_values.add(training["queries_per_class"])
        model = run["config"].get("model", {})
        encoder_descriptions.add(
            f"embedding {model.get('dimension', '?')}D, "
            f"normalization {model.get('normalization', 'không rõ')}; "
            "config không ghi pretrained backbone"
        )
    unique_epoch_descriptions = list(dict.fromkeys(epoch_descriptions))
    lines.extend([
        f"| Số episode training | {'; '.join(unique_epoch_descriptions) if unique_epoch_descriptions else 'chưa có run'} |",
        f"| Query mỗi lớp | {', '.join(map(str, sorted(query_values))) if query_values else 'chưa có trong config'} |",
        f"| Backbone và pretrained | {'; '.join(sorted(encoder_descriptions)) if encoder_descriptions else 'chưa có trong config'} |",
        "",
        "## 3. Kết quả đánh giá N-way K-shot",
        "",
        "| Thiết lập | Accuracy | 95% CI half-width | Số episode | Nguồn/split |",
        "|---|---:|---:|---:|---|",
    ])
    evaluation_rows = 0
    for run in runs:
        config = run["config"]
        n, k, q = episode_setup(config)
        title = f"{n}-way {k}-shot" if n is not None and k is not None else run["dir"].name
        evaluations = run["evaluations"]
        if evaluations:
            for path, result in evaluations:
                setup_n = result.get("n_way", n)
                setup_k = result.get("k_shot", k)
                title_eval = f"{setup_n}-way {setup_k}-shot" if setup_n and setup_k else title
                ci = result.get("ci95_half_width_pct")
                lines.append(
                    f"| {clean(title_eval)} | {fmt_num(result.get('mean_accuracy_pct'))}% | "
                    f"±{fmt_num(ci)} điểm % | {result.get('n_episodes', 'chưa có')} | "
                    f"evaluation: `{clean(path)}` |"
                )
                evaluation_rows += 1
        else:
            metrics = run["metrics"]
            best = max(metrics, key=lambda row: row.get("val_accuracy_pct", float("-inf")))
            lines.append(
                f"| {clean(title)} | {fmt_num(best.get('val_accuracy_pct'))}% | "
                f"±{fmt_num(best.get('val_ci95_half_width_pct'))} điểm % | "
                f"{eval_episode_count(config, None) or 'chưa có'} | "
                f"validation, epoch {best.get('epoch', '?')} |"
            )
            evaluation_rows += 1
    if not runs:
        lines.append("| Chưa có training run | — | — | — | — |")
    lines.extend([
        "",
        "CI trong training log là xấp xỉ chuẩn `1.96 × σ / √E` tính trên episode. "
        "Nếu có file evaluation JSON, script ưu tiên kết quả và CI trong file đó.",
        "",
        "## 4. Độ phức tạp và tốc độ suy luận",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---|",
    ])
    eval_results = [result for run in runs for _, result in run["evaluations"]]
    parameter_counts = [r["parameter_count"] for r in eval_results if r.get("parameter_count") is not None]
    trainable_counts = [r["trainable_parameter_count"] for r in eval_results if r.get("trainable_parameter_count") is not None]
    peak_bytes = [r["peak_gpu_memory_bytes"] for r in eval_results if r.get("peak_gpu_memory_bytes")]
    latencies = [r["seconds_per_episode_including_cache_io"] for r in eval_results
                 if r.get("seconds_per_episode_including_cache_io") is not None]

    def million(values):
        return f"{values[-1] / 1e6:.3f} M" if values else "chưa được lưu trong artifacts"

    lines.extend([
        f"| Tổng số tham số | {million(parameter_counts)} |",
        f"| Số tham số được huấn luyện | {million(trainable_counts)} |",
        "| FLOPs | Chưa có phép đo trong artifacts |",
        (f"| Inference latency | {latencies[-1] * 1000:.2f} ms/episode (có cache I/O; "
         "không phải ms/video) |" if latencies else "| Inference latency | Chưa có phép đo trong artifacts |"),
        (f"| Peak inference VRAM | {peak_bytes[-1] / (1024 ** 3):.3f} GB |" if peak_bytes
         else "| Peak inference VRAM | Chưa có phép đo evaluation được lưu |"),
        "",
        "## 5. Ghi chú thực nghiệm",
        "",
    ])
    if audit_records:
        for audit_path_i, audit_i in audit_records:
            eligibility = audit_i.get("eligibility", {}).get("A5", {})
            lines.append(
                f"- Audit `{clean(audit_path_i)}`: kiểm tra {audit_i.get('checked_videos', 'không rõ'):,} video; "
                f"cache errors: {len(audit_i.get('cache_errors', []))}. "
                f"Lớp đủ điều kiện 1-shot Q=5: {eligibility.get('classes_eligible_1shot_Q5', 'không rõ'):,}; "
                f"5-shot Q=5: {eligibility.get('classes_eligible_5shot_Q5', 'không rõ'):,}."
            )
    else:
        lines.append("- Không tìm thấy readiness audit JSON; thống kê dataset được để trống.")
    for run in runs:
        metrics = run["metrics"]
        best = max(metrics, key=lambda row: row.get("val_accuracy_pct", float("-inf")))
        last = metrics[-1]
        n, k, _ = episode_setup(run["config"])
        title = f"{n}-way {k}-shot" if n and k else run["dir"].name
        delta = best["val_accuracy_pct"] - last.get("val_accuracy_pct", best["val_accuracy_pct"])
        lines.append(
            f"- {title}: best validation ở epoch {best.get('epoch', '?')} là "
            f"{best['val_accuracy_pct']:.2f}%; cuối run train {last.get('train_accuracy_pct', float('nan')):.2f}% "
            f"và validation {last.get('val_accuracy_pct', float('nan')):.2f}% "
            f"(thấp hơn peak {delta:.2f} điểm %). Chọn `best.pt` theo validation."
        )
        lines.append(f"- Artifacts: `{clean(run['dir'])}/`.")
    if not eval_results:
        lines.append("- Chưa thấy evaluation JSON; accuracy ở mục 3 là validation, không phải held-out test.")
    if len(seeds) <= 1:
        lines.append("- Chỉ có một training seed trong các run được quét; chưa ước lượng độ biến thiên giữa seed.")
    lines.extend([
        "- Đây là protocol WLASL tự xây dựng của workspace; không so sánh trực tiếp với paper nếu protocol, preprocessing và test episodes khác.",
        "",
        "---",
        "",
        f"Được tạo tự động từ {len(outputs_roots)} thư mục outputs*, config, `metrics.jsonl`, evaluation JSON và readiness audit.",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Scanned {len(outputs_roots)} outputs* director(ies); generated {args.output} "
          f"from {len(runs)} training run(s) and {evaluation_rows} result row(s).")


if __name__ == "__main__":
    main()
