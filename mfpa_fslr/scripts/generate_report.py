#!/usr/bin/env python3
"""Generate Markdown and DOCX experiment reports from MFPA-FSLR output artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile
from lxml import etree

from docx import Document
from docx.shared import Pt


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


def metadata_path(config):
    value = config.get("data", {}).get("metadata")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def metadata_split_counts(config):
    """Count videos/classes from the metadata file referenced by config.yaml."""
    path = metadata_path(config)
    if path is None or not path.is_file():
        return None
    counts = {split: {"videos": 0, "classes": set()} for split in ("train", "val", "test")}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                split = row.get("split", "").strip().lower()
                if split not in counts:
                    continue
                counts[split]["videos"] += 1
                class_id = row.get("class_id") or row.get("gloss")
                if class_id:
                    counts[split]["classes"].add(class_id)
    except (OSError, csv.Error):
        return None
    return {
        split: {"videos": values["videos"], "classes": len(values["classes"])}
        for split, values in counts.items()
    }


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


def table_cells(line):
    """Split a Markdown table row while retaining escaped literal pipes."""
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [cell.strip().replace("\\|", "|") for cell in cells]


def table_separator(line):
    cells = table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def write_docx(lines, output, template):
    """Replace placeholder text in the DOCX XML, preserving every other part."""
    if not template.is_file():
        raise FileNotFoundError(f"DOCX template not found: {template}")
    document = Document(template)
    template_tables = document.tables

    # Extract generated values by report section; preserve the template's rows,
    # column widths, shading, borders, page settings, and footer.
    markdown_tables = []
    index = 0
    current_section = 0
    while index < len(lines):
        line = lines[index]
        heading = re.match(r"##\s+(\d+)\.", line)
        if heading:
            current_section = int(heading.group(1))
        if line.startswith("|") and index + 1 < len(lines) and table_separator(lines[index + 1]):
            rows = [table_cells(line)]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(table_cells(lines[index]))
                index += 1
            markdown_tables.append((current_section, rows))
            continue
        index += 1

    replacements = {}

    section_tables = {section: rows for section, rows in markdown_tables}
    for table_index, section in ((0, 1), (1, 2), (3, 4)):
        if table_index >= len(template_tables) or section not in section_tables:
            continue
        table = template_tables[table_index]
        rows = section_tables[section][1:]
        values = {row[0]: row[1] for row in rows if len(row) > 1}
        for row in table.rows:
            if len(row.cells) < 2:
                continue
            label = row.cells[0].text.strip()
            if label in values and "." in row.cells[1].text:
                value = values[label]
                # Keep units already printed after the dotted placeholder.
                suffix = " frame" if " frame" in row.cells[1].text else " episode" if " episode" in row.cells[1].text else " M" if row.cells[1].text.rstrip().endswith(" M") else ""
                value = value.removesuffix(suffix).strip() if suffix else value
                replacements[row.cells[1].text] = value

    # Fill the four existing result rows in place. Never add or delete rows.
    if len(template_tables) > 2 and 3 in section_tables:
        result_rows = section_tables[3][1:]
        table = template_tables[2]
        for row_index, row in enumerate(table.rows[1:]):
            if row_index >= len(result_rows):
                continue
            values = result_rows[row_index]
            for col, cell in enumerate(row.cells):
                if col < len(values) and "." in cell.text:
                    value = values[col]
                    if col == 1:
                        value = value.removesuffix("%").strip()
                    elif col == 2:
                        value = value.replace(" điểm %", "").strip()
                    replacements[cell.text] = value
    # Preserve explanatory paragraph text and fill each existing dotted note
    # line with the corresponding generated note, leaving remaining lines intact.
    notes = [line[2:] for line in lines if line.startswith("- ")]
    note_index = 0
    for paragraph in document.paragraphs:
        if paragraph.text.startswith(".....") and note_index < len(notes):
            replacements[paragraph.text] = notes[note_index]
            note_index += 1

    # Replace complete text per cell/paragraph in the XML tree while preserving
    # all package parts and the run properties, table formatting, and sections.
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    text_key = f"{{{ns['w']}}}t"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(template) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "word/document.xml":
                root = etree.fromstring(payload)
                for container in root.xpath(".//w:tc | .//w:p", namespaces=ns):
                    text_nodes = container.xpath(".//w:t", namespaces=ns)
                    current = "".join(node.text or "" for node in text_nodes)
                    replacement = replacements.get(current)
                    if replacement is None or not text_nodes:
                        continue
                    text_nodes[0].text = replacement
                    for node in text_nodes[1:]:
                        node.text = ""
                    if replacement[:1].isspace() or replacement[-1:].isspace():
                        text_nodes[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                payload = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            target.writestr(item, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, action="append",
                        help="Output directory to scan; repeat to specify multiple directories")
    parser.add_argument("--outputs-root", type=Path, default=ROOT,
                        help="Directory under which all outputs* directories are discovered")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "README.md",
                        help="Markdown destination (default: repository-root README.md)")
    parser.add_argument("--docx-output", type=Path, default=ROOT.parent / "REPORT.docx",
                        help="DOCX destination (default: repository-root REPORT.docx)")
    parser.add_argument("--docx-template", type=Path,
                        default=ROOT.parent / "Mau_bao_cao_ket_qua_training.docx",
                        help="DOCX layout template")
    args = parser.parse_args()

    if args.outputs:
        outputs_roots = [path.resolve() for path in args.outputs]
    else:
        outputs_roots = sorted(path.resolve() for path in args.outputs_root.glob("outputs*") if path.is_dir())
    if not outputs_roots:
        parser.error(f"No outputs* directories found under {args.outputs_root}")
    runs = find_runs(outputs_roots)

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
        "| Số frame đầu vào | Không được ghi trong config.yaml của run |",
    ])
    split_summaries = []
    seen_metadata = set()
    for run in runs:
        path = metadata_path(run["config"])
        counts = metadata_split_counts(run["config"])
        if path is None or counts is None:
            continue
        metadata_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata_hash in seen_metadata:
            continue
        seen_metadata.add(metadata_hash)
        split_summaries.append(
            f"{path.name}: train {counts['train']['videos']:,} video/{counts['train']['classes']:,} lớp; "
            f"val {counts['val']['videos']:,}/{counts['val']['classes']:,}; "
            f"test {counts['test']['videos']:,}/{counts['test']['classes']:,}"
        )
    if split_summaries:
        train_eval_test = "; ".join(split_summaries)
    else:
        train_eval_test = "metadata.csv từ config.yaml không khả dụng"
    lines.extend([
        f"| Số mẫu Train / Val / Test | Từ metadata được chỉ định trong config.yaml; {train_eval_test} |",
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
    flops = [r["flops_per_episode_profiled"] for r in eval_results
             if r.get("flops_per_episode_profiled") is not None]

    def million(values):
        return f"{values[-1] / 1e6:.3f} M" if values else "chưa được lưu trong artifacts"

    lines.extend([
        f"| Tổng số tham số | {million(parameter_counts)} |",
        f"| Số tham số được huấn luyện | {million(trainable_counts)} |",
        (f"| FLOPs | {flops[-1] / 1e9:.3f} GFLOPs/episode (torch profiler; không gồm preprocessing) |"
         if flops else "| FLOPs | Chưa có phép đo trong artifacts |"),
        (f"| Inference latency | {latencies[-1] * 1000:.2f} ms/episode (có cache I/O; "
         "không phải ms/video) |" if latencies else "| Inference latency | Chưa có phép đo trong artifacts |"),
        (f"| Peak inference VRAM | {peak_bytes[-1] / (1024 ** 3):.3f} GB |" if peak_bytes
         else "| Peak inference VRAM | Chưa có phép đo evaluation được lưu |"),
        "",
        "## 5. Ghi chú thực nghiệm",
        "",
    ])
    lines.append("- Dataset split được đếm từ metadata.csv theo đường dẫn data.metadata trong từng config.yaml.")
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
        "## 6. Bổ sung metric checkpoint (không train lại)",
        "",
        "Đánh giá `best.pt` chỉ chạy inference. Lệnh dưới tạo accuracy/CI test, số tham số, latency, peak VRAM và FLOPs cho một episode đại diện:",
        "",
        "```bash",
        "bash scripts/run_conda.sh evaluate --checkpoint <run>/best.pt --episodes <test_episodes.json> \\",
        "  --output <run>/test_metrics.json --device cuda --profile-flops",
        "```",
        "",
        "Sau đó chạy lại `python scripts/generate_report.py` để điền các số đã lưu trong `test_metrics.json`. FLOPs là tổng operator FLOPs từ `torch.profiler`, không gồm preprocessing; latency gồm cache I/O.",
        "",
        "---",
        "",
        f"Được tạo tự động từ {len(outputs_roots)} thư mục outputs*, config.yaml, `metrics.jsonl` và evaluation JSON.",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    write_docx(lines, args.docx_output, args.docx_template)
    print(f"Scanned {len(outputs_roots)} outputs* director(ies); generated {args.output} "
          f"and {args.docx_output} from {len(runs)} training run(s) and {evaluation_rows} result row(s).")


if __name__ == "__main__":
    main()
