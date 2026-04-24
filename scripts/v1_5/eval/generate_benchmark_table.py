import argparse
import csv
import json
from pathlib import Path


DEFAULT_CHECKPOINT_NAME_MAP = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "llava-v1.5-13b-moe-finetune_display_names.json"
)

BENCHMARK_DISPLAY_NAMES = {
    "charxiv": "CharXiv",
    "countbenchqa": "CountBenchQA",
    "cvbench": "CV-Bench",
    "chartqa": "ChartQA",
    "docvqa": "DocVQA",
    "dtd": "DTD",
    "gqa": "GQA",
    "hallusionbench": "HallusionBench",
    "hrbench4k": "HRBench4K",
    "hrbench8k": "HRBench8K",
    "mmbench": "MMBench",
    "mmstar": "MMStar",
    "mmvet": "MMVet",
    "mmvp": "MMVP",
    "mmmu": "MMMU",
    "mme": "MME",
    "mme_realworld": "MME-RealWorld",
    "naturalbench": "NaturalBench",
    "pope": "POPE",
    "realworldqa": "RealWorldQA",
    "seed": "SEED",
    "textvqa": "TextVQA",
    "visualoverload": "VisualOverload",
    "vizwiz": "VizWiz",
    "vqav2": "VQAv2",
    "vstar": "V*Bench",
}

BENCHMARK_ORDER = [
    "mmbench",
    "mmmu",
    "mmstar",
    "mmvet",
    "mmvp",
    "cvbench",
    "naturalbench",
    "charxiv",
    "visualoverload",
    "countbenchqa",
    "gqa",
    "seed",
    "pope",
    "textvqa",
    "docvqa",
    "mme",
    "mme_realworld",
    "vizwiz",
    "vqav2",
    "hallusionbench",
    "realworldqa",
    "chartqa",
    "hrbench4k",
    "hrbench8k",
    "vstar",
    "dtd",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a single overall-score benchmark table from summary.json files."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="playground/data/eval",
        help="Root eval directory to scan recursively for summary files.",
    )
    parser.add_argument(
        "--checkpoint-name-map",
        type=str,
        default=str(DEFAULT_CHECKPOINT_NAME_MAP),
        help="JSON file mapping checkpoint paths to display names.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output file. Extension determines format when --format is omitted.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "tsv", "json"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--missing-value",
        type=str,
        default="-",
        help="Placeholder for missing benchmark scores.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for percentage scores.",
    )
    parser.add_argument(
        "--include-raw-model-name",
        action="store_true",
        help="Add a raw checkpoint basename column after the display name.",
    )
    return parser.parse_args()


def load_name_map(path: Path):
    with open(path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint label map must be a JSON object: {path}")

    ordered = []
    for checkpoint_path, display_name in raw.items():
        if not isinstance(checkpoint_path, str) or not isinstance(display_name, str):
            continue
        checkpoint_path = checkpoint_path.strip()
        display_name = display_name.strip()
        if not checkpoint_path or not display_name:
            continue
        ordered.append((Path(checkpoint_path).name, display_name))
    return ordered


def summary_paths(eval_root: Path):
    found = list(eval_root.rglob("summary.json")) + list(eval_root.rglob("*_summary.json"))
    unique = []
    seen = set()
    for path in sorted(found):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def infer_model_name(summary_path: Path):
    name = summary_path.name
    if name.endswith("_summary.json"):
        return name[: -len("_summary.json")]
    if name == "summary.json":
        return summary_path.parent.name
    return summary_path.stem


def normalize_benchmark_name(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "doc vqa": "docvqa",
        "doc-vqa": "docvqa",
        "doc_vqa": "docvqa",
        "cv-bench": "cvbench",
        "cv_bench": "cvbench",
        "char-xiv": "charxiv",
        "char_xiv": "charxiv",
        "countbench": "countbenchqa",
        "countbench-qa": "countbenchqa",
        "countbench_qa": "countbenchqa",
        "hr-bench-4k": "hrbench4k",
        "hr-bench-8k": "hrbench8k",
        "hrbench_4k": "hrbench4k",
        "hrbench_8k": "hrbench8k",
        "mm-star": "mmstar",
        "mm_star": "mmstar",
        "mm-vet": "mmvet",
        "mm-vp": "mmvp",
        "mme realworld": "mme_realworld",
        "mme-realworld": "mme_realworld",
        "mme_real_world": "mme_realworld",
        "mme-real-world": "mme_realworld",
        "natural-bench": "naturalbench",
        "natural_bench": "naturalbench",
        "visual-overload": "visualoverload",
        "visual_overload": "visualoverload",
        "v*bench": "vstar",
        "vstar_bench": "vstar",
    }
    text = aliases.get(text, text)
    return text or None


def infer_dataset(summary, summary_path: Path):
    dataset = normalize_benchmark_name(summary.get("dataset"))
    if dataset:
        return dataset

    path_txt = summary_path.as_posix().lower()
    if "/hrbench4k/" in path_txt:
        return "hrbench4k"
    if "/hrbench8k/" in path_txt:
        return "hrbench8k"
    if "/cvbench/" in path_txt or "/cv-bench/" in path_txt:
        return "cvbench"
    if "/charxiv/" in path_txt or "/char-xiv/" in path_txt:
        return "charxiv"
    if "/countbenchqa/" in path_txt or "/countbench-qa/" in path_txt or "/countbench_qa/" in path_txt:
        return "countbenchqa"
    if "/visualoverload/" in path_txt or "/visual-overload/" in path_txt:
        return "visualoverload"
    if "/naturalbench/" in path_txt or "/natural-bench/" in path_txt:
        return "naturalbench"
    if "/mmstar/" in path_txt:
        return "mmstar"
    if "/mmvet/" in path_txt or "/mm-vet/" in path_txt:
        return "mmvet"
    if "/mmvp/" in path_txt:
        return "mmvp"
    if "/hallusionbench/" in path_txt:
        return "hallusionbench"
    if "/realworldqa/" in path_txt:
        return "realworldqa"
    if "/mme-realworld/" in path_txt or "/mme_realworld/" in path_txt:
        return "mme_realworld"
    if "/chartqa/" in path_txt:
        return "chartqa"
    if "/docvqa/" in path_txt:
        return "docvqa"
    if "/vstar_bench/" in path_txt:
        return "vstar"
    if "/mmmu/" in path_txt:
        return "mmmu"
    if "/mmbench" in path_txt:
        return "mmbench"
    if "/gqa" in path_txt:
        return "gqa"
    if "/seed" in path_txt:
        return "seed"
    if "/pope" in path_txt:
        return "pope"
    if "/textvqa" in path_txt:
        return "textvqa"
    if "/mme" in path_txt:
        return "mme"
    if "/vizwiz" in path_txt:
        return "vizwiz"
    if "/vqav2" in path_txt:
        return "vqav2"
    if "/dtd/" in path_txt:
        return "dtd"
    return None


def extract_overall_accuracy(summary):
    overall = summary.get("overall")
    if isinstance(overall, dict):
        for key in ("accuracy", "reviewed_accuracy", "score", "value"):
            value = overall.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        score_pct = overall.get("score_pct")
        if isinstance(score_pct, (int, float)):
            return float(score_pct) / 100.0
    if isinstance(overall, (int, float)):
        return float(overall)
    return None


def format_score(score, decimals, missing_value):
    if score is None:
        return missing_value
    return f"{score * 100:.{decimals}f}"


def benchmark_sort_key(benchmark):
    if benchmark in BENCHMARK_ORDER:
        return (0, BENCHMARK_ORDER.index(benchmark), benchmark)
    return (1, benchmark)


def benchmark_label(benchmark):
    return BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)


def collect_scores(eval_root: Path, model_rows):
    tracked_models = {model_name for model_name, _ in model_rows}
    scores = {model_name: {} for model_name, _ in model_rows}
    discovered_benchmarks = set()

    for path in summary_paths(eval_root):
        try:
            with open(path, "r") as f:
                summary = json.load(f)
        except Exception:
            continue

        model_name = infer_model_name(path)
        if model_name not in tracked_models:
            continue

        dataset = infer_dataset(summary, path)
        if not dataset:
            continue

        discovered_benchmarks.add(dataset)
        scores[model_name][dataset] = extract_overall_accuracy(summary)

    ordered_benchmarks = sorted(discovered_benchmarks, key=benchmark_sort_key)
    return ordered_benchmarks, scores


def build_table_rows(model_rows, benchmarks, scores, decimals, missing_value, include_raw_model_name):
    headers = ["Model"]
    if include_raw_model_name:
        headers.append("Checkpoint")
    headers.extend(benchmark_label(benchmark) for benchmark in benchmarks)

    rows = []
    for model_name, display_name in model_rows:
        row = [display_name]
        if include_raw_model_name:
            row.append(model_name)
        for benchmark in benchmarks:
            row.append(format_score(scores.get(model_name, {}).get(benchmark), decimals, missing_value))
        rows.append(row)
    return headers, rows


def markdown_table(headers, rows):
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table) + "\n"


def write_table(output_path: Path, fmt: str, headers, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        output_path.write_text(markdown_table(headers, rows))
        return
    if fmt == "json":
        payload = [dict(zip(headers, row)) for row in rows]
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        return

    delimiter = "," if fmt == "csv" else "\t"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    args = parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    name_map_path = Path(args.checkpoint_name_map).expanduser().resolve()
    model_rows = load_name_map(name_map_path)
    benchmarks, scores = collect_scores(eval_root, model_rows)
    headers, rows = build_table_rows(
        model_rows=model_rows,
        benchmarks=benchmarks,
        scores=scores,
        decimals=args.decimals,
        missing_value=args.missing_value,
        include_raw_model_name=args.include_raw_model_name,
    )

    table_text = markdown_table(headers, rows)

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        fmt = args.format
        if output_path.suffix.lower() == ".csv":
            fmt = "csv"
        elif output_path.suffix.lower() == ".tsv":
            fmt = "tsv"
        elif output_path.suffix.lower() == ".json":
            fmt = "json"
        elif output_path.suffix.lower() in {".md", ".markdown"}:
            fmt = "markdown"
        write_table(output_path, fmt, headers, rows)
        print(f"Saved {fmt} table to {output_path}")
    else:
        print(table_text, end="")


if __name__ == "__main__":
    main()
