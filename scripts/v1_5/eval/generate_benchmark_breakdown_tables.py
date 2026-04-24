import argparse
import json
import math
from collections import Counter
from pathlib import Path

from generate_benchmark_table import (
    BENCHMARK_DISPLAY_NAMES,
    BENCHMARK_ORDER,
    DEFAULT_CHECKPOINT_NAME_MAP,
    infer_dataset,
    normalize_benchmark_name,
    summary_paths,
)
from plot_benchmark_results import (
    METRIC_KEY_ORDER,
    accuracy_bucket,
    benchmark_category_order,
    benchmark_row_priority,
    infer_model_name,
    metric_display_name,
    model_metadata,
    set_model_label_overrides,
)


SKIP_SUMMARY_KEYS = {
    "dataset",
    "metric",
    "overall",
    "missing_predictions",
    "unparsable_predictions",
    "questions_without_ground_truth",
    "duplicates_ignored",
    "official_accuracy_components",
    "bias_stats",
}

EXTRA_METRIC_KEY_ORDER = [
    "by_difficulty",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one markdown breakdown table file per benchmark from eval summaries."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="playground/data/eval",
        help="Root eval directory to scan recursively for summary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground/data/eval/breakdown_tables",
        help="Output directory for per-benchmark markdown files.",
    )
    parser.add_argument(
        "--checkpoint-name-map",
        type=str,
        default=str(DEFAULT_CHECKPOINT_NAME_MAP),
        help="JSON file mapping checkpoint paths to display names.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional subset of benchmarks to generate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of raw model folder names to include.",
    )
    parser.add_argument(
        "--include-raw-model-name",
        action="store_true",
        help="Add a raw checkpoint basename column after the display name.",
    )
    parser.add_argument(
        "--missing-value",
        type=str,
        default="-",
        help="Placeholder for missing scores.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for percentage scores.",
    )
    return parser.parse_args()


def benchmark_sort_key(benchmark):
    if benchmark in BENCHMARK_ORDER:
        return (0, BENCHMARK_ORDER.index(benchmark), benchmark)
    return (1, benchmark)


def benchmark_display_name(benchmark):
    return BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)


def extract_score(node):
    if isinstance(node, (int, float)):
        return float(node)

    if not isinstance(node, dict):
        return None

    for key in ("accuracy", "reviewed_accuracy", "score", "value"):
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    correct = node.get("correct")
    total = node.get("total")
    if isinstance(correct, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return float(correct) / float(total)

    score_pct = node.get("score_pct")
    if isinstance(score_pct, (int, float)):
        return float(score_pct) / 100.0

    return None


def format_score(score, decimals, missing_value):
    if score is None:
        return missing_value
    return f"{score * 100:.{decimals}f}"


def load_rows(eval_root: Path, benchmarks=None, model_names=None):
    allowed_benchmarks = None
    if benchmarks:
        allowed_benchmarks = {
            normalize_benchmark_name(benchmark)
            for benchmark in benchmarks
            if normalize_benchmark_name(benchmark)
        }

    allowed_models = set(model_names) if model_names else None
    deduped = {}

    for path in summary_paths(eval_root):
        try:
            with open(path, "r") as f:
                summary = json.load(f)
        except Exception:
            continue

        dataset = infer_dataset(summary, path)
        dataset = normalize_benchmark_name(dataset)
        if not dataset:
            continue
        if allowed_benchmarks and dataset not in allowed_benchmarks:
            continue

        model_name = infer_model_name(path)
        if allowed_models and model_name not in allowed_models:
            continue

        metadata = model_metadata(model_name)
        row = {
            "dataset": dataset,
            "dataset_priority": benchmark_row_priority(dataset, path),
            "model": model_name,
            "display_model": metadata["label"],
            "sort_key": metadata["sort_key"],
            "summary_path": path,
            "summary": summary,
        }
        key = (dataset, model_name)
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = row
            continue
        if row["dataset_priority"] > previous["dataset_priority"]:
            deduped[key] = row
            continue
        if row["dataset_priority"] == previous["dataset_priority"] and str(path) < str(previous["summary_path"]):
            deduped[key] = row

    grouped = {}
    for row in deduped.values():
        grouped.setdefault(row["dataset"], []).append(row)
    for dataset in grouped:
        grouped[dataset].sort(key=lambda item: item["sort_key"])
    return grouped


def metric_bucket_for_row(row, metric_key):
    cache = row.setdefault("_metric_bucket_cache", {})
    if metric_key not in cache:
        cache[metric_key] = accuracy_bucket(row["summary"], metric_key)
    return cache[metric_key]


def overall_score_for_row(row):
    if "_overall_score" not in row:
        row["_overall_score"] = extract_score(row["summary"].get("overall"))
    return row["_overall_score"]


def available_metric_keys(rows):
    found = set()
    for row in rows:
        for key, value in row["summary"].items():
            if key in SKIP_SUMMARY_KEYS or not isinstance(value, dict):
                continue
            if metric_bucket_for_row(row, key):
                found.add(key)

    ordered = []
    for key in METRIC_KEY_ORDER + EXTRA_METRIC_KEY_ORDER:
        if key in found:
            ordered.append(key)
            found.remove(key)
    ordered.extend(sorted(found))
    return ordered


def ordered_categories(rows, benchmark, metric_key):
    categories = []
    seen = set()

    for category in benchmark_category_order(benchmark, metric_key):
        seen.add(category)
        categories.append(category)

    for row in rows:
        for category in metric_bucket_for_row(row, metric_key):
            if category in seen:
                continue
            seen.add(category)
            categories.append(category)

    return categories


def display_name_for_row(row, display_name_counts):
    display_name = row["display_model"]
    if display_name_counts[display_name] > 1:
        return f"{display_name} ({row['model']})"
    return display_name


def build_table_rows(rows, headers, value_getters, decimals, missing_value):
    table_rows = []
    for row in rows:
        cells = []
        numeric_cells = []
        for getter in value_getters:
            rendered, numeric = getter(row, decimals, missing_value)
            cells.append(rendered)
            numeric_cells.append(numeric)
        table_rows.append({"cells": cells, "numeric_cells": numeric_cells})
    return {"headers": headers, "rows": table_rows}


def markdown_table(headers, rows):
    column_maxima = [None] * len(headers)
    for col_idx in range(len(headers)):
        values = [
            row["numeric_cells"][col_idx]
            for row in rows
            if row["numeric_cells"][col_idx] is not None
        ]
        if values:
            column_maxima[col_idx] = max(values)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered_cells = []
        for col_idx, cell in enumerate(row["cells"]):
            numeric_value = row["numeric_cells"][col_idx]
            column_max = column_maxima[col_idx]
            if (
                numeric_value is not None
                and column_max is not None
                and math.isclose(numeric_value, column_max, rel_tol=1e-12, abs_tol=1e-12)
            ):
                rendered_cells.append(f"***{cell}***")
            else:
                rendered_cells.append(cell)
        lines.append("| " + " | ".join(rendered_cells) + " |")
    return "\n".join(lines) + "\n"


def overall_table(rows, include_raw_model_name, decimals, missing_value):
    display_name_counts = Counter(row["display_model"] for row in rows)

    headers = ["Model"]
    if include_raw_model_name:
        headers.append("Checkpoint")
    headers.append("Overall")

    value_getters = [
        lambda row, _decimals, _missing: (display_name_for_row(row, display_name_counts), None),
    ]
    if include_raw_model_name:
        value_getters.append(lambda row, _decimals, _missing: (row["model"], None))
    value_getters.append(
        lambda row, table_decimals, table_missing: (
            format_score(overall_score_for_row(row), table_decimals, table_missing),
            overall_score_for_row(row),
        )
    )
    return build_table_rows(rows, headers, value_getters, decimals, missing_value)


def metric_table(rows, benchmark, metric_key, include_raw_model_name, decimals, missing_value):
    display_name_counts = Counter(row["display_model"] for row in rows)
    categories = ordered_categories(rows, benchmark, metric_key)
    if not categories:
        return None

    headers = ["Model"]
    if include_raw_model_name:
        headers.append("Checkpoint")
    headers.extend(categories)

    value_getters = [
        lambda row, _decimals, _missing: (display_name_for_row(row, display_name_counts), None),
    ]
    if include_raw_model_name:
        value_getters.append(lambda row, _decimals, _missing: (row["model"], None))
    for category in categories:
        value_getters.append(
            lambda row, table_decimals, table_missing, metric=metric_key, label=category: (
                format_score(metric_bucket_for_row(row, metric).get(label), table_decimals, table_missing),
                metric_bucket_for_row(row, metric).get(label),
            )
        )

    return build_table_rows(rows, headers, value_getters, decimals, missing_value)


def benchmark_markdown(benchmark, rows, include_raw_model_name, decimals, missing_value):
    lines = [
        f"# {benchmark_display_name(benchmark)} Breakdown Tables",
        "",
        f"Baselines: {len(rows)}",
        "",
        "Best score per column is shown as `***value***`.",
        "",
        "## Overall",
        "",
        markdown_table(**overall_table(rows, include_raw_model_name, decimals, missing_value)).rstrip(),
        "",
    ]

    for metric_key in available_metric_keys(rows):
        table = metric_table(rows, benchmark, metric_key, include_raw_model_name, decimals, missing_value)
        if table is None:
            continue
        lines.extend(
            [
                f"## {metric_display_name(metric_key)}",
                "",
                markdown_table(**table).rstrip(),
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_model_label_overrides(Path(args.checkpoint_name_map).expanduser().resolve())
    grouped_rows = load_rows(eval_root, benchmarks=args.benchmarks, model_names=args.models)
    if not grouped_rows:
        raise FileNotFoundError(f"No benchmark summaries found under {eval_root}")

    for benchmark in sorted(grouped_rows, key=benchmark_sort_key):
        output_path = output_dir / f"{benchmark}_breakdown_tables.md"
        output_path.write_text(
            benchmark_markdown(
                benchmark=benchmark,
                rows=grouped_rows[benchmark],
                include_raw_model_name=args.include_raw_model_name,
                decimals=args.decimals,
                missing_value=args.missing_value,
            )
        )
        print(f"Saved markdown breakdown tables to {output_path}")


if __name__ == "__main__":
    main()
