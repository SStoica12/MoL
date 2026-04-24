import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from plot_benchmark_results import (
    DEFAULT_CHECKPOINT_NAME_MAP,
    DEFAULT_VSTAR_CATEGORIES,
    discover_benchmark_summaries,
    set_model_label_overrides,
)


"""
How to run:

$ python scripts/v1_5/eval/generate_vstar_table.py \
  --answers-root playground/data/eval/vstar_bench/answers \
  --output-file playground/data/eval/vstar_bench/answers/vstar_category_accuracy_table.md \
  --include-overall
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a V*Bench per-category results table from summary.json files."
    )
    parser.add_argument(
        "--answers-root",
        type=str,
        default="playground/data/eval/vstar_bench/answers",
        help="Root folder containing per-model subfolders with summary.json.",
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
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of raw model folder names to include.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category display order. Defaults to the standard V*Bench order plus extras found in summaries.",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="Also include the overall accuracy as a leading score column.",
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
        help="Placeholder for missing category scores.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for percentage scores.",
    )
    return parser.parse_args()


def extract_accuracy_value(node):
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


def discover_categories(rows, requested_categories):
    if requested_categories:
        return list(dict.fromkeys(requested_categories))

    discovered = []
    seen = set()
    preferred = list(DEFAULT_VSTAR_CATEGORIES)

    for category in preferred:
        seen.add(category)
        discovered.append(category)

    for row in rows:
        by_category = row["summary"].get("by_category", {})
        if not isinstance(by_category, dict):
            continue
        for category in by_category:
            if category in seen:
                continue
            seen.add(category)
            discovered.append(category)

    return discovered


def build_table_rows(rows, categories, decimals, missing_value, include_overall, include_raw_model_name):
    display_name_counts = Counter(row["display_model"] for row in rows)
    headers = ["Model"]
    if include_raw_model_name:
        headers.append("Checkpoint")
    if include_overall:
        headers.append("Overall")
    headers.extend(categories)

    table_rows = []
    for row in sorted(rows, key=lambda item: item["sort_key"]):
        display_name = row["display_model"]
        if display_name_counts[display_name] > 1:
            display_name = f"{display_name} ({row['model']})"

        values = [display_name]
        numeric_values = [None]
        if include_raw_model_name:
            values.append(row["model"])
            numeric_values.append(None)
        if include_overall:
            overall = extract_accuracy_value(row["summary"].get("overall"))
            values.append(format_score(overall, decimals, missing_value))
            numeric_values.append(overall)

        by_category = row["summary"].get("by_category", {})
        if not isinstance(by_category, dict):
            by_category = {}
        for category in categories:
            score = extract_accuracy_value(by_category.get(category))
            values.append(format_score(score, decimals, missing_value))
            numeric_values.append(score)
        table_rows.append({"cells": values, "numeric_cells": numeric_values})

    return headers, table_rows


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

    table = [
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
        table.append("| " + " | ".join(rendered_cells) + " |")
    return "\n".join(table) + "\n"


def write_table(output_path: Path, fmt: str, headers, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        output_path.write_text(markdown_table(headers, rows))
        return
    if fmt == "json":
        payload = [dict(zip(headers, row["cells"])) for row in rows]
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        return

    delimiter = "," if fmt == "csv" else "\t"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(row["cells"] for row in rows)


def main():
    args = parse_args()

    answers_root = Path(args.answers_root).expanduser().resolve()
    set_model_label_overrides(Path(args.checkpoint_name_map).expanduser().resolve())

    rows = discover_benchmark_summaries(answers_root, benchmark="vstar", model_names=args.models)
    if not rows:
        raise FileNotFoundError(
            f"No summary files found under {answers_root}. "
            "Expected layout: <answers-root>/<model-name>/summary.json"
        )

    categories = discover_categories(rows, args.categories)
    headers, table_rows = build_table_rows(
        rows=rows,
        categories=categories,
        decimals=args.decimals,
        missing_value=args.missing_value,
        include_overall=args.include_overall,
        include_raw_model_name=args.include_raw_model_name,
    )

    table_text = markdown_table(headers, table_rows)
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
        write_table(output_path, fmt, headers, table_rows)
        print(f"Saved {fmt} table to {output_path}")
    else:
        print(table_text, end="")


if __name__ == "__main__":
    main()
