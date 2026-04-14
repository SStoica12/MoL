import argparse
from pathlib import Path

from plot_benchmark_results import DEFAULT_VSTAR_CATEGORIES, discover_benchmark_summaries, plot_metric_rows


"""
How to run:

$ python scripts/v1_5/eval/plot_vstar_results.py \
  --answers-root playground/data/eval/vstar_bench/answers \
  --output playground/data/eval/vstar_bench/answers/vstar_category_accuracy.png \
  --include-overall
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped bar chart for V*Bench category accuracy across models."
    )
    parser.add_argument(
        "--answers-root",
        type=str,
        default="playground/data/eval/vstar_bench/answers",
        help="Root folder containing per-model subfolders with summary.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playground/data/eval/vstar_bench/answers/vstar_category_accuracy.png",
        help="Output image path (.png recommended).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model folder names to plot.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_VSTAR_CATEGORIES,
        help="Category display order.",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="Also include the overall accuracy as an extra category.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="V*Bench Accuracy by Category",
        help="Chart title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    answers_root = Path(args.answers_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = discover_benchmark_summaries(answers_root, benchmark="vstar", model_names=args.models)
    if not rows:
        raise FileNotFoundError(
            f"No summary files found under {answers_root}. "
            "Expected layout: <answers-root>/<model-name>/summary.json"
        )

    plot_metric_rows(
        dataset_rows=rows,
        benchmark="vstar",
        metric_key="by_category",
        output_path=output_path,
        categories=args.categories,
        include_overall=args.include_overall,
        max_categories_per_row=6,
        title=args.title,
    )

    print(f"Saved plot to: {output_path}")
    print("Models plotted:")
    for row in rows:
        overall = row["summary"].get("overall", {}).get("accuracy")
        overall_str = "NA" if overall is None else f"{100.0 * overall:.2f}%"
        print(f"  - {row['display_model']}: overall={overall_str} (raw={row['model']})")


if __name__ == "__main__":
    main()
