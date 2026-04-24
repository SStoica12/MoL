import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vstar_router_analysis_utils import (
    DEFAULT_ANALYSIS_FILE_NAME,
    DEFAULT_VSTAR_CATEGORIES,
    discover_vstar_attention_models,
    load_json,
    load_jsonl,
)


FAMILY_ORDER = ["attn-layer", "attn-token", "patch_layer", "hybrid"]
CATEGORY_ORDER = ["overall"] + DEFAULT_VSTAR_CATEGORIES
FAMILY_COLORS = {
    "attn-layer": "#0f766e",
    "attn-token": "#1d4ed8",
    "patch_layer": "#b45309",
    "hybrid": "#b91c1c",
}
CATEGORY_COLORS = {
    "overall": "#111827",
    "GPT4V-hard": "#7c3aed",
    "OCR": "#0f766e",
    "direct_attributes": "#d97706",
    "relative_position": "#b91c1c",
}
CATEGORY_MARKERS = {
    "overall": "o",
    "GPT4V-hard": "X",
    "OCR": "s",
    "direct_attributes": "^",
    "relative_position": "D",
}
POINT_MARKERS = ["o", "x"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize V*Bench router receptive-field and reserve-shift analysis outputs."
    )
    parser.add_argument(
        "--answers-root",
        type=str,
        default="playground/data/eval/vstar_bench/answers",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="/shared/nas2/jk100/MoL/checkpoints",
    )
    parser.add_argument(
        "--analysis-file-name",
        type=str,
        default=DEFAULT_ANALYSIS_FILE_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground/data/eval/vstar_bench/router_analysis",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    return parser.parse_args()


def mean_vector(rows, key):
    values = [np.asarray(row[key], dtype=np.float64) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return np.mean(values, axis=0)


def mean_optional(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def infer_selection_mode(rows):
    counts = Counter(row.get("selection_mode") for row in rows if row.get("selection_mode") is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def bootstrap_mean_ci(values, rng, iters):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    if values.size == 1:
        value = float(values[0])
        return value, value
    samples = rng.choice(values, size=(iters, values.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    xr = rankdata(x)
    yr = rankdata(y)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt(np.sum(xr ** 2) * np.sum(yr ** 2))
    if denom <= 0:
        return None
    return float(np.sum(xr * yr) / denom)


def bootstrap_spearman_ci(category_rows, overall_rows, rng, iters):
    if not category_rows or not overall_rows:
        return None, None
    if len(category_rows) == 1 and len(overall_rows) == 1:
        return None, None
    values = []
    for _ in range(iters):
        category_sample = [category_rows[idx] for idx in rng.integers(0, len(category_rows), len(category_rows))]
        overall_sample = [overall_rows[idx] for idx in rng.integers(0, len(overall_rows), len(overall_rows))]
        delta_sampling = mean_vector(category_sample, "layer_prob_mass") - mean_vector(overall_sample, "layer_prob_mass")
        delta_rf = mean_vector(category_sample, "rf_by_layer") - mean_vector(overall_sample, "rf_by_layer")
        rho = spearman_correlation(delta_sampling, delta_rf)
        if rho is not None:
            values.append(rho)
    if not values:
        return None, None
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def category_accuracy(rows, category):
    if not rows:
        return None
    if category == "overall":
        return rows[0].get("model_overall_accuracy")
    return rows[0].get("model_category_accuracy")


def aggregate_model_category(model_name, family, rows, category, overall_rows, rng, bootstrap_iters):
    sampling_vector = mean_vector(rows, "layer_prob_mass")
    rf_vector = mean_vector(rows, "rf_by_layer")
    sampled_rf_contribution_vector = mean_vector(rows, "sampled_rf_contribution_by_layer")
    rf_routed_values = [row["rf_routed"] for row in rows]
    rf_routed_mean = float(np.mean(rf_routed_values))
    rf_routed_ci_low, rf_routed_ci_high = bootstrap_mean_ci(rf_routed_values, rng, bootstrap_iters)

    top3_layers = np.argsort(-sampling_vector)[:3].tolist()
    overall_sampling = mean_vector(overall_rows, "layer_prob_mass")
    overall_rf = mean_vector(overall_rows, "rf_by_layer")
    overall_rf_routed = float(np.mean([row["rf_routed"] for row in overall_rows]))

    if category == "overall":
        spearman = None
        spearman_ci_low = None
        spearman_ci_high = None
        rf_routed_delta = 0.0
    else:
        delta_sampling = sampling_vector - overall_sampling
        delta_rf = rf_vector - overall_rf
        spearman = spearman_correlation(delta_sampling, delta_rf)
        spearman_ci_low, spearman_ci_high = bootstrap_spearman_ci(
            rows,
            overall_rows,
            rng,
            bootstrap_iters,
        )
        rf_routed_delta = rf_routed_mean - overall_rf_routed

    reserve_enabled = any(row.get("reserve_enabled") for row in rows)
    contribution_vector = None
    reserve_displacement_vector = None
    if reserve_enabled:
        contribution_vector = mean_vector(rows, "contribution_weighted_layer_effect")
        reserve_displacement_vector = mean_vector(rows, "reserve_displacement_by_layer")

    return {
        "model_name": model_name,
        "family": family,
        "selection_mode": infer_selection_mode(rows),
        "category": category,
        "sample_count": len(rows),
        "low_support": category == "GPT4V-hard",
        "category_accuracy": category_accuracy(rows, category),
        "model_overall_accuracy": rows[0].get("model_overall_accuracy"),
        "sampling_vector": sampling_vector,
        "top3_layers": top3_layers,
        "rf_vector": rf_vector,
        "sampled_rf_contribution_vector": sampled_rf_contribution_vector,
        "rf_routed_mean": rf_routed_mean,
        "rf_routed_ci_low": rf_routed_ci_low,
        "rf_routed_ci_high": rf_routed_ci_high,
        "rf_routed_delta": rf_routed_delta,
        "sampling_rf_spearman": spearman,
        "sampling_rf_spearman_ci_low": spearman_ci_low,
        "sampling_rf_spearman_ci_high": spearman_ci_high,
        "reserve_enabled": reserve_enabled,
        "reserve_mean": mean_optional(rows, "reserve_mean"),
        "reserve_shift_l2_mean": mean_optional(rows, "reserve_shift_l2"),
        "reserve_shift_ratio_mean": mean_optional(rows, "reserve_shift_ratio"),
        "reserve_cosine_mean": mean_optional(rows, "reserve_cosine"),
        "contribution_vector": contribution_vector,
        "reserve_displacement_vector": reserve_displacement_vector,
    }


def aggregate_correctness_split(model_name, family, category, rows):
    payload = {
        "model_name": model_name,
        "family": family,
        "category": category,
        "correct_count": sum(int(row["correct"]) for row in rows),
        "incorrect_count": sum(1 - int(row["correct"]) for row in rows),
    }
    for label, subset in (
        ("correct", [row for row in rows if int(row["correct"]) == 1]),
        ("incorrect", [row for row in rows if int(row["correct"]) == 0]),
    ):
        payload[f"{label}_reserve_mean"] = mean_optional(subset, "reserve_mean")
        payload[f"{label}_reserve_shift_ratio"] = mean_optional(subset, "reserve_shift_ratio")
        payload[f"{label}_reserve_cosine"] = mean_optional(subset, "reserve_cosine")
        payload[f"{label}_rf_routed"] = mean_optional(subset, "rf_routed")
    return payload


def quartile_chunks(rows):
    rows = sorted(rows, key=lambda row: row["reserve_mean"])
    if not rows:
        return []
    total = len(rows)
    quartiles = []
    for quartile_idx in range(4):
        start = (quartile_idx * total) // 4
        end = ((quartile_idx + 1) * total) // 4
        if quartile_idx == 3:
            end = total
        chunk = rows[start:end]
        if chunk:
            quartiles.append((quartile_idx + 1, chunk))
    return quartiles


def aggregate_reserve_quartiles(model_name, family, rows):
    reserve_rows = [row for row in rows if row.get("reserve_mean") is not None]
    payloads = []
    for quartile, chunk in quartile_chunks(reserve_rows):
        payloads.append({
            "model_name": model_name,
            "family": family,
            "quartile": quartile,
            "sample_count": len(chunk),
            "reserve_mean": mean_optional(chunk, "reserve_mean"),
            "reserve_shift_ratio_mean": mean_optional(chunk, "reserve_shift_ratio"),
            "reserve_cosine_mean": mean_optional(chunk, "reserve_cosine"),
            "rf_routed_mean": mean_optional(chunk, "rf_routed"),
            "accuracy_mean": mean_optional(chunk, "correct"),
        })
    return payloads


def model_sort_key(record):
    family_idx = FAMILY_ORDER.index(record["family"]) if record["family"] in FAMILY_ORDER else len(FAMILY_ORDER)
    category_idx = CATEGORY_ORDER.index(record["category"]) if record["category"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (family_idx, record["model_name"], category_idx)


def model_display_label(model_name, family):
    tokens = [family]

    if "_k2" in model_name:
        tokens.append("k=2")
    elif "_k4" in model_name:
        tokens.append("k=4")
    elif "_k6" in model_name:
        tokens.append("k=6")
    else:
        tokens.append("k=1")

    if "disagreement-gate" in model_name:
        tokens.append("disagreement")
    elif "learnable-gate" in model_name:
        tokens.append("learnable")

    if "no-delta" in model_name:
        tokens.append("no-delta")
    if "no_reserve" in model_name or "no-reserve" in model_name:
        tokens.append("no-reserve")

    return " | ".join(tokens)


def build_model_style_map(records):
    style_map = {}
    family_counts = Counter()
    for record in sorted(records, key=model_sort_key):
        model_name = record["model_name"]
        if model_name in style_map:
            continue
        family = record["family"]
        idx = family_counts[family]
        family_counts[family] += 1
        style_map[model_name] = {
            "color": FAMILY_COLORS.get(family),
            "linestyle": "-",
            "linewidth": 1.8,
            "alpha": 0.95,
            "marker": POINT_MARKERS[idx % len(POINT_MARKERS)],
            "label": model_display_label(model_name, family),
        }
    return style_map


def is_k1_model(model_name):
    return "_k2" not in model_name and "_k4" not in model_name and "_k6" not in model_name


def include_in_sampling_plots(record):
    model_name = record["model_name"]
    family = record["family"]
    if family == "attn-token":
        return False
    if not is_k1_model(model_name):
        return False
    if family == "patch_layer":
        return "no-delta" in model_name
    if family == "hybrid":
        return "disagreement-gate" in model_name
    return True


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_sampling_rows(records):
    rows = []
    for record in sorted(records, key=model_sort_key):
        row = {
            "model_name": record["model_name"],
            "family": record["family"],
            "selection_mode": record["selection_mode"],
            "category": record["category"],
            "sample_count": record["sample_count"],
            "low_support": int(record["low_support"]),
            "category_accuracy": record["category_accuracy"],
            "top1_layer": record["top3_layers"][0] if len(record["top3_layers"]) > 0 else None,
            "top2_layer": record["top3_layers"][1] if len(record["top3_layers"]) > 1 else None,
            "top3_layer": record["top3_layers"][2] if len(record["top3_layers"]) > 2 else None,
        }
        for idx, value in enumerate(record["sampling_vector"]):
            row[f"sampling_layer_{idx}"] = float(value)
        rows.append(row)
    return rows


def flatten_rf_rows(records):
    rows = []
    for record in sorted(records, key=model_sort_key):
        row = {
            "model_name": record["model_name"],
            "family": record["family"],
            "selection_mode": record["selection_mode"],
            "category": record["category"],
            "sample_count": record["sample_count"],
            "low_support": int(record["low_support"]),
            "category_accuracy": record["category_accuracy"],
            "rf_routed_mean": record["rf_routed_mean"],
            "rf_routed_ci_low": record["rf_routed_ci_low"],
            "rf_routed_ci_high": record["rf_routed_ci_high"],
            "rf_routed_delta": record["rf_routed_delta"],
            "sampling_rf_spearman": record["sampling_rf_spearman"],
            "sampling_rf_spearman_ci_low": record["sampling_rf_spearman_ci_low"],
            "sampling_rf_spearman_ci_high": record["sampling_rf_spearman_ci_high"],
        }
        for idx, value in enumerate(record["rf_vector"]):
            row[f"rf_layer_{idx}"] = float(value)
        for idx in range(24):
            row[f"sampled_rf_contribution_layer_{idx}"] = None
        if record["sampled_rf_contribution_vector"] is not None:
            for idx, value in enumerate(record["sampled_rf_contribution_vector"]):
                row[f"sampled_rf_contribution_layer_{idx}"] = float(value)
        rows.append(row)
    return rows


def flatten_reserve_rows(records):
    rows = []
    for record in sorted(records, key=model_sort_key):
        row = {
            "model_name": record["model_name"],
            "family": record["family"],
            "selection_mode": record["selection_mode"],
            "category": record["category"],
            "sample_count": record["sample_count"],
            "reserve_enabled": int(record["reserve_enabled"]),
            "reserve_mean": record["reserve_mean"],
            "reserve_shift_l2_mean": record["reserve_shift_l2_mean"],
            "reserve_shift_ratio_mean": record["reserve_shift_ratio_mean"],
            "reserve_cosine_mean": record["reserve_cosine_mean"],
        }
        contribution_vector = record["contribution_vector"]
        reserve_displacement_vector = record["reserve_displacement_vector"]
        for idx in range(24):
            row[f"contribution_layer_{idx}"] = None
            row[f"reserve_displacement_layer_{idx}"] = None
        if contribution_vector is not None:
            for idx, value in enumerate(contribution_vector):
                row[f"contribution_layer_{idx}"] = float(value)
        if reserve_displacement_vector is not None:
            for idx, value in enumerate(reserve_displacement_vector):
                row[f"reserve_displacement_layer_{idx}"] = float(value)
        rows.append(row)
    return rows


def plot_receptive_field_curves(records, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for axis, category in zip(axes, CATEGORY_ORDER):
        axis.set_title(category)
        for family in FAMILY_ORDER:
            family_vectors = [
                record["rf_vector"]
                for record in records
                if record["family"] == family and record["category"] == category
            ]
            if not family_vectors:
                continue
            mean_vector_family = np.mean(family_vectors, axis=0)
            normalized_depth = np.linspace(0.0, 1.0, len(mean_vector_family))
            axis.plot(
                normalized_depth,
                mean_vector_family,
                label=family,
                color=FAMILY_COLORS.get(family),
                marker="o",
                linewidth=1.5,
            )
        axis.set_xlabel("Normalized Depth")
        axis.set_ylabel("Receptive Field")
    for axis in axes[len(CATEGORY_ORDER):]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sampled_rf_contribution_curves(sample_rows_by_model, family_by_model, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for axis, family in zip(axes, FAMILY_ORDER):
        axis.set_title(family)
        family_rows = []
        for model_name, rows in sample_rows_by_model.items():
            if family_by_model.get(model_name) == family:
                family_rows.extend(rows)

        by_category = defaultdict(list)
        for row in family_rows:
            by_category[row["category"]].append(row)
        by_category["overall"] = family_rows

        for category in CATEGORY_ORDER:
            category_rows = by_category.get(category, [])
            if not category_rows:
                continue
            contribution_vector = mean_vector(category_rows, "sampled_rf_contribution_by_layer")
            if contribution_vector is None:
                continue
            normalized_depth = np.linspace(0.0, 1.0, len(contribution_vector))
            axis.plot(
                normalized_depth,
                contribution_vector,
                label=category,
                color=CATEGORY_COLORS.get(category),
                marker=CATEGORY_MARKERS.get(category, "o"),
                linewidth=1.5,
                markersize=4,
            )
        axis.set_xlabel("Normalized Depth")
        axis.set_ylabel("Sampled RF Contribution")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_layer_sampling_overall(records, output_path):
    overall_records = [
        record
        for record in sorted(records, key=model_sort_key)
        if record["category"] == "overall" and include_in_sampling_plots(record)
    ]
    style_map = build_model_style_map(overall_records)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for record in overall_records:
        sampling_vector = record["sampling_vector"]
        normalized_depth = np.linspace(0.0, 1.0, len(sampling_vector))
        style = style_map[record["model_name"]]
        ax.plot(
            normalized_depth,
            sampling_vector,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            marker=style["marker"],
            markersize=6,
        )

    ax.set_title("Overall Layer Sampling by Baseline (k=1)")
    ax.set_xlabel("Normalized Depth")
    ax.set_ylabel("Average Sampling Mass")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layer_sampling_by_category(records, output_path):
    plot_categories = [category for category in CATEGORY_ORDER if category != "overall"]
    filtered_records = [record for record in records if include_in_sampling_plots(record)]
    style_map = build_model_style_map(filtered_records)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for axis, category in zip(axes, plot_categories):
        category_records = [
            record
            for record in sorted(records, key=model_sort_key)
            if record["category"] == category and include_in_sampling_plots(record)
        ]
        axis.set_title(category)
        for record in category_records:
            sampling_vector = record["sampling_vector"]
            normalized_depth = np.linspace(0.0, 1.0, len(sampling_vector))
            style = style_map[record["model_name"]]
            axis.plot(
                normalized_depth,
                sampling_vector,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                marker=style["marker"],
                markersize=6,
            )
        axis.set_xlabel("Normalized Depth")
        axis.set_ylabel("Average Sampling Mass")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def paired_no_reserve_rows(records_by_key):
    rows = []
    for (model_name, category), record in records_by_key.items():
        if not model_name.endswith("_no_reserve"):
            continue
        paired_name = model_name[: -len("_no_reserve")]
        paired = records_by_key.get((paired_name, category))
        if paired is None:
            continue
        rows.append({
            "model_name_no_reserve": model_name,
            "model_name_reserve": paired_name,
            "family": record["family"],
            "category": category,
            "accuracy_no_reserve": record["category_accuracy"],
            "accuracy_reserve": paired["category_accuracy"],
            "accuracy_delta_reserve_minus_no_reserve": (
                None
                if record["category_accuracy"] is None or paired["category_accuracy"] is None
                else paired["category_accuracy"] - record["category_accuracy"]
            ),
            "rf_routed_no_reserve": record["rf_routed_mean"],
            "rf_routed_reserve": paired["rf_routed_mean"],
            "rf_routed_delta_reserve_minus_no_reserve": (
                paired["rf_routed_mean"] - record["rf_routed_mean"]
            ),
        })
    return rows


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    answers_root = Path(args.answers_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runnable, appendix = discover_vstar_attention_models(
        answers_root=answers_root,
        checkpoint_root=checkpoint_root,
        analysis_file_name=args.analysis_file_name,
    )

    usable_models = []
    for record in runnable:
        if record["has_analysis"]:
            usable_models.append(record)
        else:
            missing_record = dict(record)
            missing_record["issues"] = ["missing_analysis"]
            appendix.append(missing_record)

    if not usable_models:
        raise FileNotFoundError(
            f"No runnable V*Bench router analysis files found under {answers_root} "
            f"with analysis file name '{args.analysis_file_name}'."
        )

    aggregate_records = []
    correctness_rows = []
    quartile_rows = []
    sample_rows_by_model = {}
    family_by_model = {}

    for record in usable_models:
        sample_rows = load_jsonl(record["analysis_path"])
        if not sample_rows:
            empty_record = dict(record)
            empty_record["issues"] = ["empty_analysis"]
            appendix.append(empty_record)
            continue
        sample_rows_by_model[record["model_name"]] = sample_rows
        family_by_model[record["model_name"]] = record["family"]
        by_category = defaultdict(list)
        for row in sample_rows:
            by_category[row["category"]].append(row)
        by_category["overall"] = sample_rows

        overall_rows = by_category["overall"]
        quartile_rows.extend(
            aggregate_reserve_quartiles(record["model_name"], record["family"], overall_rows)
        )

        for category in CATEGORY_ORDER:
            rows = by_category.get(category, [])
            if not rows:
                continue
            aggregate_records.append(
                aggregate_model_category(
                    model_name=record["model_name"],
                    family=record["family"],
                    rows=rows,
                    category=category,
                    overall_rows=overall_rows,
                    rng=rng,
                    bootstrap_iters=args.bootstrap_iters,
                )
            )
            correctness_rows.append(
                aggregate_correctness_split(
                    model_name=record["model_name"],
                    family=record["family"],
                    category=category,
                    rows=rows,
                )
            )

    if not aggregate_records:
        raise RuntimeError("No aggregate records were produced from the analysis files.")

    sampling_csv = flatten_sampling_rows(aggregate_records)
    rf_csv = flatten_rf_rows(aggregate_records)
    reserve_csv = flatten_reserve_rows(aggregate_records)

    write_csv(
        output_dir / "model_category_sampling.csv",
        fieldnames=list(sampling_csv[0].keys()),
        rows=sampling_csv,
    )
    write_csv(
        output_dir / "model_category_rf.csv",
        fieldnames=list(rf_csv[0].keys()),
        rows=rf_csv,
    )
    write_csv(
        output_dir / "model_category_reserve.csv",
        fieldnames=list(reserve_csv[0].keys()),
        rows=reserve_csv,
    )
    write_csv(
        output_dir / "model_category_correctness_split.csv",
        fieldnames=list(correctness_rows[0].keys()),
        rows=sorted(correctness_rows, key=lambda row: (row["family"], row["model_name"], row["category"])),
    )
    if quartile_rows:
        write_csv(
            output_dir / "model_reserve_quartiles.csv",
            fieldnames=list(quartile_rows[0].keys()),
            rows=sorted(quartile_rows, key=lambda row: (row["family"], row["model_name"], row["quartile"])),
        )

    appendix_rows = []
    for record in sorted(appendix, key=lambda row: (row["family"], row["model_name"])):
        summary = None
        if record.get("summary_path") and Path(record["summary_path"]).exists():
            summary = load_json(Path(record["summary_path"]))
        appendix_rows.append({
            "model_name": record["model_name"],
            "family": record["family"],
            "selection_mode": record.get("selection_mode"),
            "issues": ",".join(record.get("issues", [])),
            "checkpoint_path": str(record.get("checkpoint_path", "")),
            "analysis_path": str(record.get("analysis_path", "")),
            "overall_accuracy": None if summary is None else summary.get("overall", {}).get("accuracy"),
        })
    if appendix_rows:
        write_csv(
            output_dir / "appendix_manifest.csv",
            fieldnames=list(appendix_rows[0].keys()),
            rows=appendix_rows,
        )

    records_by_key = {
        (record["model_name"], record["category"]): record
        for record in aggregate_records
    }
    paired_rows = paired_no_reserve_rows(records_by_key)
    if paired_rows:
        write_csv(
            output_dir / "paired_no_reserve_ablation.csv",
            fieldnames=list(paired_rows[0].keys()),
            rows=sorted(paired_rows, key=lambda row: (row["family"], row["model_name_no_reserve"], row["category"])),
        )

    sorted_records = sorted(aggregate_records, key=model_sort_key)
    plot_receptive_field_curves(
        sorted_records,
        figures_dir / "receptive_field_curves.png",
    )
    plot_sampled_rf_contribution_curves(
        sample_rows_by_model,
        family_by_model,
        figures_dir / "sampled_rf_contribution_curves.png",
    )
    plot_layer_sampling_overall(
        sorted_records,
        figures_dir / "layer_sampling_overall.png",
    )
    plot_layer_sampling_by_category(
        sorted_records,
        figures_dir / "layer_sampling_by_category.png",
    )

    manifest = {
        "usable_models": [record["model_name"] for record in usable_models],
        "num_usable_models": len(usable_models),
        "num_appendix_models": len(appendix_rows),
        "categories": CATEGORY_ORDER,
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved aggregate tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
