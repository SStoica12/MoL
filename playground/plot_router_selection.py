#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_ROOT / "plots"
DEFAULT_CHECKPOINT_NAME_MAP = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "llava-v1.5-13b-moe-finetune_display_names.json"
)
DEFAULT_MIN_NUM_LAYERS = 24

BENCHMARK_ALIASES = {
    "char_xiv": "charxiv",
    "char-xiv": "charxiv",
    "charxiv": "charxiv",
    "cv-bench": "cvbench",
    "cvbench": "cvbench",
    "chartqa": "chartqa",
    "docvqa": "docvqa",
    "dtd": "dtd",
    "hallusionbench": "hallusionbench",
    "hrbench4k": "hrbench4k",
    "hrbench8k": "hrbench8k",
    "mm_star": "mmstar",
    "mm-star": "mmstar",
    "mmstar": "mmstar",
    "mm-vet": "mmvet",
    "mmvet": "mmvet",
    "mm-vp": "mmvp",
    "mmvp": "mmvp",
    "llava_bench_in_the_wild": "llava_bench_wild",
    "llava-bench-in-the-wild": "llava_bench_wild",
    "mme_realworld": "mme_realworld",
    "mme-realworld": "mme_realworld",
    "mmmu": "mmmu",
    "natural_bench": "naturalbench",
    "natural-bench": "naturalbench",
    "naturalbench": "naturalbench",
    "realworldqa": "realworldqa",
    "vstar": "vstar",
    "vstar_bench": "vstar",
    "v*bench": "vstar",
}

BENCHMARK_DISPLAY_NAMES = {
    "charxiv": "CharXiv",
    "cvbench": "CV-Bench",
    "chartqa": "ChartQA",
    "docvqa": "DocVQA",
    "dtd": "DTD",
    "hallusionbench": "HallusionBench",
    "hrbench4k": "HRBench4K",
    "hrbench8k": "HRBench8K",
    "mmstar": "MMStar",
    "mmvet": "MMVet",
    "mmvp": "MMVP",
    "llava_bench_wild": "LLaVA-Bench (Wild)",
    "mme_realworld": "MME-RealWorld",
    "mmmu": "MMMU",
    "naturalbench": "NaturalBench",
    "realworldqa": "RealWorldQA",
    "vstar": "V*Bench",
}

BENCHMARK_ORDER = [
    "mmmu",
    "mmstar",
    "mmvet",
    "mmvp",
    "cvbench",
    "naturalbench",
    "charxiv",
    "docvqa",
    "chartqa",
    "hallusionbench",
    "realworldqa",
    "mme_realworld",
    "hrbench4k",
    "hrbench8k",
    "vstar",
    "dtd",
    "llava_bench_wild",
]

TASK_GROUP_BY_BENCHMARK = {
    "charxiv": "document_ocr_and_structured_reading",
    "cvbench": "general_visual_reasoning",
    "chartqa": "document_ocr_and_structured_reading",
    "docvqa": "document_ocr_and_structured_reading",
    "dtd": "texture_and_material_understanding",
    "hallusionbench": "hallucination_and_grounding",
    "hrbench4k": "high_res_fine_grained_visual_understanding",
    "hrbench8k": "high_res_fine_grained_visual_understanding",
    "llava_bench_wild": "open_ended_visual_chat",
    "mme_realworld": "high_res_fine_grained_visual_understanding",
    "mmmu": "general_visual_reasoning",
    "naturalbench": "general_visual_reasoning",
    "mmstar": "general_visual_reasoning",
    "mmvet": "general_visual_reasoning",
    "mmvp": "high_res_fine_grained_visual_understanding",
    "realworldqa": "general_visual_reasoning",
    "vstar": "high_res_fine_grained_visual_understanding",
}

TASK_GROUP_DISPLAY_NAMES = {
    "document_ocr_and_structured_reading": "Document / OCR / Structured Reading",
    "general_visual_reasoning": "General Visual Reasoning",
    "hallucination_and_grounding": "Hallucination / Grounding",
    "high_res_fine_grained_visual_understanding": (
        "High-Res / Fine-Grained Visual Understanding"
    ),
    "open_ended_visual_chat": "Open-Ended Visual Chat",
    "other": "Other",
    "texture_and_material_understanding": "Texture / Material Understanding",
}

TASK_GROUP_ORDER = [
    "high_res_fine_grained_visual_understanding",
    "document_ocr_and_structured_reading",
    "hallucination_and_grounding",
    "general_visual_reasoning",
    "texture_and_material_understanding",
    "open_ended_visual_chat",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot layer-selection behavior from router metadata stored in "
            "predictions.jsonl files."
        )
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default=str(DEFAULT_EVAL_ROOT),
        help="Root evaluation directory to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plot images will be saved.",
    )
    parser.add_argument(
        "--checkpoint-name-map",
        type=str,
        default=str(DEFAULT_CHECKPOINT_NAME_MAP),
        help="Optional JSON map from raw checkpoint names to display labels.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of raw model folder names to include.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Optional subset of benchmarks to include. Accepts folder names or "
            "canonical names such as vstar, hrbench4k, mme_realworld."
        ),
    )
    parser.add_argument(
        "--max-model-panels-per-row",
        type=int,
        default=4,
        help="Maximum number of checkpoint panels per row for layer heatmaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser.parse_args()


def canonicalize_benchmark(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace("*", "star")
    return BENCHMARK_ALIASES.get(name, BENCHMARK_ALIASES.get(normalized, normalized))


def wrap_label(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def load_checkpoint_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        mapping[key] = value
        mapping[Path(key).name] = value
    return mapping


def resolve_display_model_name(
    raw_name: str,
    checkpoint_name_map: dict[str, str],
    repo_root: Path,
) -> str:
    candidates = [
        raw_name,
        str((repo_root / "checkpoints" / raw_name).resolve()),
        f"liuhaotian/{raw_name}",
    ]
    for candidate in candidates:
        if candidate in checkpoint_name_map:
            return checkpoint_name_map[candidate]
    return raw_name


def infer_benchmark(predictions_path: Path, eval_root: Path) -> str | None:
    # import pdb; pdb.set_trace()
    #rel_path = predictions_path.relative_to(eval_root)
    # rel_path = Path("/shared/nas2/jk100/MoL/playground/data/eval/vstar_bench")
    # if "answers" not in rel_path.parts:
    #     return None
    #return canonicalize_benchmark(rel_path.parts[0])
    return canonicalize_benchmark(predictions_path.parent.parent.parent.name)


def infer_num_layers(router_metadata: dict[str, Any]) -> int:
    num_layers = router_metadata.get("num_layers_available")
    if isinstance(num_layers, int) and num_layers > 0:
        return num_layers

    max_layer_index = -1
    for key in ("selected_layer_prob_mass", "selected_layer_frequency", "selected_layer_counts"):
        values = router_metadata.get(key) or {}
        for layer_idx in values:
            try:
                max_layer_index = max(max_layer_index, int(layer_idx))
            except (TypeError, ValueError):
                continue

    if max_layer_index >= 0:
        return max_layer_index + 1

    selected_layers = router_metadata.get("selected_layers")
    if selected_layers:
        flat = np.asarray(selected_layers, dtype=np.int64).reshape(-1)
        if flat.size:
            return int(flat.max()) + 1

    return DEFAULT_MIN_NUM_LAYERS


def build_normalized_layer_vector(router_metadata: dict[str, Any]) -> np.ndarray | None:
    num_layers = infer_num_layers(router_metadata)
    for key in ("selected_layer_prob_mass", "selected_layer_frequency", "selected_layer_counts"):
        values = router_metadata.get(key) or {}
        if not values:
            continue
        dense = np.zeros(num_layers, dtype=np.float64)
        for layer_idx, value in values.items():
            dense[int(layer_idx)] = float(value)
        total = dense.sum()
        if total > 0:
            return dense / total

    selected_layers = router_metadata.get("selected_layers")
    if not selected_layers:
        return None

    flat = np.asarray(selected_layers, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        return None

    dense = np.bincount(flat, minlength=num_layers).astype(np.float64)
    total = dense.sum()
    if total <= 0:
        return None
    return dense / total


def extract_reserve_usage(router_metadata: dict[str, Any]) -> float:
    if "reserve_layer_prob" in router_metadata:
        return float(router_metadata["reserve_layer_prob"])

    reserve_summary = router_metadata.get("reserve_layer_prob_summary")
    if isinstance(reserve_summary, dict) and "mean" in reserve_summary:
        return float(reserve_summary["mean"])

    return float("nan")


def load_example_records(
    eval_root: Path,
    checkpoint_name_map: dict[str, str],
    model_filter: set[str] | None,
    benchmark_filter: set[str] | None,
) -> tuple[pd.DataFrame, int]:
    repo_root = Path(__file__).resolve().parents[3]
    records: list[dict[str, Any]] = []
    layer_vectors: list[np.ndarray] = []
    max_num_layers = DEFAULT_MIN_NUM_LAYERS

    pred_paths = [Path("/shared/nas2/jk100/MoL/playground/data/eval/vstar_bench/answers/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer_corrected_k=4/predictions.jsonl")]
    eval_root = Path("/shared/nas2/jk100/MoL/playground/data/eval")
    for predictions_path in sorted(pred_paths): #eval_root.rglob("predictions.jsonl")
        # import pdb; pdb.set_trace()
        benchmark = infer_benchmark(predictions_path, eval_root)
        if benchmark is None:
            continue
        if benchmark_filter and benchmark not in benchmark_filter:
            continue

        raw_model = predictions_path.parent.name
        if model_filter and raw_model not in model_filter:
            continue

        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        task_group = TASK_GROUP_BY_BENCHMARK.get(benchmark, "other")
        task_group_display = TASK_GROUP_DISPLAY_NAMES[task_group]
        display_model = resolve_display_model_name(
            raw_model,
            checkpoint_name_map,
            repo_root=repo_root,
        )

        with predictions_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                router_metadata = row.get("metadata", {}).get("router")
                if not router_metadata:
                    continue

                layer_vector = build_normalized_layer_vector(router_metadata)
                if layer_vector is None:
                    continue

                max_num_layers = max(max_num_layers, int(layer_vector.shape[0]))
                layer_vectors.append(layer_vector)

                layer_indices = np.arange(layer_vector.shape[0], dtype=np.float64)
                positive = layer_vector[layer_vector > 0]
                entropy_bits = float(-(positive * np.log2(positive)).sum())

                records.append(
                    {
                        "benchmark": benchmark,
                        "benchmark_display": benchmark_display,
                        "task_group": task_group,
                        "task_group_display": task_group_display,
                        "model": raw_model,
                        "display_model": display_model,
                        "question_id": row.get("question_id"),
                        "selection_mode": router_metadata.get("selection_mode", "unknown"),
                        "num_layers_to_choose": router_metadata.get("num_layers_to_choose"),
                        "reserve_mean": extract_reserve_usage(router_metadata),
                        "mean_layer_index": float(np.dot(layer_indices, layer_vector)),
                        "entropy_bits": entropy_bits,
                        "dominant_layer_share": float(layer_vector.max()),
                        "early_mass": float(layer_vector[:8].sum()),
                        "mid_mass": float(layer_vector[8:16].sum()),
                        "late_mass": float(layer_vector[16:].sum()),
                    }
                )

    if not records:
        raise FileNotFoundError(
            "No router-enabled predictions.jsonl files were found under "
            f"{eval_root} for the selected filters."
        )

    for record, layer_vector in zip(records, layer_vectors):
        padded = np.zeros(max_num_layers, dtype=np.float64)
        padded[: layer_vector.shape[0]] = layer_vector
        for layer_idx, value in enumerate(padded):
            record[f"layer_{layer_idx:02d}"] = float(value)

    return pd.DataFrame.from_records(records), max_num_layers


def ordered_values(values: list[str], preferred_order: list[str]) -> list[str]:
    unique_values = list(dict.fromkeys(values))
    ordered = [value for value in preferred_order if value in unique_values]
    ordered.extend(value for value in unique_values if value not in ordered)
    return ordered


def build_model_order(
    example_df: pd.DataFrame,
    checkpoint_name_map: dict[str, str],
) -> list[str]:
    checkpoint_order: dict[str, int] = {}
    for idx, key in enumerate(checkpoint_name_map):
        checkpoint_order[Path(key).name] = idx

    raw_models = list(dict.fromkeys(example_df["model"].tolist()))
    return sorted(
        raw_models,
        key=lambda name: (
            checkpoint_order.get(name, 10**9),
            example_df.loc[example_df["model"] == name, "display_model"].iloc[0],
            name,
        ),
    )


def aggregate_by_group(
    example_df: pd.DataFrame,
    group_field: str,
    layer_cols: list[str],
) -> pd.DataFrame:
    group_display_field = f"{group_field}_display"
    metric_cols = [
        "reserve_mean",
        "mean_layer_index",
        "entropy_bits",
        "dominant_layer_share",
        "early_mass",
        "mid_mass",
        "late_mass",
    ]
    agg_spec = {"example_count": ("question_id", "size")}
    for column in metric_cols + layer_cols:
        agg_spec[column] = (column, "mean")

    grouped = (
        example_df.groupby(
            [group_field, group_display_field, "model", "display_model"],
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )
    return grouped


def panel_grid(num_panels: int, max_cols: int) -> tuple[int, int]:
    ncols = min(max_cols, max(1, math.ceil(math.sqrt(num_panels))))
    nrows = int(math.ceil(num_panels / ncols))
    return nrows, ncols


def annotate_heatmap_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = "{:.2f}",
) -> None:
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = plt.get_cmap(cmap)
    font_size = max(
        5.0,
        min(
            8.5,
            190.0 / max(matrix.shape[1], 1),
            120.0 / max(matrix.shape[0], 1),
        ),
    )

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue

            red, green, blue, _ = cmap_obj(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "#0f172a" if luminance >= 0.60 else "#f8fafc"

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=font_size,
                color=text_color,
            )


def plot_layer_usage_heatmaps(
    aggregated_df: pd.DataFrame,
    group_field: str,
    group_order: list[str],
    group_display_map: dict[str, str],
    model_order: list[str],
    model_display_map: dict[str, str],
    layer_cols: list[str],
    output_path: Path,
    max_panels_per_row: int,
    dpi: int,
) -> None:
    values = aggregated_df[layer_cols].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmax = 1.0
    if finite_values.size:
        vmax = min(1.0, max(0.1, float(np.nanquantile(finite_values, 0.98))))

    ncols = min(2, len(model_order))
    nrows = int(math.ceil(len(model_order) / max(1, ncols)))
    width_per_panel = max(7.8, 0.48 * len(layer_cols) + 2.5)
    height_per_panel = max(3.6, 0.44 * len(group_order) + 2.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_panel * ncols + 1.6, height_per_panel * nrows + 1.0),
        squeeze=False,
    )

    sns.set_theme(style="whitegrid")
    mappable = None

    for panel_idx, model in enumerate(model_order):
        ax = axes.flat[panel_idx]
        matrix_df = (
            aggregated_df.loc[aggregated_df["model"] == model]
            .set_index(group_field)
            .reindex(group_order)
        )
        matrix = matrix_df[layer_cols].to_numpy(dtype=float)

        if np.isnan(matrix).all():
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            continue

        heatmap = sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            vmin=0.0,
            vmax=vmax,
            cbar=False,
            mask=np.isnan(matrix),
            linewidths=0.3,
            linecolor="white",
            xticklabels=[str(idx) for idx in range(len(layer_cols))],
            yticklabels=[group_display_map[group] for group in group_order],
        )
        mappable = heatmap.collections[0]
        annotate_heatmap_cells(
            ax=ax,
            matrix=matrix,
            cmap="mako",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(wrap_label(model_display_map[model], width=28), fontsize=13)
        ax.set_xlabel("Layer", fontsize=11)
        if panel_idx % ncols == 0:
            ax.set_ylabel("Benchmark" if group_field == "benchmark" else "Task Group", fontsize=11)
            ax.tick_params(axis="y", labelrotation=0, labelsize=10)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    for ax in axes.flat[len(model_order) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.06, wspace=0.18, hspace=0.34)
    if mappable is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
        colorbar = fig.colorbar(
            mappable,
            cax=cax,
        )
        colorbar.set_label("Mean normalized non-reserved layer mass", fontsize=11)
        colorbar.ax.tick_params(labelsize=10)

    level_name = "Benchmark" if group_field == "benchmark" else "Task Group"
    fig.suptitle(
        f"Router Layer Usage by {level_name}",
        fontsize=18,
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        "Each example is normalized over selected non-reserved layer mass before aggregation.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_metric_heatmap(
    aggregated_df: pd.DataFrame,
    group_field: str,
    group_order: list[str],
    group_display_map: dict[str, str],
    model_order: list[str],
    model_display_map: dict[str, str],
    value_col: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    transpose: bool = False,
) -> None:
    pivot = aggregated_df.pivot(
        index=group_field,
        columns="model",
        values=value_col,
    ).reindex(index=group_order, columns=model_order)
    if transpose:
        pivot = pivot.T

    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    if transpose:
        fig_width = max(8.5, 1.25 * len(group_order) + 2.5)
        fig_height = max(8.0, 0.72 * len(model_order) + 2.6)
    else:
        fig_width = max(8.0, 0.6 * len(model_order) + 2.8)
        fig_height = max(3.8, 0.45 * len(group_order) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        mask=pivot.isna(),
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": colorbar_label},
    )
    ax.set_title(title, fontsize=17)
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    if transpose:
        ax.set_xlabel("Benchmark" if group_field == "benchmark" else "Task Group", fontsize=12)
        ax.set_ylabel("Checkpoint", fontsize=12)
        ax.set_xticklabels(
            [group_display_map[group] for group in pivot.columns],
            rotation=0,
            fontsize=11,
        )
        ax.set_yticklabels(
            [wrap_label(model_display_map[model], width=24) for model in pivot.index],
            rotation=0,
            fontsize=10,
        )
    else:
        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Benchmark" if group_field == "benchmark" else "Task Group", fontsize=12)
        ax.set_xticklabels(
            [wrap_label(model_display_map[model], width=18) for model in pivot.columns],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_yticklabels(
            [group_display_map[group] for group in pivot.index],
            rotation=0,
            fontsize=10,
        )
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stats_panel(
    aggregated_df: pd.DataFrame,
    group_field: str,
    group_order: list[str],
    group_display_map: dict[str, str],
    model_order: list[str],
    model_display_map: dict[str, str],
    output_path: Path,
    dpi: int,
    num_layers: int,
) -> None:
    metric_specs = [
        ("mean_layer_index", "Mean Layer Index", "viridis", 0.0, max(num_layers - 1, 1)),
        ("entropy_bits", "Entropy (bits)", "magma", 0.0, math.log2(max(num_layers, 2))),
        ("dominant_layer_share", "Dominant Layer Share", "crest", 0.0, 1.0),
        ("early_mass", "Early Mass (0-7)", "Blues", 0.0, 1.0),
        ("mid_mass", "Mid Mass (8-15)", "Purples", 0.0, 1.0),
        ("late_mass", "Late Mass (16-23)", "Oranges", 0.0, 1.0),
    ]

    fig_width = max(10.0, 1.25 * len(group_order) + 3.0)
    fig_height = max(18.0, 3.2 * len(metric_specs))
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(fig_width, fig_height), squeeze=False)

    for panel_idx, (metric, panel_title, cmap, vmin, vmax) in enumerate(metric_specs):
        ax = axes.flat[panel_idx]
        pivot = aggregated_df.pivot(
            index=group_field,
            columns="model",
            values=metric,
        ).reindex(index=group_order, columns=model_order).T

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            mask=pivot.isna(),
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"shrink": 0.82},
        )
        ax.set_title(panel_title, fontsize=14)
        ax.set_xlabel("Benchmark" if group_field == "benchmark" else "Task Group", fontsize=12)
        ax.set_ylabel("Checkpoint", fontsize=12)
        ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
        ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
        ax.set_xticklabels(
            [group_display_map[group] for group in pivot.columns],
            rotation=0,
            fontsize=11,
        )
        ax.set_yticklabels(
            [wrap_label(model_display_map[model], width=24) for model in pivot.index],
            rotation=0,
            fontsize=9,
        )
        ax.collections[0].colorbar.ax.tick_params(labelsize=10)

    level_name = "Benchmark" if group_field == "benchmark" else "Task Group"
    fig.suptitle(
        f"Router Selection Summary Metrics by {level_name}",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.4)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filter = set(args.models) if args.models else None
    benchmark_filter = (
        {canonicalize_benchmark(name) for name in args.benchmarks}
        if args.benchmarks
        else None
    )
    checkpoint_name_map = load_checkpoint_name_map(
        Path(args.checkpoint_name_map).expanduser().resolve()
    )

    example_df, num_layers = load_example_records(
        eval_root=eval_root,
        checkpoint_name_map=checkpoint_name_map,
        model_filter=model_filter,
        benchmark_filter=benchmark_filter,
    )

    layer_cols = sorted(
        [column for column in example_df.columns if column.startswith("layer_")]
    )
    benchmark_agg = aggregate_by_group(example_df, "benchmark", layer_cols)
    task_group_agg = aggregate_by_group(example_df, "task_group", layer_cols)

    benchmark_order = ordered_values(
        benchmark_agg["benchmark"].tolist(),
        BENCHMARK_ORDER,
    )
    task_group_order = ordered_values(
        task_group_agg["task_group"].tolist(),
        TASK_GROUP_ORDER,
    )
    model_order = build_model_order(example_df, checkpoint_name_map)

    model_display_map = (
        example_df[["model", "display_model"]]
        .drop_duplicates()
        .set_index("model")["display_model"]
        .to_dict()
    )
    benchmark_display_map = (
        benchmark_agg[["benchmark", "benchmark_display"]]
        .drop_duplicates()
        .set_index("benchmark")["benchmark_display"]
        .to_dict()
    )
    task_group_display_map = (
        task_group_agg[["task_group", "task_group_display"]]
        .drop_duplicates()
        .set_index("task_group")["task_group_display"]
        .to_dict()
    )

    sns.set_theme(style="whitegrid", context="talk")

    plot_layer_usage_heatmaps(
        aggregated_df=benchmark_agg,
        group_field="benchmark",
        group_order=benchmark_order,
        group_display_map=benchmark_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        layer_cols=layer_cols,
        output_path=output_dir / "router_layer_usage_by_benchmark.png",
        max_panels_per_row=args.max_model_panels_per_row,
        dpi=args.dpi,
    )
    plot_layer_usage_heatmaps(
        aggregated_df=task_group_agg,
        group_field="task_group",
        group_order=task_group_order,
        group_display_map=task_group_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        layer_cols=layer_cols,
        output_path=output_dir / "router_layer_usage_by_task_group.png",
        max_panels_per_row=args.max_model_panels_per_row,
        dpi=args.dpi,
    )

    plot_single_metric_heatmap(
        aggregated_df=benchmark_agg,
        group_field="benchmark",
        group_order=benchmark_order,
        group_display_map=benchmark_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        value_col="reserve_mean",
        title="Reserved-Layer Mixing Weight by Benchmark",
        colorbar_label="Mean reserved-layer mixing weight",
        output_path=output_dir / "router_reserve_usage_by_benchmark.png",
        dpi=args.dpi,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        transpose=True,
    )
    plot_single_metric_heatmap(
        aggregated_df=task_group_agg,
        group_field="task_group",
        group_order=task_group_order,
        group_display_map=task_group_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        value_col="reserve_mean",
        title="Reserved-Layer Mixing Weight by Task Group",
        colorbar_label="Mean reserved-layer mixing weight",
        output_path=output_dir / "router_reserve_usage_by_task_group.png",
        dpi=args.dpi,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        transpose=True,
    )

    plot_stats_panel(
        aggregated_df=benchmark_agg,
        group_field="benchmark",
        group_order=benchmark_order,
        group_display_map=benchmark_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        output_path=output_dir / "router_selection_stats_by_benchmark.png",
        dpi=args.dpi,
        num_layers=num_layers,
    )
    plot_stats_panel(
        aggregated_df=task_group_agg,
        group_field="task_group",
        group_order=task_group_order,
        group_display_map=task_group_display_map,
        model_order=model_order,
        model_display_map=model_display_map,
        output_path=output_dir / "router_selection_stats_by_task_group.png",
        dpi=args.dpi,
        num_layers=num_layers,
    )

    print("Discovered router-enabled examples:", len(example_df))
    print("Benchmarks:", ", ".join(benchmark_display_map[name] for name in benchmark_order))
    print("Task groups:", ", ".join(task_group_display_map[name] for name in task_group_order))
    print("Models:")
    for model in model_order:
        print(f"  - {model_display_map[model]} (raw={model})")
    print("Saved plots:")
    for filename in [
        "router_layer_usage_by_benchmark.png",
        "router_layer_usage_by_task_group.png",
        "router_reserve_usage_by_benchmark.png",
        "router_reserve_usage_by_task_group.png",
        "router_selection_stats_by_benchmark.png",
        "router_selection_stats_by_task_group.png",
    ]:
        print(f"  - {output_dir / filename}")


if __name__ == "__main__":
    main()
