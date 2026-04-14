import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


"""
Run this first to obtain summaries of benchmarks:

$ bash scripts/v1_5/eval/summarize_benchmark_outputs.sh mmbench

How to run:

$ python scripts/v1_5/eval/plot_benchmark_results.py \
  --eval-root playground/data/eval \
  --benchmarks mmbench gqa seed pope textvqa docvqa mme mme_realworld vizwiz vqav2 hallusionbench realworldqa chartqa hrbench4k hrbench8k mmstar mmvet mmvp \
  --output-dir playground/data/eval/plots \
  --include-overall \
  --max-categories-per-row 6 \
  --gqa-max-categories-per-row 5
"""

DEFAULT_CHECKPOINT_NAME_MAP = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "llava-v1.5-13b-moe-finetune_display_names.json"
)
MODEL_LABEL_OVERRIDES = None


SUPPORTED_BENCHMARKS = [
    "mmbench",
    "mmmu",
    "mmstar",
    "mmvet",
    "mmvp",
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

DEFAULT_VSTAR_CATEGORIES = ["GPT4V-hard", "OCR", "direct_attributes", "relative_position"]
DEFAULT_DTD_CATEGORIES = [f"test{i}" for i in range(1, 11)]
DEFAULT_HRBENCH_CATEGORIES = ["single", "cross"]
DEFAULT_HRBENCH_CYCLE_CATEGORIES = ["0", "1", "2", "3"]
DEFAULT_MMMU_DOMAINS = [
    "Art and Design",
    "Business",
    "Health and Medicine",
    "Humanities and Social Science",
    "Science",
    "Tech and Engineering",
]
DEFAULT_MMMU_SUBJECTS = [
    "Art",
    "Art_Theory",
    "Design",
    "Music",
    "Accounting",
    "Economics",
    "Finance",
    "Manage",
    "Marketing",
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
    "History",
    "Literature",
    "Psychology",
    "Sociology",
    "Biology",
    "Chemistry",
    "Geography",
    "Math",
    "Physics",
    "Agriculture",
    "Architecture_and_Engineering",
    "Computer_Science",
    "Electronics",
    "Energy_and_Power",
    "Materials",
    "Mechanical_Engineering",
]
DEFAULT_MMMU_QUESTION_TYPES = ["multiple_choice", "open"]

BENCHMARK_DISPLAY_NAMES = {
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
    "pope": "POPE",
    "realworldqa": "RealWorldQA",
    "seed": "SEED",
    "textvqa": "TextVQA",
    "vizwiz": "VizWiz",
    "vqav2": "VQAv2",
    "vstar": "V*Bench",
}

BENCHMARK_CATEGORY_ORDER = {
    ("dtd", "by_split"): DEFAULT_DTD_CATEGORIES,
    ("hrbench4k", "by_category"): DEFAULT_HRBENCH_CATEGORIES,
    ("hrbench4k", "by_cycle_category"): DEFAULT_HRBENCH_CYCLE_CATEGORIES,
    ("hrbench8k", "by_category"): DEFAULT_HRBENCH_CATEGORIES,
    ("hrbench8k", "by_cycle_category"): DEFAULT_HRBENCH_CYCLE_CATEGORIES,
    ("mmmu", "by_domain"): DEFAULT_MMMU_DOMAINS,
    ("mmmu", "by_question_type"): DEFAULT_MMMU_QUESTION_TYPES,
    ("mmmu", "by_subject"): DEFAULT_MMMU_SUBJECTS,
    ("vstar", "by_category"): DEFAULT_VSTAR_CATEGORIES,
}

METRIC_KEY_ORDER = [
    "leaderboard",
    "question_accuracy",
    "figure_accuracy",
    "question_pair_accuracy",
    "error_breakdown",
    "by_capability",
    "by_capability_combo",
    "by_category",
    "by_subcategory",
    "by_cycle_category",
    "by_domain",
    "by_subject",
    "by_structural_type",
    "by_semantic_type",
    "by_question_type",
    "by_data_type",
    "by_type",
    "by_answer_type",
    "by_split",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-benchmark grouped bar charts from summary.json files."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="playground/data/eval",
        help="Root folder to scan recursively for summary files.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=SUPPORTED_BENCHMARKS,
        help="Benchmarks to plot. Use any subset of: " + ", ".join(SUPPORTED_BENCHMARKS),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground/data/eval/plots",
        help="Where plot images will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional raw model folder names to include.",
    )
    parser.add_argument(
        "--checkpoint-name-map",
        type=str,
        default=str(DEFAULT_CHECKPOINT_NAME_MAP),
        help="Optional JSON map from checkpoint path or basename to display label.",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="Also generate an overall-accuracy chart per benchmark.",
    )
    parser.add_argument(
        "--max-categories-per-row",
        type=int,
        default=6,
        help="Wrap category groups into rows when category count exceeds this value.",
    )
    parser.add_argument(
        "--gqa-max-categories-per-row",
        type=int,
        default=5,
        help="GQA-specific cap for category groups per row.",
    )
    parser.add_argument(
        "--dtd-max-categories-per-row",
        type=int,
        default=5,
        help="DTD-specific cap for category groups per row.",
    )
    return parser.parse_args()


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def multiline_text_size(draw, text, font, spacing=2):
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return right - left, bottom - top


def infer_model_name(summary_path: Path):
    name = summary_path.name
    if name.endswith("_summary.json"):
        return name[: -len("_summary.json")]
    if name == "summary.json":
        return summary_path.parent.name
    return summary_path.stem


def load_model_label_overrides(path: Path):
    if path is None:
        return {}

    label_path = Path(path).expanduser()
    if not label_path.exists():
        return {}

    with open(label_path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint label map must be a JSON object: {label_path}")

    overrides = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        label = value.strip()
        if not label:
            continue
        overrides[key] = label
        overrides[Path(key).name] = label
    return overrides


def set_model_label_overrides(path=None):
    global MODEL_LABEL_OVERRIDES
    target_path = DEFAULT_CHECKPOINT_NAME_MAP if path is None else Path(path)
    MODEL_LABEL_OVERRIDES = load_model_label_overrides(target_path)


def get_model_label_overrides():
    global MODEL_LABEL_OVERRIDES
    if MODEL_LABEL_OVERRIDES is None:
        set_model_label_overrides()
    return MODEL_LABEL_OVERRIDES


def extract_lr_schedule(model_name: str):
    match = re.search(r"(?:router-)?lr(.+?)-pretrain_lr(.+?)-finetune", model_name)
    if not match:
        return None
    return match.group(1), match.group(2)


def extract_softmix_lr_schedule(model_name: str):
    return extract_lr_schedule(model_name)


def extract_k_value(model_name: str):
    match = re.search(r"_k(\d+)(?:_|$)", model_name)
    if not match:
        return None
    return match.group(1)


def descriptive_moe_label(model_name: str):
    if "text-cond-attn-patchlayer-softmix-router" in model_name:
        lr_schedule = extract_lr_schedule(model_name)
        if lr_schedule:
            return f"Text-Cond SoftMix (pre {lr_schedule[0]}, ft {lr_schedule[1]})"
        return "Text-Cond SoftMix"

    if "text-cond-attn-patchlayer-router_pretrain" in model_name:
        return "Text-Cond PatchLayer Pretrain"

    if "text-cond-attn-patchlayer-router" in model_name:
        parts = []
        lr_schedule = extract_lr_schedule(model_name)
        if lr_schedule:
            parts.append(f"pre {lr_schedule[0]}")
            parts.append(f"ft {lr_schedule[1]}")
        k_value = extract_k_value(model_name)
        if k_value:
            parts.append(f"k={k_value}")
        if parts:
            return f"Text-Cond PatchLayer ({', '.join(parts)})"
        return "Text-Cond PatchLayer"

    if "text-cond-attn-layer-router" in model_name:
        return "Text-Cond Attn-Layer"

    if "text-cond-attn-token-router" in model_name:
        return "Text-Cond Attn-Token"

    if "text-cond-film-router_confidence-gate" in model_name:
        return "Text-Cond FiLM (confidence gate)"

    if "text-cond-film-router_learnable-gate" in model_name:
        return "Text-Cond FiLM"

    if "router_learnable-gate" in model_name:
        return "Image-Cond Router"

    return model_name


def resolve_model_label(model_name: str, default_label: str):
    label_override = get_model_label_overrides().get(model_name)
    if label_override:
        return label_override
    if model_name.startswith("llava-v1.5-13b-moe-finetune"):
        return descriptive_moe_label(model_name)
    return default_label


def default_model_metadata(model_name: str):
    if model_name == "llava-v1.5-13b":
        return {
            "label": "LLaVA-v1.5-13B",
            "sort_key": (0, model_name),
            "style": {
                "fill": (110, 131, 156),
                "outline": (78, 98, 122),
                "pattern": "none",
            },
        }

    if "text-cond-attn-patchlayer-softmix-router" in model_name:
        lr_schedule = extract_softmix_lr_schedule(model_name)
        label = "Text-Cond SoftMix"
        sort_rank = 60
        fill = (118, 148, 142)
        outline = (84, 112, 106)
        if lr_schedule == ("1e-5", "1e-5"):
            label = "Text-Cond SoftMix (pre 1e-5, ft 1e-5)"
            sort_rank = 61
            fill = (99, 132, 126)
            outline = (70, 101, 96)
        elif lr_schedule == ("2e-5", "5e-6"):
            label = "Text-Cond SoftMix (pre 2e-5, ft 5e-6)"
            sort_rank = 62
            fill = (142, 162, 111)
            outline = (105, 122, 77)
        return {
            "label": label,
            "sort_key": (sort_rank, model_name),
            "style": {
                "fill": fill,
                "outline": outline,
                "pattern": "softmix",
            },
        }

    if "text-cond-attn-patchlayer" in model_name:
        return {
            "label": "Text-Cond PatchLayer",
            "sort_key": (59, model_name),
            "style": {
                "fill": (127, 150, 153),
                "outline": (92, 113, 117),
                "pattern": "attn",
            },
        }

    if "text-cond-attn-layer-router" in model_name:
        return {
            "label": "Text-Cond Attn-Layer",
            "sort_key": (20, model_name),
            "style": {
                "fill": (146, 133, 163),
                "outline": (111, 98, 127),
                "pattern": "attn",
            },
        }

    if "text-cond-attn-token-router" in model_name:
        return {
            "label": "Text-Cond Attn-Token",
            "sort_key": (30, model_name),
            "style": {
                "fill": (124, 136, 173),
                "outline": (91, 103, 139),
                "pattern": "attn",
            },
        }

    if "text-cond-film-router_confidence-gate" in model_name:
        return {
            "label": "Text-Cond FiLM (confidence gate)",
            "sort_key": (40, model_name),
            "style": {
                "fill": (170, 145, 156),
                "outline": (132, 109, 120),
                "pattern": "film",
            },
        }

    if "text-cond-film-router_learnable-gate" in model_name:
        return {
            "label": "Text-Cond FiLM (learnable gate)",
            "sort_key": (50, model_name),
            "style": {
                "fill": (154, 145, 166),
                "outline": (117, 108, 128),
                "pattern": "film",
            },
        }

    if "text-cond-attn" in model_name:
        return {
            "label": "Text-Cond Attention",
            "sort_key": (70, model_name),
            "style": {
                "fill": (137, 138, 170),
                "outline": (103, 104, 133),
                "pattern": "attn",
            },
        }

    if "text-cond-film" in model_name:
        return {
            "label": "Text-Cond FiLM",
            "sort_key": (80, model_name),
            "style": {
                "fill": (158, 148, 165),
                "outline": (120, 110, 126),
                "pattern": "film",
            },
        }

    if "text-cond" in model_name:
        return {
            "label": "Text-Cond Router",
            "sort_key": (90, model_name),
            "style": {
                "fill": (150, 140, 164),
                "outline": (114, 104, 126),
                "pattern": "text-cond",
            },
        }

    if "router_learnable-gate" in model_name:
        return {
            "label": "Image-Cond Router (learnable gate)",
            "sort_key": (10, model_name),
            "style": {
                "fill": (160, 131, 129),
                "outline": (123, 97, 95),
                "pattern": "none",
            },
        }

    return {
        "label": model_name,
        "sort_key": (100, model_name),
        "style": {
            "fill": (151, 151, 151),
            "outline": (112, 112, 112),
            "pattern": "other",
        },
    }


def model_metadata(model_name: str):
    metadata = default_model_metadata(model_name).copy()
    metadata["label"] = resolve_model_label(model_name, metadata["label"])
    return metadata


def display_model_name(model_name: str):
    return model_metadata(model_name)["label"]


def normalize_benchmark_name(name):
    if not isinstance(name, str):
        return None
    normalized = name.strip().lower()
    aliases = {
        "doc vqa": "docvqa",
        "doc-vqa": "docvqa",
        "doc_vqa": "docvqa",
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
        "v*bench": "vstar",
        "vstar_bench": "vstar",
    }
    return aliases.get(normalized, normalized)


def infer_dataset(summary, summary_path: Path):
    dataset = normalize_benchmark_name(summary.get("dataset"))
    if dataset:
        return dataset

    path_txt = summary_path.as_posix().lower()
    if "/hrbench4k/" in path_txt:
        return "hrbench4k"
    if "/hrbench8k/" in path_txt:
        return "hrbench8k"
    if "/mmstar/" in path_txt:
        return "mmstar"
    if "/mmvet/" in path_txt or "/mm-vet/" in path_txt:
        return "mmvet"
    if "/mmvp/" in path_txt:
        return "mmvp"
    if "/mme-realworld/" in path_txt or "/mme_realworld/" in path_txt:
        return "mme_realworld"
    if "/docvqa/" in path_txt:
        return "docvqa"
    if "/vstar_bench/" in path_txt and "by_category" in summary:
        return "vstar"
    if "/dtd/" in path_txt and ("by_split" in summary or "by_class" in summary):
        return "dtd"
    return None


def benchmark_display_name(benchmark: str):
    normalized = normalize_benchmark_name(benchmark)
    if normalized in BENCHMARK_DISPLAY_NAMES:
        return BENCHMARK_DISPLAY_NAMES[normalized]
    return normalized.upper()


def metric_display_name(metric_key):
    labels = {
        "by_answer_type": "Answer Type Accuracy",
        "by_capability": "Capability Score",
        "by_capability_combo": "Capability Combo Score",
        "by_category": "Category Accuracy",
        "by_cycle_category": "Cycle Accuracy",
        "by_data_type": "Data Type Accuracy",
        "by_domain": "Domain Accuracy",
        "by_question_type": "Question Type Accuracy",
        "by_semantic_type": "Semantic Type Accuracy",
        "by_subcategory": "Subcategory Accuracy",
        "by_subject": "Subject Accuracy",
        "by_split": "Split Accuracy",
        "by_structural_type": "Structural Type Accuracy",
        "by_type": "Type Accuracy",
        "error_breakdown": "Error Breakdown",
        "figure_accuracy": "Figure Accuracy",
        "leaderboard": "Leaderboard Accuracy",
        "overall": "Overall Accuracy",
        "question_accuracy": "Question Accuracy",
        "question_pair_accuracy": "Question Pair Accuracy",
    }
    if metric_key in labels:
        return labels[metric_key]
    return metric_key.replace("by_", "").replace("_", " ").title()


def benchmark_category_order(benchmark: str, metric_key: str):
    return BENCHMARK_CATEGORY_ORDER.get((normalize_benchmark_name(benchmark), metric_key), [])


def benchmark_row_priority(benchmark: str, summary_path: Path):
    benchmark = normalize_benchmark_name(benchmark)
    path_txt = summary_path.as_posix().lower()
    if benchmark == "dtd":
        if "/answers/test_1-10/" in path_txt:
            return 100
        if re.search(r"/answers/test_\d+/", path_txt):
            return 10
    return 0


def classify_model(model_name: str):
    metadata = model_metadata(model_name)
    style = metadata["style"]
    return style["fill"], style["outline"], style["pattern"]


def style_for_model(model_name: str):
    fill, outline, pattern = classify_model(model_name)
    return {"fill": fill, "outline": outline, "pattern": pattern}


def draw_pattern(draw, rect, pattern, color):
    xl, yt, xr, yb = rect
    if pattern == "none":
        return

    if pattern == "attn":
        # Dotted pattern for text-cond-attn.
        spacing = 7
        for x in range(xl + 3, xr - 1, spacing):
            for y in range(yt + 3, yb - 1, spacing):
                draw.rectangle((x, y, x + 1, y + 1), fill=color)
        return

    if pattern == "film":
        # Diagonal hatch for text-cond-film.
        spacing = 8
        diag_len = 11
        for x in range(xl + 1, xr - 1, spacing):
            x2 = min(x + diag_len, xr - 1)
            y2 = max(yb - diag_len, yt + 1)
            draw.line((x, yb - 1, x2, y2), fill=color, width=1)
        for y in range(yb - 1, yt + 1, -spacing):
            x2 = min(xl + 1 + diag_len, xr - 1)
            y2 = max(y - diag_len, yt + 1)
            draw.line((xl + 1, y, x2, y2), fill=color, width=1)
        return

    if pattern == "softmix":
        spacing = 7
        for x in range(xl + 3, xr - 1, spacing):
            draw.line((x, yt + 1, x, yb - 1), fill=color, width=1)
        return

    # Fallback sparse dots.
    spacing = 7
    for x in range(xl + 2, xr, spacing):
        for y in range(yt + 2, yb, spacing):
            draw.point((x, y), fill=color)


def format_category_label(label: str):
    txt = str(label).replace("_", " ")
    if len(txt) <= 16:
        return txt
    words = txt.split()
    if len(words) < 2:
        return txt
    half = len(words) // 2
    return " ".join(words[:half]) + "\n" + " ".join(words[half:])


def chunk_list(items, chunk_size):
    out = []
    for i in range(0, len(items), chunk_size):
        out.append(items[i : i + chunk_size])
    return out


def summary_paths(search_root: Path, recursive=True):
    if recursive:
        found = list(search_root.rglob("summary.json")) + list(search_root.rglob("*_summary.json"))
    else:
        found = list(search_root.glob("*/summary.json")) + list(search_root.glob("*_summary.json"))

    seen = set()
    unique = []
    for path in sorted(found):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def build_summary_row(summary_path: Path, summary, dataset_override=None):
    dataset = normalize_benchmark_name(dataset_override) or infer_dataset(summary, summary_path)
    if not dataset:
        return None

    model_name = infer_model_name(summary_path)
    metadata = model_metadata(model_name)
    return {
        "dataset": dataset,
        "dataset_priority": benchmark_row_priority(dataset, summary_path),
        "model": model_name,
        "display_model": metadata["label"],
        "sort_key": metadata["sort_key"],
        "summary_path": summary_path,
        "summary": summary,
    }


def discover_summary_rows(summary_paths_to_load, dataset_override=None, model_names=None):
    rows = []
    allowed_models = set(model_names) if model_names else None
    for path in summary_paths_to_load:
        try:
            with open(path, "r") as f:
                summary = json.load(f)
        except Exception:
            continue

        row = build_summary_row(path, summary, dataset_override=dataset_override)
        if row is None:
            continue
        if allowed_models and row["model"] not in allowed_models:
            continue
        rows.append(row)
    return rows


def discover_all_summaries(eval_root: Path):
    return discover_summary_rows(summary_paths(eval_root, recursive=True))


def discover_benchmark_summaries(answers_root: Path, benchmark: str, model_names=None):
    return discover_summary_rows(
        summary_paths(answers_root, recursive=False),
        dataset_override=benchmark,
        model_names=model_names,
    )


def accuracy_bucket(summary, key):
    bucket = summary.get(key, {})
    if not isinstance(bucket, dict):
        return {}

    def extract_metric(node, prefix=""):
        out = {}
        if not isinstance(node, dict):
            return out

        acc = node.get("accuracy")
        if isinstance(acc, (int, float)):
            if prefix:
                out[prefix] = float(acc)
            return out

        correct = node.get("correct")
        total = node.get("total")
        if isinstance(correct, (int, float)) and isinstance(total, (int, float)) and total > 0:
            if prefix:
                out[prefix] = float(correct) / float(total)
            return out

        for child_key, child_value in node.items():
            if isinstance(child_value, (int, float)):
                if prefix:
                    continue
                out[str(child_key)] = float(child_value)
            elif isinstance(child_value, dict):
                label = str(child_key) if not prefix else f"{prefix} / {child_key}"
                out.update(extract_metric(child_value, label))
        return out

    return extract_metric(bucket)


def draw_metric_plot(
    output_path: Path,
    benchmark: str,
    metric_key: str,
    categories,
    model_names,
    model_display_names,
    value_map,
    max_categories_per_row=6,
    title=None,
):
    if not categories:
        return

    n_models = len(model_names)
    category_rows = chunk_list(categories, max(1, max_categories_per_row))
    max_chunk = max(len(r) for r in category_rows)

    left_margin = 110
    right_margin = 60
    row_gap = 88
    chart_h = 480
    category_gap = 56
    bar_slot = max(28, min(64, int(180 / max(n_models, 1))))
    bar_width = int(bar_slot * 0.96)
    group_width = n_models * bar_slot
    chart_w = max(960, max_chunk * (group_width + category_gap))

    # Prepare fonts.
    font = load_font(17)
    title_font = load_font(32)
    value_font = load_font(18)
    legend_font = load_font(15)

    # Measure legend height on a temp canvas.
    temp_img = Image.new("RGB", (10, 10), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)

    legend_x = left_margin
    legend_y = 88
    cursor_x = legend_x + text_size(temp_draw, "Model:", legend_font)[0] + 14
    cursor_y = legend_y
    row_h = 24
    x_end = left_margin + chart_w
    for i, model_name in enumerate(model_names):
        label = model_display_names[i]
        label_w, _ = text_size(temp_draw, label, legend_font)
        item_w = 16 + 8 + label_w + 20
        if cursor_x + item_w > x_end:
            cursor_x = legend_x
            cursor_y += row_h
        cursor_x += item_w
    legend_bottom = cursor_y + row_h

    top_margin = legend_bottom + 36
    bottom_margin = 145
    width = left_margin + chart_w + right_margin
    height = top_margin + len(category_rows) * chart_h + max(0, len(category_rows) - 1) * row_gap + bottom_margin

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    title = title or f"{benchmark_display_name(benchmark)} - {metric_display_name(metric_key)}"
    draw.text((left_margin, 24), title, fill=(24, 30, 36), font=title_font)

    # Legend
    draw.text((legend_x, legend_y), "Model:", fill=(24, 30, 36), font=legend_font)
    cursor_x = legend_x + text_size(draw, "Model:", legend_font)[0] + 14
    cursor_y = legend_y
    for i, model_name in enumerate(model_names):
        style = style_for_model(model_name)
        model_label = model_display_names[i]
        label_w, _ = text_size(draw, model_label, legend_font)
        item_w = 16 + 8 + label_w + 20
        if cursor_x + item_w > x_end:
            cursor_x = legend_x
            cursor_y += row_h
        swatch = (cursor_x, cursor_y + 4, cursor_x + 14, cursor_y + 18)
        draw.rectangle(swatch, fill=style["fill"], outline=style["outline"], width=1)
        draw_pattern(draw, swatch, style["pattern"], style["outline"])
        draw.text((cursor_x + 22, cursor_y), model_label, fill=(24, 30, 36), font=legend_font)
        cursor_x += item_w

    # Draw each wrapped category row as its own bar panel.
    for row_idx, row_categories in enumerate(category_rows):
        y1 = top_margin + row_idx * (chart_h + row_gap)
        y0 = y1 + chart_h
        x0 = left_margin
        x1 = left_margin + chart_w

        # Axes
        draw.line((x0, y1, x0, y0), fill=(40, 46, 52), width=2)
        draw.line((x0, y0, x1, y0), fill=(40, 46, 52), width=2)

        # Y grid and labels.
        for tick in range(0, 101, 10):
            y = y0 - int(chart_h * (tick / 100.0))
            grid_color = (224, 228, 232) if tick % 20 == 0 else (236, 239, 241)
            draw.line((x0, y, x1, y), fill=grid_color, width=1)
            label = str(tick)
            tw, th = text_size(draw, label, font)
            draw.text((x0 - tw - 10, y - th // 2), label, fill=(52, 58, 64), font=font)

        # Bars.
        for j, category in enumerate(row_categories):
            group_left = x0 + j * (group_width + category_gap)
            for i, model_name in enumerate(model_names):
                v = value_map.get(model_name, {}).get(category)
                if v is None:
                    continue
                v_pct = max(0.0, min(100.0, float(v) * 100.0))
                h = int(chart_h * v_pct / 100.0)
                xl = group_left + i * bar_slot
                xr = xl + bar_width
                yt = y0 - h
                style = style_for_model(model_name)
                rect = (xl, yt, xr, y0)
                draw.rectangle(rect, fill=style["fill"], outline=style["outline"], width=1)
                draw_pattern(draw, rect, style["pattern"], style["outline"])
                value_txt = f"{v_pct:.1f}"
                tw, th = text_size(draw, value_txt, value_font)
                draw.text((xl + (xr - xl - tw) // 2, yt - th - 5), value_txt, fill=(36, 42, 48), font=value_font)

            # category label
            cat_txt = format_category_label(category)
            tw, th = multiline_text_size(draw, cat_txt, font, spacing=2)
            center_x = group_left + group_width // 2
            draw.multiline_text(
                (center_x - tw // 2, y0 + 12),
                cat_txt,
                fill=(24, 30, 36),
                font=font,
                spacing=2,
                align="center",
            )

        # Y-axis label on first row only.
        if row_idx == 0:
            draw.text((8, y1 - 24), "Accuracy (%)", fill=(24, 30, 36), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def ordered_metric_categories(benchmark, metric_key, discovered_categories):
    preferred = benchmark_category_order(benchmark, metric_key)
    ordered = []
    seen = set()
    for category in preferred:
        if category in discovered_categories and category not in seen:
            ordered.append(category)
            seen.add(category)

    extras = [category for category in discovered_categories if category not in seen]
    if benchmark == "dtd" and metric_key == "by_split":
        def split_sort_key(label):
            match = re.fullmatch(r"test(\d+)", str(label))
            if match:
                return (0, int(match.group(1)))
            return (1, str(label))

        extras = sorted(extras, key=split_sort_key)
    else:
        extras = sorted(extras)
    return ordered + extras


def collect_metric_categories(dataset_rows, benchmark, metric_key):
    if metric_key == "overall":
        return ["overall"]
    cats = set()
    for row in dataset_rows:
        bucket = accuracy_bucket(row["summary"], metric_key)
        cats.update(bucket.keys())
    return ordered_metric_categories(benchmark, metric_key, cats)


def values_for_metric(dataset_rows, model_names, metric_key, benchmark=""):
    categories = ["overall"] if metric_key == "overall" else collect_metric_categories(dataset_rows, benchmark, metric_key)
    return values_for_categories(dataset_rows, model_names, metric_key, categories)


def values_for_categories(dataset_rows, model_names, metric_key, categories):
    per_model = {}
    summaries_by_model = {row["model"]: row["summary"] for row in dataset_rows}
    for model in model_names:
        summary = summaries_by_model[model]
        metric_values = {} if metric_key == "overall" else accuracy_bucket(summary, metric_key)
        overall_acc = summary.get("overall", {}).get("accuracy")
        per_model[model] = {}
        for category in categories:
            if category == "overall":
                per_model[model][category] = overall_acc if isinstance(overall_acc, (int, float)) else None
            elif metric_key == "overall":
                per_model[model][category] = None
            else:
                per_model[model][category] = metric_values.get(category)
    return per_model


def deduplicate_dataset_rows(dataset_rows):
    dedup = {}
    for row in dataset_rows:
        existing = dedup.get(row["model"])
        if existing is None:
            dedup[row["model"]] = row
            continue
        if row["dataset_priority"] > existing["dataset_priority"]:
            dedup[row["model"]] = row
            continue
        if (
            row["dataset_priority"] == existing["dataset_priority"]
            and str(row["summary_path"]) < str(existing["summary_path"])
        ):
            dedup[row["model"]] = row
    return sorted(dedup.values(), key=lambda r: r["sort_key"])


def plot_metric_rows(
    dataset_rows,
    benchmark,
    metric_key,
    output_path: Path,
    categories=None,
    include_overall=False,
    max_categories_per_row=6,
    title=None,
):
    if not dataset_rows:
        return None

    dataset_rows = deduplicate_dataset_rows(sorted(dataset_rows, key=lambda r: r["sort_key"]))
    model_names = [r["model"] for r in dataset_rows]
    model_display_names = [r["display_model"] for r in dataset_rows]

    plot_categories = list(categories or benchmark_category_order(benchmark, metric_key))
    if include_overall and "overall" not in plot_categories:
        plot_categories.append("overall")

    discovered_categories = collect_metric_categories(dataset_rows, benchmark, metric_key)
    for category in discovered_categories:
        if category not in plot_categories:
            plot_categories.append(category)

    value_map = values_for_categories(dataset_rows, model_names, metric_key, plot_categories)
    draw_metric_plot(
        output_path=output_path,
        benchmark=benchmark,
        metric_key=metric_key,
        categories=plot_categories,
        model_names=model_names,
        model_display_names=model_display_names,
        value_map=value_map,
        max_categories_per_row=max_categories_per_row,
        title=title,
    )
    return output_path


def plot_benchmark(dataset_rows, benchmark, output_dir, include_overall=False, max_categories_per_row=6):
    if not dataset_rows:
        return []

    dataset_rows = deduplicate_dataset_rows(sorted(dataset_rows, key=lambda r: r["sort_key"]))

    model_names = [r["model"] for r in dataset_rows]
    model_display_names = [r["display_model"] for r in dataset_rows]

    metric_keys = []
    if include_overall:
        metric_keys.append("overall")
    for key in METRIC_KEY_ORDER:
        if any(accuracy_bucket(row["summary"], key) for row in dataset_rows):
            metric_keys.append(key)

    generated = []
    for metric_key in metric_keys:
        categories = collect_metric_categories(dataset_rows, benchmark, metric_key)
        if not categories:
            continue
        value_map = values_for_categories(dataset_rows, model_names, metric_key, categories)
        out_name = f"{benchmark}_{metric_key}.png"
        out_path = output_dir / out_name
        draw_metric_plot(
            output_path=out_path,
            benchmark=benchmark,
            metric_key=metric_key,
            categories=categories,
            model_names=model_names,
            model_display_names=model_display_names,
            value_map=value_map,
            max_categories_per_row=max_categories_per_row,
        )
        generated.append(out_path)
    return generated


def main():
    args = parse_args()
    set_model_label_overrides(args.checkpoint_name_map)
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_benchmarks = []
    seen_benchmarks = set()
    for benchmark in args.benchmarks:
        normalized = normalize_benchmark_name(benchmark)
        if normalized not in seen_benchmarks:
            requested_benchmarks.append(normalized)
            seen_benchmarks.add(normalized)

    invalid = sorted(set(requested_benchmarks) - set(SUPPORTED_BENCHMARKS))
    if invalid:
        raise ValueError(f"Unsupported benchmarks: {invalid}. Supported: {SUPPORTED_BENCHMARKS}")

    all_rows = discover_all_summaries(eval_root)
    if not all_rows:
        raise FileNotFoundError(f"No parseable summary files found under {eval_root}")

    if args.models:
        allowed = set(args.models)
        all_rows = [r for r in all_rows if r["model"] in allowed]

    total_generated = 0
    for benchmark in requested_benchmarks:
        rows = [r for r in all_rows if r["dataset"] == benchmark]
        if benchmark == "gqa":
            max_per_row = args.gqa_max_categories_per_row
        elif benchmark == "dtd":
            max_per_row = args.dtd_max_categories_per_row
        else:
            max_per_row = args.max_categories_per_row
        generated = plot_benchmark(
            dataset_rows=rows,
            benchmark=benchmark,
            output_dir=output_dir,
            include_overall=args.include_overall,
            max_categories_per_row=max(1, max_per_row),
        )
        if not generated:
            print(f"[{benchmark}] No plots generated (no matching summary data).")
            continue
        print(f"[{benchmark}] Generated {len(generated)} plot(s):")
        for p in generated:
            print(f"  - {p}")
            total_generated += 1

    print(f"Done. Total plots generated: {total_generated}")


if __name__ == "__main__":
    main()
