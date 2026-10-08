import argparse
import json
import os
import re
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


DESCRIPTIVE_PROMPTS = {
    1: "{}What is the title of the plot?\nAnswer with the exact title if present. If the plot does not have an explicit title, answer 'Not Applicable'.",
    2: "{}What is the label of the x-axis?\nAnswer with the exact x-axis label if present. If the plot does not have an explicit x-axis label, answer 'Not Applicable'.",
    3: "{}What is the label of the y-axis?\nAnswer with the exact y-axis label if present. If the plot does not have an explicit y-axis label, answer 'Not Applicable'.",
    4: "{}What is the leftmost labeled tick on the x-axis?\nAnswer with the exact tick value only.",
    5: "{}What is the rightmost labeled tick on the x-axis?\nAnswer with the exact tick value only.",
    6: "{}What is the spatially lowest labeled tick on the y-axis?\nAnswer with the exact tick value only.",
    7: "{}What is the spatially highest labeled tick on the y-axis?\nAnswer with the exact tick value only.",
    8: "{}What is the difference between consecutive numerical tick values on the x-axis?\nIf the axis is missing, non-numerical, or has non-constant spacing, answer 'Not Applicable'.",
    9: "{}What is the difference between consecutive numerical tick values on the y-axis?\nIf the axis is missing, non-numerical, or has non-constant spacing, answer 'Not Applicable'.",
    10: "{}How many lines are there?\nIgnore grid lines, tick marks, and auxiliary lines. If this is not a line plot, answer 'Not Applicable'.",
    11: "{}Do any lines intersect?\nAnswer with Yes or No. Ignore grid lines, tick marks, and auxiliary lines. If this is not a line plot, answer 'Not Applicable'.",
    12: "{}How many discrete labels are there in the legend?\nIf the plot does not have a relevant legend, answer 'Not Applicable'.",
    13: "{}What are the names of the labels in the legend?\nList them from top to bottom, then left to right, separated by commas. If the plot does not have a relevant legend, answer 'Not Applicable'.",
    14: "{}What is the difference between the maximum and minimum tick labels on the continuous legend (colorbar)?\nRemove the percent sign if present. If there is no relevant colorbar, answer 'Not Applicable'.",
    15: "{}What is the maximum tick label on the continuous legend (colorbar)?\nRemove the percent sign if present. If there is no relevant colorbar, answer 'Not Applicable'.",
    16: "{}What is the general trend of the data from left to right?\nAnswer in a short phrase such as 'increases' or 'increases then stabilizes'.",
    17: "{}What is the total number of explicitly labeled ticks across all axes?\nAnswer with a single number only.",
    18: "{}What is the layout of the subplots?\nAnswer in 'n by m' format. If there are no subplots, answer '1 by 1'.",
    19: "{}What is the number of subplots?\nAnswer with a single number only. If there are no subplots, answer '1'.",
}

REASONING_INSTRUCTIONS = {
    1: "Answer with the most relevant text label or term grounded in the chart. If multiple terms are required, separate them with commas.",
    2: "Answer with a short phrase only. If the question presents options, answer with one of the provided options.",
    3: "Answer with a single number only. Scientific notation is allowed.",
    4: "Answer with a single number only.",
}

IMAGE_KEYS = ("image", "figure_image", "figure", "Image")
PATH_KEYS = ("figure_path", "original_figure_path", "image_path", "path")
COMMON_SUFFIXES = ("", ".png", ".jpg", ".jpeg", ".webp")


def load_charxiv_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local CharXiv snapshot."
            )

        rows = []
        for path in parquet_files:
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    if args.hf_dataset:
        return hf_datasets.load_dataset(
            args.hf_dataset,
            split=args.split,
            revision=args.hf_revision,
            token=args.hf_token,
            cache_dir=cache_dir,
        )

    raise ValueError("Please provide either --hf-dataset or --parquet-dir.")


def to_pil_image(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes))
        if image_path:
            return Image.open(image_path)
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def normalize_space(value):
    return " ".join(str(value).strip().split())


def load_image_from_paths(sample, figure_id, benchmark_dir: Path, images_dir: Path | None, parquet_dir: Path | None):
    search_roots = []
    if images_dir is not None:
        search_roots.append(images_dir)
    search_roots.append(benchmark_dir / "images")
    if parquet_dir is not None:
        search_roots.append(parquet_dir)

    for key in PATH_KEYS:
        raw_path = sample.get(key)
        if not raw_path:
            continue

        raw_path = str(raw_path).strip()
        candidates = [Path(raw_path)]
        if not Path(raw_path).suffix:
            candidates.extend(Path(raw_path + suffix) for suffix in COMMON_SUFFIXES[1:])

        for candidate in candidates:
            if candidate.is_file():
                return Image.open(candidate)
            for root in search_roots:
                rooted = root / candidate
                if rooted.is_file():
                    return Image.open(rooted)

    for root in search_roots:
        for suffix in COMMON_SUFFIXES[1:]:
            candidate = root / f"{figure_id}{suffix}"
            if candidate.is_file():
                return Image.open(candidate)

    raise FileNotFoundError(
        "Could not locate a CharXiv figure image. "
        "If the Hugging Face dataset snapshot does not embed images, provide --images-dir."
    )


def extract_image(sample, figure_id, benchmark_dir: Path, images_dir: Path | None, parquet_dir: Path | None):
    for key in IMAGE_KEYS:
        value = sample.get(key)
        if value is not None:
            return to_pil_image(value)
    return load_image_from_paths(sample, figure_id, benchmark_dir, images_dir, parquet_dir)


def get_figure_id(sample, fallback_index):
    for key in ("figure_id", "original_id", "id", "index"):
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(fallback_index)


def format_subplot_prefix(sample):
    num_subplots = sample.get("num_subplots")
    subplot_loc = sample.get("subplot_loc")

    try:
        num_subplots = int(num_subplots)
    except (TypeError, ValueError):
        num_subplots = None

    if num_subplots is not None and num_subplots <= 1:
        return ""

    if isinstance(subplot_loc, str):
        location = normalize_space(subplot_loc)
        if location:
            return f"Focus on {location}. "

    if isinstance(subplot_loc, (list, tuple)) and len(subplot_loc) == 2:
        row, col = subplot_loc
        return f"Focus on the subplot in row {row}, column {col}. "

    subplot_row = sample.get("subplot_row")
    subplot_col = sample.get("subplot_col")
    if subplot_row is not None and subplot_col is not None:
        return f"Focus on the subplot in row {subplot_row}, column {subplot_col}. "

    return ""


def descriptive_entries(sample):
    qids = sample.get("qids")
    answers = sample.get("answers")
    if isinstance(qids, list) and isinstance(answers, list):
        for idx, (qid, answer) in enumerate(zip(qids, answers), start=1):
            yield idx, qid, answer
        return

    for idx in range(1, 5):
        qid = sample.get(f"descriptive_q{idx}")
        answer = sample.get(f"descriptive_a{idx}")
        if qid is None and answer is None:
            continue
        yield idx, qid, answer


def reasoning_entry(sample):
    query = sample.get("reasoning_q", sample.get("query"))
    answer = sample.get("reasoning_a", sample.get("answer"))
    inst_category = sample.get("reasoning_inst_category", sample.get("inst_category"))
    qa_source = sample.get("reasoning_qa_source", sample.get("qa_source"))
    return query, answer, inst_category, qa_source


def descriptive_prompt(qid, subplot_prefix):
    qid = int(qid)
    template = DESCRIPTIVE_PROMPTS.get(qid)
    if template is None:
        raise KeyError(f"Unsupported descriptive question id: {qid}")
    return template.format(subplot_prefix)


def reasoning_prompt(query, inst_category):
    question = normalize_space(query)
    try:
        inst_key = int(inst_category)
    except (TypeError, ValueError):
        inst_key = 0
    inst_text = REASONING_INSTRUCTIONS.get(inst_key, "Answer as briefly as possible.")
    return f"{question}\n{inst_text}"


def main():
    parser = argparse.ArgumentParser(description="Convert CharXiv to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/CharXiv",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="princeton-nlp/CharXiv",
        help="Optional Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="main",
        help="Revision for --hf-dataset.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private/gated datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir when using --hf-dataset.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=None,
        help="Optional local directory containing CharXiv parquet files.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional local directory containing figure image files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    parquet_dir = None if not args.parquet_dir else Path(args.parquet_dir).expanduser().resolve()
    images_dir = None if not args.images_dir else Path(args.images_dir).expanduser().resolve()

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_charxiv_dataset(args, benchmark_dir=benchmark_dir)
    seen_figure_ids = set()
    num_samples = 0

    with open(output_file, "w") as f:
        for row_index, sample in enumerate(dataset):
            figure_id = get_figure_id(sample, row_index)
            subplot_prefix = format_subplot_prefix(sample)

            rel_image_path = Path("images") / f"{figure_id}.png"
            abs_image_path = benchmark_dir / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)

            if figure_id not in seen_figure_ids or not abs_image_path.exists():
                image = extract_image(sample, figure_id, benchmark_dir, images_dir, parquet_dir)
                image.save(abs_image_path, format="PNG")
                seen_figure_ids.add(figure_id)

            for slot_idx, qid, answer in descriptive_entries(sample):
                if qid is None:
                    continue
                qid = int(qid)
                row = {
                    "question_id": f"{figure_id}_descriptive_{slot_idx}",
                    "figure_id": str(figure_id),
                    "image": rel_image_path.as_posix(),
                    "text": descriptive_prompt(qid, subplot_prefix),
                    "label": "" if answer is None else str(answer).strip(),
                    "mode": "descriptive",
                    "descriptive_qid": qid,
                    "instruction_category": None,
                    "question_category": "descriptive",
                    "source_split": args.split,
                    "subplot_loc": sample.get("subplot_loc"),
                    "num_subplots": sample.get("num_subplots"),
                }
                f.write(json.dumps(row) + "\n")
                num_samples += 1

            query, answer, inst_category, qa_source = reasoning_entry(sample)
            if query is not None:
                row = {
                    "question_id": f"{figure_id}_reasoning",
                    "figure_id": str(figure_id),
                    "image": rel_image_path.as_posix(),
                    "text": reasoning_prompt(query, inst_category),
                    "label": "" if answer is None else str(answer).strip(),
                    "mode": "reasoning",
                    "descriptive_qid": None,
                    "instruction_category": None if inst_category is None else int(inst_category),
                    "question_category": "reasoning",
                    "qa_source": qa_source,
                    "source_split": args.split,
                    "raw_question": str(query).strip(),
                }
                f.write(json.dumps(row) + "\n")
                num_samples += 1

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images'}")


if __name__ == "__main__":
    main()
