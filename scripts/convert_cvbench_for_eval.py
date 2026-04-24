import argparse
import json
import os
import string
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


def normalize_config_name(config_name):
    if config_name is None:
        return "default"
    return str(config_name).strip().lower() or "default"


def load_cvbench_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    normalized_config = normalize_config_name(args.config_name)

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local CV-Bench snapshot."
            )

        if normalized_config in {"2d", "3d"}:
            preferred = [path for path in parquet_files if normalized_config in path.as_posix().lower()]
            if preferred:
                parquet_files = preferred
        else:
            preferred = [
                path
                for path in parquet_files
                if path.name.lower() == f"{args.split}.parquet" or "/default/" in path.as_posix().lower()
            ]
            if preferred:
                parquet_files = preferred

        rows = []
        for path in parquet_files:
            rows.extend(pq.read_table(path).to_pylist())

        if normalized_config in {"2d", "3d"}:
            rows = [
                row for row in rows if str(row.get("type", "")).strip().lower() == normalized_config
            ]
        return rows

    if args.hf_dataset:
        if normalized_config == "default":
            return hf_datasets.load_dataset(
                args.hf_dataset,
                split=args.split,
                revision=args.hf_revision,
                token=args.hf_token,
                cache_dir=cache_dir,
            )

        return hf_datasets.load_dataset(
            args.hf_dataset,
            args.config_name,
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


def make_question_id(sample, index, seen_ids):
    for candidate in (sample.get("idx"), sample.get("question_id"), sample.get("id"), sample.get("filename")):
        if candidate is None:
            continue
        qid = str(candidate).strip()
        if qid and qid not in seen_ids:
            seen_ids.add(qid)
            return qid

    qid = str(index)
    while qid in seen_ids:
        qid = f"{qid}_{index}"
    seen_ids.add(qid)
    return qid


def normalize_gold_label(answer, choices):
    text = str(answer).strip()
    for char in text.upper():
        if char in string.ascii_uppercase:
            return char

    normalized_answer = " ".join(text.split()).strip().lower()
    option_chars = list(string.ascii_uppercase[: len(choices)])
    for option_char, choice in zip(option_chars, choices):
        if normalized_answer == " ".join(str(choice).split()).strip().lower():
            return option_char
    return text.upper()


def build_prompt(sample, add_instruction=True):
    prompt = " ".join(str(sample.get("prompt", "")).strip().split())
    if not prompt:
        question = " ".join(str(sample.get("question", "")).strip().split())
        choices = [str(choice).strip() for choice in (sample.get("choices") or [])]
        lines = [question]
        for option_char, choice in zip(string.ascii_uppercase, choices):
            lines.append(f"({option_char}) {choice}")
        prompt = " ".join(part for part in lines if part)
    if add_instruction:
        prompt += "\nAnswer with the option's letter from the given choices directly."
    return prompt


def resolve_relative_image_path(sample, question_id):
    filename = str(sample.get("filename", "")).strip()
    if filename:
        rel_path = Path(filename.replace("\\", "/").lstrip("./"))
        if rel_path.suffix:
            return rel_path
        return rel_path.with_suffix(".png")
    return Path("images") / f"{question_id}.png"


def main():
    parser = argparse.ArgumentParser(description="Convert CV-Bench to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/CVBench",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="nyu-visionx/CV-Bench",
        help="Optional Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="default",
        help="Optional dataset config name: default, 2D, or 3D.",
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
        help="Optional local directory containing CV-Bench parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the letter-only answer instruction to each prompt.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_cvbench_dataset(args, benchmark_dir=benchmark_dir)
    seen_ids = set()
    num_samples = 0

    with open(output_file, "w") as f:
        for index, sample in enumerate(dataset):
            question_id = make_question_id(sample, index, seen_ids)
            choices = [str(choice).strip() for choice in (sample.get("choices") or [])]
            option_chars = list(string.ascii_uppercase[: len(choices)])
            label = normalize_gold_label(sample.get("answer", ""), choices)
            rel_image_path = resolve_relative_image_path(sample, question_id)
            abs_image_path = benchmark_dir / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)

            if not abs_image_path.exists():
                image = to_pil_image(sample["image"])
                image.save(abs_image_path)

            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": build_prompt(sample, add_instruction=not args.no_answer_instruction),
                "label": label,
                "question": str(sample.get("question", "")).strip(),
                "options": choices,
                "option_chars": option_chars,
                "type": str(sample.get("type", "")).strip(),
                "task": str(sample.get("task", "")).strip(),
                "source": str(sample.get("source", "")).strip(),
                "source_dataset": str(sample.get("source_dataset", "")).strip(),
                "source_filename": str(sample.get("source_filename", "")).strip(),
                "filename": str(sample.get("filename", "")).strip(),
                "config_name": args.config_name,
                "split": args.split,
            }
            f.write(json.dumps(row) + "\n")
            num_samples += 1

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir}")


if __name__ == "__main__":
    main()
