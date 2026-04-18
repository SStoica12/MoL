import argparse
import json
import os
import re
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


OPTION_BLOCK_RE = re.compile(r"\boptions?\s*:\s*", flags=re.IGNORECASE)
OPTION_RE = re.compile(r"(?:^|[\s\n])([A-D])\s*[:.)]\s*(.*?)(?=(?:[\s\n]+[A-D]\s*[:.)]\s*)|$)", re.S)


def load_mmstar_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local MMStar snapshot."
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


def normalize_relative_image_path(raw_image_path, question_id):
    if raw_image_path:
        rel_path = Path(str(raw_image_path).replace("\\", "/").lstrip("./"))
        return (Path("images") / rel_path).with_suffix(".png")
    return Path("images") / f"{question_id}.png"


def make_question_id(sample, index, seen_ids):
    candidates = [
        sample.get("index"),
        sample.get("question_id"),
        sample.get("id"),
        sample.get("meta_info", {}).get("image_path") if isinstance(sample.get("meta_info"), dict) else None,
    ]
    for candidate in candidates:
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


def parse_question_and_options(question):
    compact_question = " ".join(str(question).strip().split())
    if not compact_question:
        return "", [], []

    option_block_match = OPTION_BLOCK_RE.search(compact_question)
    if option_block_match:
        stem = compact_question[: option_block_match.start()].strip()
        option_block = compact_question[option_block_match.end() :].strip()
    else:
        stem = compact_question
        option_block = compact_question

    matches = OPTION_RE.findall(option_block)
    if not matches and option_block != compact_question:
        matches = OPTION_RE.findall(compact_question)

    option_chars = []
    options = []
    for option_char, option_text in matches:
        option_chars.append(option_char.strip())
        options.append(" ".join(str(option_text).strip().split()))

    return stem, options, option_chars


def format_question(question, add_instruction=True):
    stem, options, option_chars = parse_question_and_options(question)
    if not options:
        text = " ".join(str(question).strip().split())
        if add_instruction:
            text += "\nAnswer with the option's letter from the given choices directly."
        return text, options, option_chars

    lines = [stem or "Select the best answer."]
    for option_char, option_text in zip(option_chars, options):
        lines.append(f"{option_char}. {option_text}")
    if add_instruction:
        lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines), options, option_chars


def main():
    parser = argparse.ArgumentParser(description="Convert MMStar to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/MMStar",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="Lin-Chen/MMStar",
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
        help="Optional local directory containing MMStar parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
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

    dataset = load_mmstar_dataset(args, benchmark_dir=benchmark_dir)
    seen_ids = set()
    num_samples = 0

    with open(output_file, "w") as f:
        for index, sample in enumerate(dataset):
            question_id = make_question_id(sample, index, seen_ids)
            question_text, options, option_chars = format_question(
                sample.get("question", ""),
                add_instruction=not args.no_answer_instruction,
            )
            answer = str(sample.get("answer", "")).strip().upper()
            raw_image_path = None
            if isinstance(sample.get("meta_info"), dict):
                raw_image_path = sample["meta_info"].get("image_path")

            rel_image_path = normalize_relative_image_path(raw_image_path, question_id)
            abs_image_path = benchmark_dir / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)

            image = to_pil_image(sample["image"])
            image.save(abs_image_path, format="PNG")

            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": question_text,
                "label": answer,
                "category": str(sample.get("category", "")).strip(),
                "subcategory": str(sample.get("l2_category", "")).strip(),
                "options": options,
                "option_chars": option_chars,
                "source_split": args.split,
                "source_meta_info": sample.get("meta_info"),
                "raw_question": sample.get("question"),
            }
            f.write(json.dumps(row) + "\n")
            num_samples += 1

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images'}")


if __name__ == "__main__":
    main()
