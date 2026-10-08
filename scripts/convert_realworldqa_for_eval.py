import argparse
import json
import os
import re
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


OPTION_RE = re.compile(r"(?:^|\n)\s*([A-Z])\.\s*(.+?)(?=(?:\n\s*[A-Z]\.\s)|$)", re.S)
INLINE_OPTION_RE = re.compile(r"(?:^|\s)([A-Z])\.\s*(.*?)(?=(?:\s+[A-Z]\.\s)|$)", re.S)


def load_realworldqa_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.hf_dataset:
        return hf_datasets.load_dataset(
            args.hf_dataset,
            split=args.split,
            revision=args.hf_revision,
            token=args.hf_token,
            cache_dir=cache_dir,
        )

    parquet_dir = Path(args.parquet_dir).expanduser().resolve() if args.parquet_dir else benchmark_dir
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {parquet_dir}. "
            "Pass --hf-dataset or point --parquet-dir at a local RealWorldQA snapshot."
        )
    rows = []
    for path in parquet_files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


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
        sample.get("question_id"),
        sample.get("id"),
        sample.get("image_path"),
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


def extract_options(question):
    matches = OPTION_RE.findall(str(question))
    if len(matches) < 2:
        matches = INLINE_OPTION_RE.findall(str(question))
    if len(matches) < 2:
        return [], []

    option_chars = []
    options = []
    for option_char, option_text in matches:
        option_chars.append(option_char.strip())
        options.append(" ".join(str(option_text).strip().split()))
    return options, option_chars


def infer_answer_type(answer, option_chars):
    answer = str(answer).strip().upper()
    if len(answer) == 1 and answer in option_chars:
        return "multiple_choice"
    return "open"


def main():
    parser = argparse.ArgumentParser(description="Convert RealWorldQA to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/RealWorldQA",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo id, e.g. lmms-lab/RealWorldQA.",
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
        help="Optional local directory containing RealWorldQA parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_realworldqa_dataset(args, benchmark_dir=benchmark_dir)
    seen_ids = set()
    num_samples = 0

    with open(output_file, "w") as f:
        for index, sample in enumerate(dataset):
            question_id = make_question_id(sample, index, seen_ids)
            question = str(sample.get("question", "")).strip()
            answer = str(sample.get("answer", "")).strip()
            raw_image_path = sample.get("image_path")

            rel_image_path = normalize_relative_image_path(raw_image_path, question_id)
            abs_image_path = benchmark_dir / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)

            image = to_pil_image(sample["image"])
            image.save(abs_image_path, format="PNG")

            options, option_chars = extract_options(question)
            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": question,
                "label": answer,
                "answer_type": infer_answer_type(answer, option_chars),
                "options": options,
                "option_chars": option_chars,
                "source_split": args.split,
                "source_image_path": raw_image_path,
            }
            f.write(json.dumps(row) + "\n")
            num_samples += 1

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images'}")


if __name__ == "__main__":
    main()
