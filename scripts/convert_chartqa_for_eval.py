import argparse
import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


def load_chartqa_dataset(args, benchmark_dir: Path):
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
            "Pass --hf-dataset or point --parquet-dir at a local ChartQA snapshot."
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


def build_prompt(question: str, add_instruction: bool = True):
    question = str(question).strip()
    if not add_instruction:
        return question
    return f"{question}\nAnswer with a single word, number, or short phrase."


def make_question_id(split: str, index: int):
    return f"{split}_{index}"


def main():
    parser = argparse.ArgumentParser(description="Convert ChartQA to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/ChartQA",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo id, e.g. lmms-lab/ChartQA.",
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
        help="Optional local directory containing ChartQA parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append a short answer-format instruction to each prompt.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_chartqa_dataset(args, benchmark_dir=benchmark_dir)
    type_counts = Counter()

    with open(output_file, "w") as f:
        for index, sample in enumerate(dataset):
            question_id = make_question_id(args.split, index)
            rel_image_path = Path("images") / args.split / f"{question_id}.png"
            abs_image_path = benchmark_dir / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)

            image = to_pil_image(sample["image"])
            image.save(abs_image_path, format="PNG")

            sample_type = str(sample.get("type", "unknown"))
            type_counts[sample_type] += 1
            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": build_prompt(
                    sample["question"],
                    add_instruction=not args.no_answer_instruction,
                ),
                "label": str(sample["answer"]).strip(),
                "type": sample_type,
                "question": str(sample["question"]).strip(),
                "answer": str(sample["answer"]).strip(),
                "source_split": args.split,
            }
            f.write(json.dumps(row) + "\n")

    total = sum(type_counts.values())
    print(f"Saved {total} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images' / args.split}")
    for sample_type, count in sorted(type_counts.items()):
        print(f"{sample_type}: {count}")


if __name__ == "__main__":
    main()
