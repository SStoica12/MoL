import argparse
import json
import os
import re
import string
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


OPTION_BLOCK_RE = re.compile(r"\boptions?\s*:\s*", flags=re.IGNORECASE)
OPTION_RE = re.compile(r"(?:^|[\s\n;])([A-D])\s*[:.)]\s*(.*?)(?=(?:[\s\n;]+[A-D]\s*[:.)]\s*)|$)", re.S)


def load_naturalbench_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local NaturalBench snapshot."
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


def parse_question_type(raw_question_type):
    text = str(raw_question_type).strip().lower().replace(" ", "_")
    if "multiple" in text:
        return "multiple_choice"
    return "yes_no"


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


def format_question(question, question_type, add_instruction=True):
    compact_question = " ".join(str(question).strip().split())
    if question_type == "multiple_choice":
        stem, options, option_chars = parse_question_and_options(compact_question)
        if options:
            lines = [stem or "Select the best answer."]
            for option_char, option_text in zip(option_chars, options):
                lines.append(f"{option_char}. {option_text}")
            if add_instruction:
                lines.append("Answer with the option's letter from the given choices directly.")
            return "\n".join(lines), options, option_chars

    if add_instruction:
        compact_question += "\nAnswer with Yes or No directly."
    return compact_question, ["Yes", "No"], ["YES", "NO"]


def normalize_gold_label(answer, question_type):
    text = str(answer).strip()
    if question_type == "multiple_choice":
        for char in text.upper():
            if char in string.ascii_uppercase:
                return char
        return text.upper()
    normalized = text.lower()
    if normalized.startswith("y"):
        return "YES"
    if normalized.startswith("n"):
        return "NO"
    return text.upper()


def make_image_path(index_value, image_index):
    return Path("images") / f"{index_value}_image_{image_index}.png"


def sample_index_value(sample, fallback_index):
    for key in ("Index", "index", "idx", "id"):
        candidate = sample.get(key)
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return str(fallback_index)


def main():
    parser = argparse.ArgumentParser(description="Convert NaturalBench to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/NaturalBench",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="BaiqiL/NaturalBench",
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
        help="Optional local directory containing NaturalBench parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the direct-answer instruction to each prompt.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_naturalbench_dataset(args, benchmark_dir=benchmark_dir)
    num_samples = 0

    with open(output_file, "w") as f:
        for row_index, sample in enumerate(dataset):
            index_value = sample_index_value(sample, row_index)
            source = str(sample.get("Source", sample.get("source", ""))).strip()
            raw_question_type = sample.get("Question Type", sample.get("Question_Type", sample.get("question_type", "")))
            question_type = parse_question_type(raw_question_type)

            image_paths = {}
            for image_index in (0, 1):
                image_value = sample.get(f"Image_{image_index}")
                rel_image_path = make_image_path(index_value, image_index)
                abs_image_path = benchmark_dir / rel_image_path
                abs_image_path.parent.mkdir(parents=True, exist_ok=True)
                if not abs_image_path.exists():
                    image = to_pil_image(image_value)
                    image.save(abs_image_path, format="PNG")
                image_paths[image_index] = rel_image_path.as_posix()

            for question_index in (0, 1):
                question_key = f"Question_{question_index}"
                question_text = sample.get(question_key, "")
                prompt, options, option_chars = format_question(
                    question_text,
                    question_type,
                    add_instruction=not args.no_answer_instruction,
                )

                for image_index in (0, 1):
                    label_key = f"Image_{image_index}_Question_{question_index}"
                    label = normalize_gold_label(sample.get(label_key, ""), question_type)
                    question_id = f"{index_value}_q{question_index}_i{image_index}"
                    row = {
                        "question_id": question_id,
                        "sample_id": str(index_value),
                        "question_group_id": f"{index_value}_q{question_index}",
                        "image_group_id": f"{index_value}_i{image_index}",
                        "question_index": question_index,
                        "image_index": image_index,
                        "image": image_paths[image_index],
                        "text": prompt,
                        "label": label,
                        "question_type": question_type,
                        "question_type_raw": str(raw_question_type).strip(),
                        "source": source,
                        "question": str(question_text).strip(),
                        "options": options,
                        "option_chars": option_chars,
                        "split": args.split,
                    }
                    f.write(json.dumps(row) + "\n")
                    num_samples += 1

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images'}")


if __name__ == "__main__":
    main()