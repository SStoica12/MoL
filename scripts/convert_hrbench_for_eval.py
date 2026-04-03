import argparse
import base64
import csv
import hashlib
import json
import re
import string
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image


SPLIT_ALIASES = {
    "4k": "hrbench_4k",
    "8k": "hrbench_8k",
    "hr-bench-4k": "hrbench_4k",
    "hr-bench-8k": "hrbench_8k",
    "hrbench4k": "hrbench_4k",
    "hrbench8k": "hrbench_8k",
    "hrbench_4k": "hrbench_4k",
    "hrbench_8k": "hrbench_8k",
}

SPLIT_TO_DATASET_NAME = {
    "hrbench_4k": "hrbench4k",
    "hrbench_8k": "hrbench8k",
}

SPLIT_TO_PARQUET = {
    "hrbench_4k": "hr_bench_4k.parquet",
    "hrbench_8k": "hr_bench_8k.parquet",
}

SPLIT_TO_TSV = {
    "hrbench_4k": "hr_bench_4k.tsv",
    "hrbench_8k": "hr_bench_8k.tsv",
}

OPTION_KEYS = ["A", "B", "C", "D"]
IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}


def normalize_split_name(value):
    if not isinstance(value, str):
        raise ValueError(f"Invalid split: {value}")
    key = value.strip().lower()
    normalized = SPLIT_ALIASES.get(key, key)
    if normalized not in SPLIT_TO_DATASET_NAME:
        raise ValueError(
            f"Unsupported split: {value}. Expected one of: {sorted(SPLIT_TO_DATASET_NAME)}"
        )
    return normalized


def format_question(question, options, add_instruction=True):
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(f"Too many options ({len(options)}), max supported is 26.")

    lines = [str(question).strip()]
    for idx, option in enumerate(options):
        lines.append(f"{string.ascii_uppercase[idx]}. {str(option).strip()}")
    if add_instruction:
        lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def resolve_gold_label(row, options):
    answer = row.get("answer")
    valid_letters = string.ascii_uppercase[: len(options)]

    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return valid_letters[answer]

    if answer is not None:
        answer_text = str(answer).strip()
        upper_answer = answer_text.upper()
        if upper_answer in valid_letters:
            return upper_answer

        normalized_answer = normalize_text(answer_text)
        for idx, option in enumerate(options):
            if normalize_text(option) == normalized_answer:
                return valid_letters[idx]

    raise ValueError(f"Unable to resolve gold label from answer={answer!r}")


def clean_image_value(value):
    text = str(value).strip()
    if not text:
        raise ValueError("Empty image field")

    if (text.startswith("b'") and text.endswith("'")) or (text.startswith('b"') and text.endswith('"')):
        text = text[2:-1]

    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]

    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]

    return re.sub(r"\s+", "", text)


def decode_image_bytes(value):
    cleaned = clean_image_value(value)
    try:
        return base64.b64decode(cleaned, validate=False)
    except Exception as exc:
        raise ValueError("Failed to decode base64 image data") from exc


def infer_image_extension(raw_bytes):
    try:
        image = Image.open(BytesIO(raw_bytes))
        image.load()
        image_format = (image.format or "").upper()
        return IMAGE_EXTENSIONS.get(image_format, ".jpg")
    except Exception:
        return ".jpg"


def save_image(raw_value, image_dir: Path, image_cache):
    raw_bytes = decode_image_bytes(raw_value)
    digest = hashlib.sha1(raw_bytes).hexdigest()
    cached = image_cache.get(digest)
    if cached is not None:
        return cached

    ext = infer_image_extension(raw_bytes)
    image_path = image_dir / f"{digest[:24]}{ext}"
    if not image_path.exists():
        image_path.write_bytes(raw_bytes)
    rel_path = image_path.relative_to(image_dir.parent).as_posix()
    image_cache[digest] = rel_path
    return rel_path


def load_rows_from_tsv(tsv_file: Path):
    with open(tsv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def load_rows_from_parquet(parquet_file: Path):
    try:
        import pandas as pd

        dataframe = pd.read_parquet(parquet_file)
        return dataframe.to_dict(orient="records")
    except ImportError:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading HR-Bench parquet files requires either pandas or pyarrow. "
                "Install one of them, or use --tsv-file instead."
            ) from exc

        table = pq.read_table(parquet_file)
        return table.to_pylist()


def load_rows_from_hf(args, split):
    dataset_name = args.hf_dataset
    if not dataset_name:
        raise ValueError("Missing --hf-dataset")

    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None

    if load_dataset is not None:
        load_kwargs = {
            "path": dataset_name,
            "split": split,
            "revision": args.hf_revision,
        }
        if args.hf_token:
            load_kwargs["token"] = args.hf_token
        if args.cache_dir:
            load_kwargs["cache_dir"] = args.cache_dir

        try:
            dataset = load_dataset(**load_kwargs)
        except Exception:
            load_kwargs["name"] = "hrbench_version_split"
            dataset = load_dataset(**load_kwargs)
        return [dict(row) for row in dataset]

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading HR-Bench from Hugging Face requires either `datasets` or `huggingface_hub`."
        ) from exc

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    parquet_name = SPLIT_TO_PARQUET[split]
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        revision=args.hf_revision,
        token=args.hf_token,
        local_dir=str(benchmark_dir),
        cache_dir=args.cache_dir,
        allow_patterns=[parquet_name, "README.md"],
    )

    parquet_file = benchmark_dir / parquet_name
    if parquet_file.exists():
        return load_rows_from_parquet(parquet_file)
    raise FileNotFoundError(
        f"Downloaded snapshot for {dataset_name}, but {parquet_name} was not found."
    )


def load_rows(args, split):
    if args.parquet_file:
        parquet_file = Path(args.parquet_file).expanduser().resolve()
        print(f"Loading parquet file: {parquet_file}")
        return load_rows_from_parquet(parquet_file)

    if args.tsv_file:
        tsv_file = Path(args.tsv_file).expanduser().resolve()
        print(f"Loading tsv file: {tsv_file}")
        return load_rows_from_tsv(tsv_file)

    print(f"Loading Hugging Face dataset: {args.hf_dataset} [{split}]")
    return load_rows_from_hf(args, split)


def build_examples(rows, benchmark_dir: Path, split, add_instruction=True, strict=False):
    dataset_name = SPLIT_TO_DATASET_NAME[split]
    image_dir = benchmark_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    per_category = defaultdict(int)
    image_cache = {}
    seen_qids = set()

    for row_idx, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        options = [str(row.get(key, "")).strip() for key in OPTION_KEYS]
        if not question or any(not option for option in options):
            message = f"Invalid row (missing question/options) at row {row_idx}"
            if strict:
                raise ValueError(message)
            print(f"[Warning] {message}")
            continue

        category = str(row.get("category", "unknown")).strip().lower() or "unknown"
        cycle_category = str(row.get("cycle_category", "")).strip()
        raw_index = str(row.get("index", row_idx)).strip() or str(row_idx)
        question_id = f"{dataset_name}_{raw_index}_{cycle_category or 'na'}"
        if question_id in seen_qids:
            question_id = f"{question_id}_{row_idx}"
        seen_qids.add(question_id)

        try:
            label = resolve_gold_label(row, options)
            image_rel_path = save_image(row.get("image", ""), image_dir=image_dir, image_cache=image_cache)
        except Exception as exc:
            message = f"Failed to process row {row_idx} ({question_id}): {exc}"
            if strict:
                raise
            print(f"[Warning] {message}")
            continue

        examples.append(
            {
                "question_id": question_id,
                "image": image_rel_path,
                "text": format_question(question, options, add_instruction=add_instruction),
                "label": label,
                "category": category,
                "cycle_category": cycle_category,
                "options": options,
                "raw_index": raw_index,
                "dataset": dataset_name,
            }
        )
        per_category[category] += 1

    return examples, per_category, len(image_cache)


def main():
    parser = argparse.ArgumentParser(description="Convert HR-Bench raw data to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/HRBench4K",
        help="Directory where images and converted questions will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to convert: hrbench_4k or hrbench_8k.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="DreamMr/HR-Bench",
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
        help="Optional cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--parquet-file",
        type=str,
        default=None,
        help="Optional local parquet file. Overrides --hf-dataset.",
    )
    parser.add_argument(
        "--tsv-file",
        type=str,
        default=None,
        help="Optional local TSV file. Overrides --hf-dataset and avoids parquet dependencies.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the letter-only answer instruction to each prompt.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed rows instead of warning and skipping.",
    )
    args = parser.parse_args()

    split = normalize_split_name(args.split)
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args, split)
    examples, per_category, num_unique_images = build_examples(
        rows=rows,
        benchmark_dir=benchmark_dir,
        split=split,
        add_instruction=not args.no_answer_instruction,
        strict=args.strict,
    )

    with open(output_file, "w") as f:
        for sample in examples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(examples)} samples to {output_file}")
    print(f"Unique decoded images: {num_unique_images}")
    for category, count in sorted(per_category.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()