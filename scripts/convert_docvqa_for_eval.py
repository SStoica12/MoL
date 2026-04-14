import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


def load_docvqa_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.hf_dataset:
        dataset = hf_datasets.load_dataset(
            args.hf_dataset,
            args.hf_config,
            split=args.split,
            revision=args.hf_revision,
            token=args.hf_token,
            cache_dir=cache_dir,
        )
        if hasattr(dataset, "column_names") and "image" in dataset.column_names:
            try:
                dataset = dataset.cast_column("image", hf_datasets.Image(decode=False))
            except Exception:
                pass
        return dataset

    parquet_dir = Path(args.parquet_dir).expanduser().resolve() if args.parquet_dir else benchmark_dir
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {parquet_dir}. "
            "Pass --hf-dataset or point --parquet-dir at a local DocVQA parquet snapshot."
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
    return f"Answer the question according to the image using a single word or phrase.\n{question}"


def image_file_is_valid(path: Path):
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_rgb_png_atomic(image: Image.Image, output_path: Path):
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def scan_existing_output(output_file: Path):
    completed_ids = set()
    valid_rows = 0
    valid_bytes = 0

    if not output_file.exists():
        return completed_ids, valid_rows, valid_bytes

    with open(output_file, "rb") as f:
        while True:
            offset = f.tell()
            raw_line = f.readline()
            if not raw_line:
                valid_bytes = f.tell()
                break

            stripped = raw_line.strip()
            if not stripped:
                valid_bytes = f.tell()
                continue

            try:
                row = json.loads(stripped)
            except Exception:
                valid_bytes = offset
                break

            question_id = str(row.get("question_id", "")).strip()
            if question_id:
                completed_ids.add(question_id)
            valid_rows += 1
            valid_bytes = f.tell()

    return completed_ids, valid_rows, valid_bytes


def prepare_output_file(output_file: Path):
    completed_ids, valid_rows, valid_bytes = scan_existing_output(output_file)
    current_size = output_file.stat().st_size if output_file.exists() else 0

    if output_file.exists() and current_size != valid_bytes:
        with open(output_file, "rb+") as f:
            f.truncate(valid_bytes)
        print(
            f"Trimmed incomplete JSONL tail in {output_file} "
            f"from {current_size} bytes to {valid_bytes} bytes.",
            flush=True,
        )

    return completed_ids, valid_rows


def main():
    parser = argparse.ArgumentParser(description="Convert DocVQA to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/DocVQA",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="lmms-lab/DocVQA",
        help="Optional Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="DocVQA",
        help="Dataset config name, e.g. DocVQA.",
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
        help="Optional local directory containing DocVQA parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the single-word-or-phrase instruction to each prompt.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_docvqa_dataset(args, benchmark_dir=benchmark_dir)
    total_samples = len(dataset)
    split_dir = benchmark_dir / "images" / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    completed_ids, valid_rows = prepare_output_file(output_file)

    if completed_ids:
        print(
            f"Resuming DocVQA conversion: found {len(completed_ids)} completed "
            f"question ids in {output_file}.",
            flush=True,
        )
    else:
        print(f"Starting DocVQA conversion from scratch: {total_samples} samples.", flush=True)

    append_mode = "a" if valid_rows > 0 else "w"
    new_rows = 0
    reused_rows = 0
    written_images = 0
    repaired_images = 0
    reused_images = 0

    with open(output_file, append_mode) as f:
        progress = tqdm(total=total_samples, desc="Converting DocVQA", unit="sample")
        for index, sample in enumerate(dataset):
            question_id = str(sample.get("questionId", index))
            image_name = f"{question_id}.png"
            rel_image_path = Path("images") / args.split / image_name
            abs_image_path = benchmark_dir / rel_image_path
            row_already_written = question_id in completed_ids
            image_existed_before = abs_image_path.exists()
            image_valid = image_file_is_valid(abs_image_path)

            if row_already_written and image_valid:
                reused_rows += 1
                reused_images += 1
                progress.update(1)
                continue

            if not image_valid:
                image = to_pil_image(sample["image"]).convert("RGB")
                save_rgb_png_atomic(image, abs_image_path)
                if image_existed_before:
                    repaired_images += 1
                else:
                    written_images += 1
            else:
                reused_images += 1

            if row_already_written:
                reused_rows += 1
                progress.update(1)
                continue

            answers = sample.get("answers") or []
            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": build_prompt(
                    sample.get("question", ""),
                    add_instruction=not args.no_answer_instruction,
                ),
                "answers": [str(answer).strip() for answer in answers],
                "question": str(sample.get("question", "")).strip(),
                "question_types": [str(value) for value in (sample.get("question_types") or [])],
                "doc_id": sample.get("docId"),
                "source_split": args.split,
            }
            f.write(json.dumps(row) + "\n")
            completed_ids.add(question_id)
            new_rows += 1
            progress.update(1)

        progress.close()

    print(f"Prepared {len(completed_ids)} total samples at {output_file}")
    print(
        f"New rows written: {new_rows}; reused rows: {reused_rows}; "
        f"new images written: {written_images}; repaired images: {repaired_images}; "
        f"reused images: {reused_images}",
        flush=True,
    )
    print(f"Extracted images under {split_dir}")


if __name__ == "__main__":
    main()
