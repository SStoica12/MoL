import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


OPTION_RE = re.compile(r"^\(?([A-Z])\)?[\.:]?\s*(.*)$")


def load_mme_realworld_dataset(args, benchmark_dir: Path):
    if args.hf_dataset:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for --hf-dataset. "
                "Install it with `pip install huggingface_hub`."
            ) from e

        if args.cache_dir:
            download_dir = Path(args.cache_dir).expanduser().resolve()
        else:
            download_dir = benchmark_dir / ".hf_snapshot"
        download_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset snapshot: {args.hf_dataset} -> {download_dir}")
        snapshot_path = snapshot_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            revision=args.hf_revision,
            token=args.hf_token,
            local_dir=str(download_dir),
            allow_patterns=["**/*.parquet"],
        )
        snapshot_dir = Path(snapshot_path)
        parquet_files = sorted(snapshot_dir.rglob(f"*{args.split}*.parquet"))
        if not parquet_files:
            parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in snapshot: {snapshot_dir}")
        rows = []
        for path in parquet_files:
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    parquet_dir = Path(args.parquet_dir).expanduser().resolve() if args.parquet_dir else benchmark_dir
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {parquet_dir}. "
            "Pass --hf-dataset or point --parquet-dir at a local MME-RealWorld parquet snapshot."
        )

    rows = []
    for path in parquet_files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def to_pil_image(sample):
    image_value = sample.get("image")
    if isinstance(image_value, Image.Image):
        return image_value
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes))
        if image_path:
            return Image.open(image_path)

    raw_bytes = sample.get("bytes")
    if raw_bytes is None:
        raise KeyError("MME-RealWorld sample does not contain 'image' or 'bytes'")
    if isinstance(raw_bytes, str):
        raw_bytes = base64.b64decode(raw_bytes)
    return Image.open(BytesIO(raw_bytes))


def normalize_option(option_text):
    option_text = str(option_text).strip()
    match = OPTION_RE.match(option_text)
    if match:
        option_char = match.group(1)
        option_value = match.group(2).strip()
        return option_char, option_value
    return None, option_text


def build_prompt(question, options):
    lines = [str(question).strip()]
    for option in options:
        option_char, option_value = normalize_option(option)
        if option_char is None:
            lines.append(str(option).strip())
        else:
            lines.append(f"{option_char}. {option_value}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="Convert MME-RealWorld to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/MME-RealWorld",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="yifanzhang114/MME-RealWorld-Lmms-eval",
        help=(
            "Optional Hugging Face dataset repo id. "
            "The default uses the lmms-eval mirror of the official MME-RealWorld benchmark."
        ),
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
        help="Optional local directory containing MME-RealWorld parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_mme_realworld_dataset(args, benchmark_dir=benchmark_dir)
    total_samples = len(dataset)
    split_dir = benchmark_dir / "images" / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    completed_ids, valid_rows = prepare_output_file(output_file)

    if completed_ids:
        print(
            f"Resuming MME-RealWorld conversion: found {len(completed_ids)} completed "
            f"question ids in {output_file}.",
            flush=True,
        )
    else:
        print(f"Starting MME-RealWorld conversion from scratch: {total_samples} samples.", flush=True)

    append_mode = "a" if valid_rows > 0 else "w"
    new_rows = 0
    reused_rows = 0
    written_images = 0
    reused_images = 0
    repaired_images = 0

    with open(output_file, append_mode) as f:
        progress = tqdm(
            total=total_samples,
            desc="Converting MME-RealWorld",
            unit="sample",
        )
        for index, sample in enumerate(dataset):
            question_id = str(sample.get("index", index))
            rel_image_path = Path("images") / args.split / f"{question_id}.png"
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
                image = to_pil_image(sample).convert("RGB")
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

            options = [str(option).strip() for option in (sample.get("multi-choice options") or sample.get("options") or [])]
            option_chars = []
            normalized_options = []
            for option in options:
                option_char, option_value = normalize_option(option)
                if option_char is None:
                    normalized_options.append(option)
                else:
                    option_chars.append(option_char)
                    normalized_options.append(option_value)

            if not option_chars:
                option_chars = [chr(ord("A") + idx) for idx in range(len(normalized_options))]

            row = {
                "question_id": question_id,
                "image": rel_image_path.as_posix(),
                "text": build_prompt(sample.get("question", ""), options),
                "label": str(sample.get("answer", "")).strip().upper(),
                "question": str(sample.get("question", "")).strip(),
                "options": normalized_options,
                "option_chars": option_chars,
                "category": str(sample.get("category", "unknown")),
                "subcategory": str(sample.get("l2-category", sample.get("subcategory", "unknown"))),
                "source_split": args.split,
            }
            f.write(json.dumps(row) + "\n")
            completed_ids.add(question_id)
            new_rows += 1
            progress.update(1)

        progress.close()

    total_rows = len(completed_ids)
    print(f"Prepared {total_rows} total samples at {output_file}", flush=True)
    print(
        f"New rows written: {new_rows}; reused rows: {reused_rows}; "
        f"new images written: {written_images}; repaired images: {repaired_images}; "
        f"reused images: {reused_images}",
        flush=True,
    )
    print(f"Extracted images under {split_dir}", flush=True)


if __name__ == "__main__":
    main()
