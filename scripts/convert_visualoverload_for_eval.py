import argparse
import ast
import hashlib
import json
import os
import re
import string
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


WHITESPACE_RE = re.compile(r"\s+")


def load_visualoverload_dataset(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local VisualOverload snapshot."
            )

        preferred = [path for path in parquet_files if args.split.lower() in path.as_posix().lower()]
        if preferred:
            parquet_files = preferred

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
        image = value.copy()
        image.load()
        return image
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            image = Image.open(BytesIO(image_bytes))
            image.load()
            return image
        if image_path:
            image = Image.open(image_path)
            image.load()
            return image
    if isinstance(value, (str, Path)):
        image = Image.open(value)
        image.load()
        return image
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def normalize_text(value):
    return WHITESPACE_RE.sub(" ", str(value).strip()).strip()


def normalize_options(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [normalize_text(option) for option in value if normalize_text(option)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "null":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [normalize_text(option) for option in parsed if normalize_text(option)]
        return [normalize_text(stripped)]
    return [normalize_text(value)]


def build_prompt(sample, use_default_prompt=True):
    default_prompt = normalize_text(sample.get("default_prompt", ""))
    if use_default_prompt and default_prompt:
        return default_prompt

    question = normalize_text(sample.get("question", ""))
    question_type = normalize_text(sample.get("question_type", "")).lower()
    options = normalize_options(sample.get("options"))

    if question_type == "choice" and options:
        option_text = " ".join(f"{letter}. {option}" for letter, option in zip(string.ascii_uppercase, options))
        return f"{question} Options: {option_text} Answer with the option's letter from the given choices directly."
    if question_type == "counting":
        return f"{question} Answer with a number directly."
    if question_type == "ocr":
        return f"{question} Answer with the text directly."
    return question


def ensure_unique_question_id(sample, index, seen_ids):
    for candidate in (sample.get("question_id"), sample.get("id"), sample.get("uid")):
        if candidate is None:
            continue
        question_id = str(candidate).strip()
        if question_id and question_id not in seen_ids:
            seen_ids.add(question_id)
            return question_id

    question_id = str(index)
    while question_id in seen_ids:
        question_id = f"{question_id}_{index}"
    seen_ids.add(question_id)
    return question_id


def encode_png_bytes(image: Image.Image):
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def stable_image_path(png_bytes):
    digest = hashlib.sha1(png_bytes).hexdigest()[:16]
    return Path("images") / f"{digest}.png"


def main():
    parser = argparse.ArgumentParser(description="Convert VisualOverload to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/VisualOverload",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="paulgavrikov/visualoverload",
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
        help="Optional local directory containing VisualOverload parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--rebuild-prompt",
        action="store_true",
        help="Ignore the dataset default_prompt field and rebuild prompts from raw fields.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    temp_output_file = output_file.with_name(f".{output_file.name}.tmp")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_visualoverload_dataset(args, benchmark_dir=benchmark_dir)
    seen_ids = set()
    unique_images = set()
    num_samples = 0

    if temp_output_file.exists():
        temp_output_file.unlink()

    try:
        with open(temp_output_file, "w") as handle:
            for index, sample in enumerate(dataset):
                question_id = ensure_unique_question_id(sample, index, seen_ids)
                question_type = normalize_text(sample.get("question_type", "")).lower()
                options = normalize_options(sample.get("options"))
                option_chars = list(string.ascii_uppercase[: len(options)])
                prompt = build_prompt(sample, use_default_prompt=not args.rebuild_prompt)

                image = to_pil_image(sample["image"])
                png_bytes = encode_png_bytes(image)
                rel_image_path = stable_image_path(png_bytes)
                abs_image_path = benchmark_dir / rel_image_path
                abs_image_path.parent.mkdir(parents=True, exist_ok=True)
                if not abs_image_path.exists():
                    abs_image_path.write_bytes(png_bytes)
                unique_images.add(rel_image_path.as_posix())

                row = {
                    "question_id": question_id,
                    "image": rel_image_path.as_posix(),
                    "text": prompt,
                    "question": normalize_text(sample.get("question", "")),
                    "question_type": question_type,
                    "options": options,
                    "option_chars": option_chars,
                    "difficulty": normalize_text(sample.get("difficulty", "")).lower(),
                    "category": normalize_text(sample.get("category", "")).lower(),
                    "default_prompt": normalize_text(sample.get("default_prompt", "")),
                    "source_split": args.split,
                }
                handle.write(json.dumps(row) + "\n")
                num_samples += 1
    except Exception:
        if temp_output_file.exists():
            temp_output_file.unlink()
        raise

    if num_samples == 0:
        temp_output_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"VisualOverload conversion produced zero samples for split '{args.split}'. "
            "Refusing to overwrite the final question file."
        )

    temp_output_file.replace(output_file)
    print(f"Saved {num_samples} samples to {output_file}")
    print(f"Saved {len(unique_images)} unique images under {benchmark_dir / 'images'}")


if __name__ == "__main__":
    main()
