import argparse
import json
import string
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
DOWNLOAD_PATTERNS = [
    "**/*.json",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.png",
    "**/*.webp",
    "**/*.bmp",
    "**/*.tif",
    "**/*.tiff",
    "**/*.JPG",
    "**/*.JPEG",
    "**/*.PNG",
    "**/*.WEBP",
    "**/*.BMP",
    "**/*.TIF",
    "**/*.TIFF",
]


def find_image_path(annotation_path: Path):
    stem = annotation_path.with_suffix("")
    for ext in IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    for candidate in annotation_path.parent.glob(f"{stem.name}.*"):
        if candidate.suffix.lower() != ".json" and candidate.is_file():
            return candidate
    return None


def format_question(question, options, add_instruction=True):
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(f"Too many options ({len(options)}), max supported is 26.")

    question_lines = [question.strip()]
    for idx, option in enumerate(options):
        question_lines.append(f"{string.ascii_uppercase[idx]}. {str(option).strip()}")

    if add_instruction:
        question_lines.append("Answer with the option's letter from the given choices directly.")

    return "\n".join(question_lines)


def infer_categories(benchmark_dir: Path):
    categories = []
    for path in sorted(benchmark_dir.iterdir()):
        if path.is_dir() and any(path.glob("*.json")):
            categories.append(path.name)
    return categories


def parse_gold_label(annotation, num_options):
    if num_options < 1:
        return "A"

    answer = annotation.get("answer", annotation.get("label"))
    valid_letters = string.ascii_uppercase[:num_options]

    if isinstance(answer, int):
        if 0 <= answer < num_options:
            return valid_letters[answer]
    elif isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in valid_letters:
            return answer

    # V*Bench default: first option is correct.
    return "A"


def resolve_benchmark_dir(args):
    if args.hf_dataset:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for --hf-dataset. "
                "Please install it with `pip install huggingface_hub`."
            ) from e

        if args.download_dir:
            download_dir = Path(args.download_dir).expanduser().resolve()
        else:
            download_dir = (Path("./playground/data/eval") / args.hf_dataset.split("/")[-1]).resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset snapshot: {args.hf_dataset} -> {download_dir}")
        snapshot_path = snapshot_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            revision=args.hf_revision,
            token=args.hf_token,
            local_dir=str(download_dir),
            allow_patterns=DOWNLOAD_PATTERNS,
        )
        benchmark_dir = Path(snapshot_path).resolve()
    elif args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    else:
        raise ValueError("Please provide either --benchmark-dir or --hf-dataset.")

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    return benchmark_dir


def build_examples(benchmark_dir: Path, categories, add_instruction=True, strict=False):
    examples = []
    per_category = {}

    for category in categories:
        category_dir = benchmark_dir / category
        if not category_dir.exists():
            if strict:
                raise FileNotFoundError(f"Category folder not found: {category_dir}")
            print(f"[Warning] Skipping missing category folder: {category_dir}")
            continue

        ann_files = sorted(category_dir.glob("*.json"))
        if not ann_files:
            if strict:
                raise FileNotFoundError(f"No annotation json files found in: {category_dir}")
            print(f"[Warning] No annotation json files found in: {category_dir}")
            continue

        count = 0
        for ann_path in ann_files:
            with open(ann_path, "r") as f:
                annotation = json.load(f)

            question = annotation.get("question", "").strip()
            options = annotation.get("options", [])
            if not question or not options:
                msg = f"Invalid annotation (missing question/options): {ann_path}"
                if strict:
                    raise ValueError(msg)
                print(f"[Warning] {msg}")
                continue

            image_path = find_image_path(ann_path)
            if image_path is None:
                msg = f"Image file not found for annotation: {ann_path}"
                if strict:
                    raise FileNotFoundError(msg)
                print(f"[Warning] {msg}")
                continue

            rel_image_path = image_path.relative_to(benchmark_dir).as_posix()
            sample_id = f"{category}/{ann_path.stem}"
            examples.append({
                "question_id": sample_id,
                "image": rel_image_path,
                "text": format_question(question, options, add_instruction=add_instruction),
                "label": parse_gold_label(annotation, num_options=len(options)),
                "category": category,
                "options": options,
                "source_json": ann_path.relative_to(benchmark_dir).as_posix(),
            })
            count += 1

        per_category[category] = count

    return examples, per_category


def main():
    parser = argparse.ArgumentParser(description="Convert V*Bench raw folders to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Path to local V*Bench root folder.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo id (e.g., craigwu/vstar_bench).",
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
        "--download-dir",
        type=str,
        default=None,
        help="Where to download --hf-dataset snapshot (defaults to ./playground/data/eval/<dataset_name>).",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category folders to include. If omitted, auto-detect folders containing json annotations.",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the letter-only answer instruction to each question.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing files or malformed annotations instead of warning and skipping.",
    )
    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    categories = args.categories if args.categories else infer_categories(benchmark_dir)
    if not categories:
        raise FileNotFoundError(
            f"No category folders with json annotations found under {benchmark_dir}. "
            "If this is a local folder, verify dataset extraction."
        )
    if not args.categories:
        print(f"Auto-detected categories: {', '.join(categories)}")

    examples, per_category = build_examples(
        benchmark_dir=benchmark_dir,
        categories=categories,
        add_instruction=not args.no_answer_instruction,
        strict=args.strict,
    )

    with open(output_file, "w") as f:
        for sample in examples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(examples)} samples to {output_file}")
    for category, count in per_category.items():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
