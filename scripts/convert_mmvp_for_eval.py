import argparse
import csv
import json
import re
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DOWNLOAD_PATTERNS = [
    "Questions.csv",
    "Questions.xlsx",
    "README.md",
    "MMVP Images/*.jpg",
    "MMVP Images/*.jpeg",
    "MMVP Images/*.png",
    "MMVP Images/*.webp",
    "MMVP Images/*.bmp",
]
OPTION_RE = re.compile(r"\((a|b)\)\s*([^()]+?)(?=\s*\((?:a|b)\)|$)", flags=re.IGNORECASE)


def resolve_benchmark_dir(args):
    if args.hf_dataset:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for --hf-dataset. "
                "Please install it with `pip install huggingface_hub`."
            ) from exc

        benchmark_dir = Path(args.download_dir or args.benchmark_dir).expanduser().resolve()
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            revision=args.hf_revision,
            token=args.hf_token,
            local_dir=str(benchmark_dir),
            allow_patterns=DOWNLOAD_PATTERNS,
        )
        return benchmark_dir

    if not args.benchmark_dir:
        raise ValueError("Please provide either --benchmark-dir or --hf-dataset.")

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    return benchmark_dir


def resolve_questions_file(benchmark_dir: Path):
    candidate = benchmark_dir / "Questions.csv"
    if candidate.exists():
        return candidate
    matches = sorted(benchmark_dir.rglob("Questions.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find Questions.csv under {benchmark_dir}")


def resolve_image_path(benchmark_dir: Path, index_value):
    image_dir_candidates = [
        benchmark_dir / "MMVP Images",
        benchmark_dir / "MMVP_Images",
        benchmark_dir / "images",
    ]
    stem = str(index_value).strip()
    for image_dir in image_dir_candidates:
        if not image_dir.is_dir():
            continue
        for extension in IMAGE_EXTENSIONS:
            candidate = image_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate

    matches = []
    for image_dir in image_dir_candidates:
        if image_dir.is_dir():
            matches.extend(sorted(image_dir.glob(f"{stem}.*")))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find image for MMVP index {index_value}")


def parse_options(raw_options):
    matches = OPTION_RE.findall(str(raw_options))
    option_chars = []
    options = []
    for option_char, option_text in matches:
        option_chars.append(option_char.upper())
        options.append(" ".join(str(option_text).strip().split()))
    if len(options) < 2:
        raise ValueError(f"Failed to parse MMVP options: {raw_options!r}")
    return options, option_chars


def normalize_correct_answer(value):
    text = str(value).strip().lower()
    if text in {"(a)", "a"}:
        return "A"
    if text in {"(b)", "b"}:
        return "B"
    raise ValueError(f"Unsupported MMVP correct answer value: {value!r}")


def build_question_text(question, option_chars, options, add_instruction=True):
    lines = [str(question).strip()]
    for option_char, option_text in zip(option_chars, options):
        lines.append(f"{option_char}. {option_text}")
    if add_instruction:
        lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert MMVP to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/MMVP",
        help="Path to local MMVP root folder.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="MMVP/MMVP",
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
        "--download-dir",
        type=str,
        default=None,
        help="Where to download --hf-dataset snapshot. Defaults to --benchmark-dir.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the letter-only answer instruction to each prompt.",
    )
    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args)
    questions_file = resolve_questions_file(benchmark_dir)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(questions_file, "r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows found in {questions_file}")

    with open(output_file, "w") as f:
        for row in rows:
            index_value = str(row["Index"]).strip()
            image_path = resolve_image_path(benchmark_dir, index_value)
            rel_image_path = image_path.relative_to(benchmark_dir).as_posix()
            options, option_chars = parse_options(row["Options"])
            question_id = index_value
            pair_id = ((int(index_value) - 1) // 2) + 1

            sample = {
                "question_id": question_id,
                "pair_id": str(pair_id),
                "image": rel_image_path,
                "text": build_question_text(
                    row["Question"],
                    option_chars,
                    options,
                    add_instruction=not args.no_answer_instruction,
                ),
                "label": normalize_correct_answer(row["Correct Answer"]),
                "question": str(row["Question"]).strip(),
                "options": options,
                "option_chars": option_chars,
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Questions file: {questions_file}")
    print(f"Saved {len(rows)} questions to {output_file}")


if __name__ == "__main__":
    main()
