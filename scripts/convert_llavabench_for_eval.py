import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_FILES = [
    "questions.jsonl",
    "context.jsonl",
    "answers_gpt4.jsonl",
]


def count_jsonl_rows(path: Path):
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def has_required_files(benchmark_dir: Path):
    return all((benchmark_dir / rel_path).exists() for rel_path in REQUIRED_FILES)


def ensure_dataset(args, benchmark_dir: Path):
    if has_required_files(benchmark_dir) and (benchmark_dir / "images").is_dir():
        return

    if not args.hf_dataset:
        missing = [rel_path for rel_path in REQUIRED_FILES if not (benchmark_dir / rel_path).exists()]
        raise FileNotFoundError(
            f"LLaVA-Bench files are missing under {benchmark_dir}: {missing}. "
            "Pass --hf-dataset to download them automatically or populate the directory manually."
        )

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    snapshot_download(
        repo_id=args.hf_dataset,
        repo_type="dataset",
        revision=args.hf_revision,
        token=args.hf_token,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(benchmark_dir),
        local_dir_use_symlinks=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare LLaVA-Bench-in-the-Wild assets for evaluation.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/llava-bench-in-the-wild",
        help="Directory containing the extracted LLaVA-Bench-in-the-Wild files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="liuhaotian/llava-bench-in-the-wild",
        help="Optional Hugging Face dataset repo id to download when files are missing.",
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
        help="Optional Hugging Face cache directory.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(args, benchmark_dir)

    missing = [rel_path for rel_path in REQUIRED_FILES if not (benchmark_dir / rel_path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required LLaVA-Bench files after preparation: {missing}")

    image_dir = benchmark_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    image_count = len([path for path in image_dir.rglob("*") if path.is_file()])
    question_count = count_jsonl_rows(benchmark_dir / "questions.jsonl")
    context_count = count_jsonl_rows(benchmark_dir / "context.jsonl")
    reference_count = count_jsonl_rows(benchmark_dir / "answers_gpt4.jsonl")

    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Questions: {question_count}")
    print(f"Contexts: {context_count}")
    print(f"Reference answers: {reference_count}")
    print(f"Image files: {image_count}")


if __name__ == "__main__":
    main()
