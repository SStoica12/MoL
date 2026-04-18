import argparse
import json
from pathlib import Path


def normalize_image_path(sample, benchmark_dir: Path, strict: bool = False):
    visual_input = str(sample.get("visual_input", "")).strip()
    if visual_input == "0":
        return None

    filename = sample.get("filename")
    candidates = []

    if filename:
        raw = str(filename).strip()
        while raw.startswith("./"):
            raw = raw[2:]
        if raw:
            candidates.append(raw)
            if not raw.startswith("hallusion_bench/"):
                candidates.append(f"hallusion_bench/{raw}")

    derived = (
        f"hallusion_bench/{sample['category']}/{sample['subcategory']}/"
        f"{sample['set_id']}_{sample['figure_id']}.png"
    )
    candidates.append(derived)

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate).as_posix()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (benchmark_dir / candidate).exists():
            return candidate

    if strict:
        raise FileNotFoundError(f"Could not resolve image for sample: {sample}")
    return candidates[0] if candidates else None


def build_question_text(question: str, add_instruction: bool = True):
    question = str(question).strip()
    if not add_instruction:
        return question
    return f"{question}\nAnswer with Yes, No, or Uncertain only."


def make_question_uid(sample):
    return "_".join(
        [
            str(sample["category"]),
            str(sample["subcategory"]),
            str(sample["set_id"]),
            str(sample["figure_id"]),
            str(sample["question_id"]),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Convert HallusionBench to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/HallusionBench",
        help="Path to the HallusionBench root directory.",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default=None,
        help="Optional path to HallusionBench.json. Defaults to <benchmark-dir>/HallusionBench.json.",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file.")
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the Yes/No/Uncertain instruction to each prompt.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing image files instead of keeping the unresolved relative path.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    raw_file = Path(args.raw_file).expanduser().resolve() if args.raw_file else benchmark_dir / "HallusionBench.json"
    output_file = Path(args.output_file).expanduser().resolve()

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw HallusionBench file not found: {raw_file}")

    rows = json.load(open(raw_file, "r"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {raw_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    num_text_only = 0
    num_with_image = 0

    with open(output_file, "w") as f:
        for sample in rows:
            image_path = normalize_image_path(sample, benchmark_dir=benchmark_dir, strict=args.strict)
            if image_path is None:
                num_text_only += 1
            else:
                num_with_image += 1

            row = {
                "question_id": make_question_uid(sample),
                "raw_question_id": str(sample["question_id"]),
                "image": image_path,
                "text": build_question_text(
                    sample["question"],
                    add_instruction=not args.no_answer_instruction,
                ),
                "label": str(sample["gt_answer"]),
                "category": str(sample["category"]),
                "subcategory": str(sample["subcategory"]),
                "visual_input": str(sample["visual_input"]),
                "set_id": str(sample["set_id"]),
                "figure_id": str(sample["figure_id"]),
                "sample_note": sample.get("sample_note"),
                "question": sample["question"],
                "gt_answer_details": sample.get("gt_answer_details"),
                "filename": sample.get("filename"),
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(rows)} samples to {output_file}")
    print(f"Samples with images: {num_with_image}")
    print(f"Text-only samples: {num_text_only}")


if __name__ == "__main__":
    main()
