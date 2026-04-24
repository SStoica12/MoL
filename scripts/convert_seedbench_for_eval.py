import argparse
import json
import string
from pathlib import Path


IMAGE_EXTENSIONS = (
    "",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
    ".BMP",
    ".GIF",
)


def normalize_text(text):
    return " ".join(str(text).strip().split())


def build_prompt(question_data, add_instruction=True):
    lines = [normalize_text(question_data.get("question", ""))]
    options = []
    for option_char, option_key in zip(string.ascii_uppercase, ("choice_a", "choice_b", "choice_c", "choice_d")):
        option_value = normalize_text(question_data.get(option_key, ""))
        options.append(option_value)
        lines.append(f"{option_char}. {option_value}")
    if add_instruction:
        lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines), options


def resolve_relative_path(seed_root: Path, candidate: Path):
    try:
        return candidate.resolve().relative_to(seed_root.resolve()).as_posix()
    except Exception:
        return str(candidate.resolve())


def resolve_image_path(seed_root: Path, image_dir: Path, data_id: str):
    base_candidate = image_dir / str(data_id)
    for suffix in IMAGE_EXTENSIONS:
        candidate = base_candidate if suffix == "" else base_candidate.with_suffix(suffix)
        if candidate.exists():
            return resolve_relative_path(seed_root, candidate)
    return None


def resolve_video_frame_path(seed_root: Path, video_frame_dir: Path, question_type_id: str, question_id: str):
    candidate = video_frame_dir / f"{question_type_id}_{question_id}.png"
    if candidate.exists():
        return resolve_relative_path(seed_root, candidate)
    return None


def resolve_annotation_path(value: str | None, seed_root: Path):
    if value:
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Annotation file not found: {path}")

    candidates = [
        seed_root / "SEED-Bench.json",
        seed_root / "SEED-Bench_v1_level1_2_3.json",
        seed_root / "SEED-Bench_v2_level1_2_3.json",
        seed_root / "SEED-Bench-2" / "SEED-Bench_v2_level1_2_3.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a SEED-Bench annotation file. "
        "Pass --annotation-file or place one under playground/data/eval/seed_bench."
    )


def main():
    parser = argparse.ArgumentParser(description="Convert SEED-Bench annotations to LLaVA eval jsonl format.")
    parser.add_argument(
        "--seed-root",
        type=str,
        default="./playground/data/eval/seed_bench",
        help="Root SEED-Bench directory.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Official SEED-Bench annotation json.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing SEED-Bench image assets.",
    )
    parser.add_argument(
        "--video-frame-dir",
        type=str,
        default=None,
        help="Directory containing extracted middle frames for SEED-Bench video questions.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./playground/data/eval/seed_bench/llava-seed-bench.jsonl",
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Keep rows even when the referenced image asset is missing.",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the answer-format instruction to each prompt.",
    )
    args = parser.parse_args()

    seed_root = Path(args.seed_root).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    temp_output_file = output_file.with_name(f".{output_file.name}.tmp")
    annotation_file = resolve_annotation_path(args.annotation_file, seed_root)
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else (seed_root / "SEED-Bench-image").resolve()
    video_frame_dir = (
        Path(args.video_frame_dir).expanduser().resolve()
        if args.video_frame_dir
        else (seed_root / "SEED-Bench-video-image").resolve()
    )

    seed_root.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotation = json.load(open(annotation_file, "r"))
    questions = annotation.get("questions", [])
    question_type_map = {str(v): k for k, v in annotation.get("question_type", {}).items()}

    created = 0
    skipped = 0
    skipped_missing = 0

    if temp_output_file.exists():
        temp_output_file.unlink()

    try:
        with open(temp_output_file, "w") as handle:
            for question_data in questions:
                answer = question_data.get("answer")
                if answer is None:
                    skipped += 1
                    continue

                question_id = str(question_data.get("question_id"))
                question_type_id = str(question_data.get("question_type_id", "unknown"))
                data_type = str(question_data.get("data_type", "unknown"))
                if data_type == "image":
                    image_path = resolve_image_path(seed_root, image_dir, str(question_data.get("data_id", "")))
                elif data_type == "video":
                    image_path = resolve_video_frame_path(seed_root, video_frame_dir, question_type_id, question_id)
                else:
                    image_path = None

                if image_path is None and not args.allow_missing_assets:
                    skipped += 1
                    skipped_missing += 1
                    continue

                prompt, options = build_prompt(
                    question_data,
                    add_instruction=not args.no_answer_instruction,
                )
                row = {
                    "question_id": question_data.get("question_id"),
                    "image": image_path,
                    "text": prompt,
                    "label": str(answer).strip(),
                    "category": question_type_map.get(question_type_id, question_type_id),
                    "data_type": data_type,
                    "question_type_id": question_data.get("question_type_id"),
                    "data_id": question_data.get("data_id"),
                    "options": options,
                    "option_chars": list(string.ascii_uppercase[: len(options)]),
                }
                handle.write(json.dumps(row) + "\n")
                created += 1
    except Exception:
        if temp_output_file.exists():
            temp_output_file.unlink()
        raise

    if created == 0:
        temp_output_file.unlink(missing_ok=True)
        raise RuntimeError(
            "SEED-Bench conversion produced zero questions. "
            "Check the annotation file and asset directories."
        )

    temp_output_file.replace(output_file)
    print(f"Saved {created} SEED-Bench questions to {output_file}")
    print(f"Skipped {skipped} questions total")
    print(f"Skipped {skipped_missing} questions due to missing assets")


if __name__ == "__main__":
    main()
