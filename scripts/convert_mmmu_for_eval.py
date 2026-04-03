import argparse
import ast
import json
import os
import re
import string
from io import BytesIO
from pathlib import Path

import datasets as hf_datasets
from PIL import Image


DOMAIN_TO_SUBJECTS = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

ALL_SUBJECTS = [subject for subjects in DOMAIN_TO_SUBJECTS.values() for subject in subjects]
DOMAIN_BY_SUBJECT = {
    subject: domain for domain, subjects in DOMAIN_TO_SUBJECTS.items() for subject in subjects
}
SUBJECT_ALIASES = {
    "management": "Manage",
    "engineering": "Energy_and_Power",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MMMU Hugging Face data to LLaVA eval jsonl format.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/MMMU",
        help="Directory for extracted images and generated jsonl files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="MMMU/MMMU",
        help="Hugging Face dataset repo id.",
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
        help="Optional Hugging Face cache dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to load for every MMMU subject.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional list of MMMU subject configs. Defaults to all subjects.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help="Do not append the standard answer instruction to each prompt.",
    )
    return parser.parse_args()


def set_hf_cache(cache_dir: Path):
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    hf_datasets.config.HF_DATASETS_CACHE = str(cache_dir)


def normalize_subjects(subjects):
    if not subjects:
        return list(ALL_SUBJECTS)

    normalized = []
    valid_lookup = {subject.lower(): subject for subject in ALL_SUBJECTS}
    for raw_subject in subjects:
        if raw_subject.lower() == "all":
            return list(ALL_SUBJECTS)
        subject = valid_lookup.get(raw_subject.lower()) or SUBJECT_ALIASES.get(raw_subject.lower())
        if subject is None:
            raise ValueError(
                f"Unsupported MMMU subject: {raw_subject}. "
                f"Valid values: {', '.join(ALL_SUBJECTS)}"
            )
        normalized.append(subject)
    return normalized


def parse_options(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(option).strip() for option in value]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return [str(option).strip() for option in parsed]

    if text.lower() in {"none", "nan", "null"}:
        return []

    inline_matches = re.findall(r"\([A-Z]\)\s*([^()]+?)(?=\s*\([A-Z]\)|$)", text)
    if len(inline_matches) >= 2:
        return [match.strip() for match in inline_matches]

    return []


def normalize_question_type(sample, options):
    raw_question_type = str(sample.get("question_type", "")).strip().lower()
    if options:
        return "multiple_choice"
    if raw_question_type in {"multiple-choice", "multiple_choice", "multi_choice", "choice"}:
        return "multiple_choice"
    return "open"


def format_prompt(question, options, question_type, add_answer_instruction=True):
    prompt_lines = [str(question).strip()]

    if question_type == "multiple_choice":
        if len(options) > len(string.ascii_uppercase):
            raise ValueError(f"Too many options ({len(options)}); max supported is 26.")
        for idx, option in enumerate(options):
            prompt_lines.append(f"({string.ascii_uppercase[idx]}) {str(option).strip()}")
        if add_answer_instruction:
            prompt_lines.append("Answer with the option's letter from the given choices directly.")
    elif add_answer_instruction:
        prompt_lines.append("Answer the question using a single word or phrase.")

    return "\n".join(prompt_lines)


def normalize_text(text):
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text.lstrip("+")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_multiple_choice_answer(answer, options):
    if answer is None:
        return None

    valid_letters = string.ascii_uppercase[: len(options)]
    if not valid_letters:
        return None

    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return valid_letters[answer]
        return None

    answer_text = str(answer).strip()
    if not answer_text:
        return None

    match = re.fullmatch(r"[\(\[]?\s*([A-Z])\s*[\)\].,:-]?\s*", answer_text.upper())
    if match:
        candidate = match.group(1)
        if candidate in valid_letters:
            return candidate

    if answer_text.isdigit():
        numeric_answer = int(answer_text)
        if 0 <= numeric_answer < len(options):
            return valid_letters[numeric_answer]
        if 1 <= numeric_answer <= len(options):
            return valid_letters[numeric_answer - 1]

    normalized_answer = normalize_text(answer_text)
    for idx, option in enumerate(options):
        if normalize_text(option) == normalized_answer:
            return valid_letters[idx]

    return None


def normalize_label(answer, question_type, options):
    if answer is None:
        return None
    if question_type == "multiple_choice":
        return normalize_multiple_choice_answer(answer, options)

    answer_text = str(answer).strip()
    return answer_text or None


def to_pil_image(value):
    if value is None:
        return None
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


def image_keys(sample):
    keys = []
    for key in sample.keys():
        match = re.fullmatch(r"image_(\d+)", str(key))
        if match:
            keys.append((int(match.group(1)), key))
    return [key for _, key in sorted(keys)]


def safe_path_component(value):
    text = str(value).strip() or "sample"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def make_question_id(subject, sample, index, seen_ids):
    candidates = [sample.get("id"), sample.get("question_id"), sample.get("uuid")]
    for candidate in candidates:
        if candidate is None:
            continue
        qid = f"{subject}/{candidate}"
        if qid not in seen_ids:
            seen_ids.add(qid)
            return qid

    qid = f"{subject}/{index}"
    while qid in seen_ids:
        qid = f"{qid}_dup"
    seen_ids.add(qid)
    return qid


def save_images(sample, benchmark_dir, subject, sample_id):
    rel_paths = []
    for key in image_keys(sample):
        image = to_pil_image(sample.get(key))
        if image is None:
            continue

        rel_path = Path("images") / subject / sample_id / f"{key}.png"
        abs_path = benchmark_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(abs_path, format="PNG")
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def main():
    args = parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else benchmark_dir / ".hf_cache"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_hf_cache(cache_dir)

    subjects = normalize_subjects(args.subjects)
    seen_ids = set()
    total_examples = 0
    per_subject_counts = {}

    with open(output_file, "w") as f:
        for subject in subjects:
            dataset = hf_datasets.load_dataset(
                args.hf_dataset,
                subject,
                split=args.split,
                revision=args.hf_revision,
                token=args.hf_token,
                cache_dir=str(cache_dir),
            )

            subject_count = 0
            domain = DOMAIN_BY_SUBJECT.get(subject, "unknown")
            for index, sample in enumerate(dataset):
                options = parse_options(sample.get("options"))
                question_type = normalize_question_type(sample, options)
                label = normalize_label(sample.get("answer"), question_type, options)
                question_id = make_question_id(subject, sample, index, seen_ids)
                sample_id = safe_path_component(question_id.split("/", 1)[1])
                images = save_images(sample, benchmark_dir=benchmark_dir, subject=subject, sample_id=sample_id)

                row = {
                    "question_id": question_id,
                    "image": images[0] if images else None,
                    "images": images,
                    "text": format_prompt(
                        sample.get("question", ""),
                        options,
                        question_type=question_type,
                        add_answer_instruction=not args.no_answer_instruction,
                    ),
                    "label": label,
                    "question_type": question_type,
                    "subject": subject,
                    "domain": domain,
                    "options": options,
                    "option_chars": list(string.ascii_uppercase[: len(options)]),
                    "source_id": sample.get("id"),
                    "source_split": args.split,
                }
                f.write(json.dumps(row) + "\n")
                subject_count += 1
                total_examples += 1

            per_subject_counts[subject] = subject_count

    print(f"Saved {total_examples} MMMU samples to {output_file}")
    print(f"Extracted images under {benchmark_dir / 'images'}")
    for subject, count in sorted(per_subject_counts.items()):
        print(f"{subject}: {count}")


if __name__ == "__main__":
    main()