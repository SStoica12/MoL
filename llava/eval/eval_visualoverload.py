import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path


NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:answer\s*[:\-]?|answer\s+is|the\s+answer\s+is|it\s+is|it's)\s*",
    flags=re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def load_json_or_jsonl(path: Path):
    if path.suffix == ".json":
        data = json.load(open(path, "r"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list in {path}")

    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def strip_answer_prefix(text):
    return ANSWER_PREFIX_RE.sub("", str(text).strip(), count=1)


def compact_text(text):
    return re.sub(r"\s+", " ", str(text).strip()).strip()


def normalize_text(text):
    text = compact_text(strip_answer_prefix(text))
    return text.strip().strip("\"'`")


def normalize_text_for_match(text):
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_option_letter(pred_text, valid_letters, options=None):
    if pred_text is None:
        return None

    text = str(pred_text).strip()
    if not text:
        return None

    upper_text = text.upper()
    valid_set = set(valid_letters)
    patterns = [
        r"^\s*[\(\[]?\s*([A-Z])\s*[\)\].,:-]?\s*$",
        r"^\s*[\(\[]?\s*([A-Z])\s*[\)\].,:-]\s*",
        r"\b(?:OPTION|ANSWER|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b",
        r"\b([A-Z])\s+(?:IS\s+CORRECT|IS\s+THE\s+ANSWER)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper_text)
        if match:
            candidate = match.group(1)
            if candidate in valid_set:
                return candidate

    if options:
        normalized_pred = normalize_text_for_match(text)
        for letter, option in zip(valid_letters, options):
            if normalized_pred == normalize_text_for_match(option):
                return letter

    return None


def format_number_string(value):
    if float(value).is_integer():
        return str(int(round(float(value))))
    formatted = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return formatted or "0"


def extract_counting_response(pred_text):
    if pred_text is None:
        return None

    text = normalize_text(pred_text)
    if not text:
        return None

    matches = NUMERIC_RE.findall(text.replace(",", ""))
    if matches:
        try:
            return format_number_string(float(matches[0]))
        except ValueError:
            pass

    lowered = text.lower()
    if lowered in NUMBER_WORDS:
        return str(NUMBER_WORDS[lowered])
    return None


def normalize_ocr_response(pred_text):
    if pred_text is None:
        return None
    text = normalize_text(pred_text)
    return text or None


def fallback_submission_response(pred_text):
    if pred_text is None:
        return ""
    return normalize_text(pred_text)


def extract_prediction_text(row):
    for key in ("text", "response", "prediction", "pred"):
        value = row.get(key)
        if value is not None:
            return value
    return None


def init_bucket():
    return {
        "total": 0,
        "predicted": 0,
        "nonempty": 0,
        "normalized": 0,
        "missing": 0,
        "empty": 0,
        "invalid_format": 0,
    }


def finalize_bucket(stats):
    return {
        "total": stats["total"],
        "predicted": stats["predicted"],
        "nonempty": stats["nonempty"],
        "normalized": stats["normalized"],
        "missing": stats["missing"],
        "empty": stats["empty"],
        "invalid_format": stats["invalid_format"],
        "prediction_rate": safe_div(stats["predicted"], stats["total"]),
        "nonempty_rate": safe_div(stats["nonempty"], stats["total"]),
        "submission_ready_rate": safe_div(stats["normalized"], stats["total"]),
    }


def update_bucket(stats, pred_text, submission_response, normalized_response, invalid_format):
    stats["total"] += 1
    if pred_text is None:
        stats["missing"] += 1
        return

    stats["predicted"] += 1
    if str(pred_text).strip() == "":
        stats["empty"] += 1
    if submission_response:
        stats["nonempty"] += 1
    if normalized_response is not None:
        stats["normalized"] += 1
    if invalid_format:
        stats["invalid_format"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Normalize VisualOverload predictions into the official submission format."
    )
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    parser.add_argument("--submission-file", type=str, default=None)
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {}
    for row in results:
        question_id = row.get("question_id")
        if question_id is None:
            continue
        pred_by_qid[str(question_id)] = extract_prediction_text(row)

    overall_stats = init_bucket()
    by_question_type = defaultdict(init_bucket)
    by_category = defaultdict(init_bucket)
    by_difficulty = defaultdict(init_bucket)
    submission_rows = []
    details = []

    for sample in questions:
        question_id = str(sample["question_id"])
        pred_text = pred_by_qid.get(question_id)
        question_type = str(sample.get("question_type", "unknown")).strip().lower() or "unknown"
        category = str(sample.get("category", "unknown")).strip().lower() or "unknown"
        difficulty = str(sample.get("difficulty", "unknown")).strip().lower() or "unknown"
        options = sample.get("options") or []
        option_chars = sample.get("option_chars") or list(string.ascii_uppercase[: len(options)])

        normalized_response = None
        invalid_format = False
        if question_type == "choice":
            normalized_response = extract_option_letter(pred_text, option_chars, options=options)
            invalid_format = pred_text is not None and str(pred_text).strip() != "" and normalized_response is None
        elif question_type == "counting":
            normalized_response = extract_counting_response(pred_text)
        else:
            normalized_response = normalize_ocr_response(pred_text)

        submission_response = normalized_response
        if submission_response is None:
            submission_response = fallback_submission_response(pred_text)

        update_bucket(overall_stats, pred_text, submission_response, normalized_response, invalid_format)
        update_bucket(by_question_type[question_type], pred_text, submission_response, normalized_response, invalid_format)
        update_bucket(by_category[category], pred_text, submission_response, normalized_response, invalid_format)
        update_bucket(by_difficulty[difficulty], pred_text, submission_response, normalized_response, invalid_format)

        submission_rows.append({"question_id": question_id, "response": submission_response})

        if args.detail_file:
            details.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "category": category,
                    "difficulty": difficulty,
                    "pred_text": pred_text,
                    "normalized_response": normalized_response,
                    "submission_response": submission_response,
                    "submission_ready": int(normalized_response is not None),
                    "missing": int(pred_text is None),
                    "empty": int(pred_text is not None and str(pred_text).strip() == ""),
                    "invalid_format": int(invalid_format),
                }
            )

    summary = {
        "dataset": "visualoverload",
        "metric": "external_eval_required",
        "overall": {
            "total": overall_stats["total"],
            "predicted": overall_stats["predicted"],
            "nonempty": overall_stats["nonempty"],
            "normalized": overall_stats["normalized"],
            "accuracy": None,
            "prediction_rate": safe_div(overall_stats["predicted"], overall_stats["total"]),
            "nonempty_rate": safe_div(overall_stats["nonempty"], overall_stats["total"]),
            "submission_ready_rate": safe_div(overall_stats["normalized"], overall_stats["total"]),
        },
        "missing_predictions": overall_stats["missing"],
        "empty_predictions": overall_stats["empty"],
        "invalid_format_predictions": overall_stats["invalid_format"],
        "external_eval_required": True,
        "note": (
            "VisualOverload keeps ground-truth labels private. "
            "Upload submission.json to the official evaluation server to obtain scores."
        ),
        "by_question_type": {
            key: finalize_bucket(stats)
            for key, stats in sorted(by_question_type.items())
        },
        "by_category": {
            key: finalize_bucket(stats)
            for key, stats in sorted(by_category.items())
        },
        "by_difficulty": {
            key: finalize_bucket(stats)
            for key, stats in sorted(by_difficulty.items())
        },
    }

    if args.submission_file:
        submission_path = Path(args.submission_file).expanduser().resolve()
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        with open(submission_path, "w") as handle:
            json.dump(submission_rows, handle, indent=2)
        summary["submission_file"] = str(submission_path)
        print(f"Saved submission file to {submission_path}")

    print(f"Predictions present: {overall_stats['predicted']}/{overall_stats['total']}")
    print(f"Submission-ready responses: {overall_stats['normalized']}/{overall_stats['total']}")
    print(f"Missing predictions: {overall_stats['missing']}")
    print(f"Empty predictions: {overall_stats['empty']}")
    print(f"Invalid-format predictions: {overall_stats['invalid_format']}")

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary to {output_path}")

    if args.detail_file:
        detail_path = Path(args.detail_file).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as handle:
            for row in details:
                handle.write(json.dumps(row) + "\n")
        print(f"Saved details to {detail_path}")


if __name__ == "__main__":
    main()
