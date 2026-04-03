import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path


def load_json_or_jsonl(path: Path):
    if path.suffix == ".json":
        data = json.load(open(path, "r"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list in {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def normalize_text(text):
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text.lstrip("+")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_yes_no(text):
    if text is None:
        return None
    text = str(text)
    if "." in text:
        text = text.split(".", 1)[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    if "Yes" in words or "yes" in words:
        return "yes"
    return None


def strip_answer_prefix(text):
    prefixes = [
        "answer:",
        "answer is",
        "the answer is",
        "it is",
        "it's",
    ]
    normalized = str(text).strip().lower()
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def extract_option_letter(pred_text, valid_letters, options=None):
    if pred_text is None:
        return None

    text = str(pred_text).strip()
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
        normalized_pred = normalize_text(strip_answer_prefix(text))
        option_map = {}
        for idx, option in enumerate(options):
            letter = valid_letters[idx]
            option_map[normalize_text(option)] = letter
        if normalized_pred in option_map:
            return option_map[normalized_pred]

    return None


def evaluate_sample(sample, pred_text):
    gold = str(sample.get("label", "")).strip()
    answer_type = sample.get("answer_type", "open")

    if pred_text is None:
        return False, None, 1, 0

    if answer_type == "multiple_choice":
        option_chars = sample.get("option_chars") or list(string.ascii_uppercase[: max(len(sample.get("options", [])), 1)])
        pred_norm = extract_option_letter(pred_text, option_chars, options=sample.get("options"))
        is_correct = pred_norm == gold.upper()
        return is_correct, pred_norm, 0, int(pred_norm is None)

    gold_norm = normalize_text(gold)
    if gold_norm in {"yes", "no"}:
        pred_norm = normalize_yes_no(pred_text)
        is_correct = pred_norm == gold_norm
        return is_correct, pred_norm, 0, int(pred_norm is None)

    pred_norm = normalize_text(strip_answer_prefix(pred_text))
    is_correct = pred_norm == gold_norm
    return is_correct, pred_norm, 0, int(pred_norm == "")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RealWorldQA predictions.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(row.get("question_id")): row.get("text", "") for row in results}

    total = 0
    correct = 0
    missing = 0
    unparsable = 0
    by_answer_type = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        pred_text = pred_by_qid.get(qid)
        is_correct, pred_norm, sample_missing, sample_unparsable = evaluate_sample(sample, pred_text)

        total += 1
        correct += int(is_correct)
        missing += sample_missing
        unparsable += sample_unparsable
        answer_type = sample.get("answer_type", "unknown")
        by_answer_type[answer_type]["total"] += 1
        by_answer_type[answer_type]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "answer_type": answer_type,
                    "gold": sample.get("label"),
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": sample_missing,
                    "unparsable": sample_unparsable,
                }
            )

    summary = {
        "dataset": "realworldqa",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_answer_type": {
            answer_type: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": safe_div(stats["correct"], stats["total"]),
            }
            for answer_type, stats in sorted(by_answer_type.items())
        },
    }

    print(f"Overall: {correct}/{total} = {safe_div(correct, total):.4f}")
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")
    for answer_type, stats in summary["by_answer_type"].items():
        print(f"{answer_type}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {output_path}")

    if args.detail_file:
        detail_path = Path(args.detail_file).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            for row in details:
                f.write(json.dumps(row) + "\n")
        print(f"Saved details to {detail_path}")


if __name__ == "__main__":
    main()