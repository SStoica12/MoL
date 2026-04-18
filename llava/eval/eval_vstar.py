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

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


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
        norm_pred = normalize_text(text)
        option_map = {}
        for idx, option in enumerate(options):
            letter = valid_letters[idx]
            option_map[normalize_text(option)] = letter
        if norm_pred in option_map:
            return option_map[norm_pred]

    return None


def safe_div(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-style predictions on V*Bench.")
    parser.add_argument("--question-file", type=str, required=True, help="Question jsonl/json file with labels.")
    parser.add_argument("--result-file", type=str, required=True, help="Model prediction jsonl file.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional summary json output path.")
    parser.add_argument(
        "--detail-file",
        type=str,
        default=None,
        help="Optional per-sample detail jsonl output path.",
    )
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    predictions = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())

    pred_by_qid = {}
    for pred in predictions:
        pred_by_qid[str(pred.get("question_id"))] = pred.get("text", "")

    total = 0
    correct = 0
    missing = 0
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample.get("question_id"))
        category = sample.get("category", "unknown")
        options = sample.get("options", [])
        valid_letters = list(string.ascii_uppercase[:max(len(options), 1)])
        gold = str(sample.get("label", "A")).strip().upper()
        if gold not in valid_letters:
            # V*Bench default: first option is correct.
            gold = "A"

        pred_text = pred_by_qid.get(qid, "")
        if qid not in pred_by_qid:
            missing += 1

        pred_letter = extract_option_letter(pred_text, valid_letters, options=options)
        is_correct = pred_letter == gold

        total += 1
        correct += int(is_correct)
        per_category[category]["total"] += 1
        per_category[category]["correct"] += int(is_correct)

        if args.detail_file:
            details.append({
                "question_id": qid,
                "category": category,
                "gold": gold,
                "pred_letter": pred_letter,
                "pred_text": pred_text,
                "correct": int(is_correct),
            })

    summary = {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": safe_div(correct, total),
        },
        "missing_predictions": missing,
        "by_category": {
            category: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": safe_div(stats["correct"], stats["total"]),
            }
            for category, stats in sorted(per_category.items())
        },
    }

    print(f"Overall: {summary['overall']['correct']}/{summary['overall']['total']} = {summary['overall']['accuracy']:.4f}")
    print(f"Missing predictions: {missing}")
    for category, stats in summary["by_category"].items():
        print(f"{category}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

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
