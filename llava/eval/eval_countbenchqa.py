import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
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
    prefixes = [
        "answer:",
        "answer is",
        "the answer is",
        "it is",
        "it's",
        "there are",
        "there is",
    ]
    normalized = str(text).strip().lower()
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def extract_count(pred_text):
    if pred_text is None:
        return None

    text = strip_answer_prefix(pred_text)
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if matches:
        try:
            value = float(matches[0])
        except ValueError:
            value = None
        if value is not None and value.is_integer():
            return int(value)
        return None

    for token in re.findall(r"[a-z]+", text):
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate CountBenchQA predictions.")
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
    by_number = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        gold = int(sample.get("label", sample.get("number")))
        pred_text = pred_by_qid.get(qid)
        pred_norm = extract_count(pred_text)

        if pred_text is None:
            missing += 1
        elif pred_norm is None:
            unparsable += 1

        is_correct = pred_norm == gold
        total += 1
        correct += int(is_correct)
        by_number[str(gold)]["total"] += 1
        by_number[str(gold)]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                    "unparsable": int(pred_text is not None and pred_norm is None),
                }
            )

    summary = {
        "dataset": "countbenchqa",
        "metric": "count_accuracy",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_number": {
            number: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": safe_div(stats["correct"], stats["total"]),
            }
            for number, stats in sorted(by_number.items(), key=lambda item: int(item[0]))
        },
    }

    print(f"Overall: {correct}/{total} = {safe_div(correct, total):.4f}")
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")
    for number, stats in summary["by_number"].items():
        print(f"{number}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

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
