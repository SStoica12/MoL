import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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


def strip_answer_prefix(text):
    lowered = str(text).strip().lower()
    prefixes = [
        "answer:",
        "answer is",
        "the answer is",
        "it is",
        "it's",
        "the chart shows",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return lowered


def normalize_text_answer(text):
    text = strip_answer_prefix(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".,!?;:")
    return text


def parse_gold_numeric(text):
    candidate = str(text).strip().replace(",", "")
    candidate = candidate.rstrip("%")
    if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate):
        return float(candidate)
    return None


def parse_pred_numeric(text):
    candidate = normalize_text_answer(text).replace(",", "")
    candidate = candidate.replace("%", " ")
    matches = NUMERIC_RE.findall(candidate)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def relaxed_correctness(gold, pred):
    gold_num = parse_gold_numeric(gold)
    if gold_num is not None:
        pred_num = parse_pred_numeric(pred)
        if pred_num is None:
            return False, "numeric", None
        if abs(gold_num) < 1e-12:
            return abs(pred_num - gold_num) < 1e-12, "numeric", pred_num
        return abs(pred_num - gold_num) / abs(gold_num) <= 0.05, "numeric", pred_num

    pred_norm = normalize_text_answer(pred)
    gold_norm = normalize_text_answer(gold)
    return pred_norm == gold_norm, "text", pred_norm


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChartQA predictions with relaxed accuracy.")
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
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_answer_type = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        sample_type = sample.get("type", "unknown")
        pred_text = pred_by_qid.get(qid)

        if pred_text is None:
            is_correct = False
            answer_type = "unknown"
            pred_norm = None
            missing += 1
        else:
            is_correct, answer_type, pred_norm = relaxed_correctness(sample["label"], pred_text)
            if answer_type == "numeric" and pred_norm is None:
                unparsable += 1

        total += 1
        correct += int(is_correct)
        by_type[sample_type]["total"] += 1
        by_type[sample_type]["correct"] += int(is_correct)
        by_answer_type[answer_type]["total"] += 1
        by_answer_type[answer_type]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "type": sample_type,
                    "gold": sample["label"],
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "answer_type": answer_type,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                }
            )

    summary = {
        "dataset": "chartqa",
        "metric": "relaxed_accuracy",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_type": {
            sample_type: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": safe_div(stats["correct"], stats["total"]),
            }
            for sample_type, stats in sorted(by_type.items())
        },
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
    for sample_type, stats in summary["by_type"].items():
        print(f"{sample_type}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

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