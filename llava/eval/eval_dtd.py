import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_json_or_jsonl(path: Path):
    if path.suffix == ".json":
        data = json.load(open(path, "r"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected JSON list in {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(text: str):
    text = str(text).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_label(pred_text, labels):
    if pred_text is None:
        return None

    norm_pred = normalize_label(pred_text)
    if not norm_pred:
        return None

    norm_to_label = {normalize_label(label): label for label in labels}

    # Exact normalized match.
    if norm_pred in norm_to_label:
        return norm_to_label[norm_pred]

    # Try common short forms like "Answer: woven".
    m = re.search(r"(?:answer|class|label)\s*[:\-]?\s*(.+)$", norm_pred)
    if m:
        candidate = m.group(1).strip()
        if candidate in norm_to_label:
            return norm_to_label[candidate]

    # Label appears as a phrase in prediction; prefer longest match.
    matches = []
    for norm_label, original_label in norm_to_label.items():
        pattern = r"\b" + re.escape(norm_label) + r"\b"
        if re.search(pattern, norm_pred):
            matches.append((len(norm_label), original_label))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]

    return None


def safe_div(a, b):
    return 0.0 if b == 0 else float(a) / float(b)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DTD classification predictions.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())

    pred_by_qid = {str(r.get("question_id")): r.get("text", "") for r in results}

    labels = sorted({q["label"] for q in questions if "label" in q})
    if not labels:
        raise ValueError("No labels found in question file.")

    total = 0
    correct = 0
    missing = 0
    unparsable = 0

    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    by_split = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for q in questions:
        qid = str(q["question_id"])
        gold = q["label"]
        split_name = f"{q.get('split', 'unknown')}{q.get('split_id', '')}"

        pred_text = pred_by_qid.get(qid, "")
        if qid not in pred_by_qid:
            missing += 1

        pred_label = extract_label(pred_text, labels)
        if pred_label is None:
            unparsable += 1

        is_correct = pred_label == gold
        total += 1
        correct += int(is_correct)

        by_class[gold]["total"] += 1
        by_class[gold]["correct"] += int(is_correct)

        by_split[split_name]["total"] += 1
        by_split[split_name]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "gold": gold,
                    "pred_label": pred_label,
                    "pred_text": pred_text,
                    "correct": int(is_correct),
                    "split": split_name,
                }
            )

    summary = {
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_split": {
            split: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": safe_div(s["correct"], s["total"]),
            }
            for split, s in sorted(by_split.items())
        },
        "by_class": {
            cls: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": safe_div(s["correct"], s["total"]),
            }
            for cls, s in sorted(by_class.items())
        },
    }

    print(
        f"Overall: {summary['overall']['correct']}/{summary['overall']['total']} = "
        f"{summary['overall']['accuracy']:.4f}"
    )
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")
    for split, s in summary["by_split"].items():
        print(f"{split}: {s['correct']}/{s['total']} = {s['accuracy']:.4f}")

    if args.output_file:
        output_file = Path(args.output_file).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {output_file}")

    if args.detail_file:
        detail_file = Path(args.detail_file).expanduser().resolve()
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_file, "w") as f:
            for row in details:
                f.write(json.dumps(row) + "\n")
        print(f"Saved details to {detail_file}")


if __name__ == "__main__":
    main()
