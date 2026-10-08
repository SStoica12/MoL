import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path


KNOWN_QUESTION_TYPES = ["yes_no", "multiple_choice"]
KNOWN_SOURCES = ["DOCCI", "Flickr", "XM3600"]


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
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        normalized_pred = normalize_text(strip_answer_prefix(text))
        option_map = {}
        for idx, option in enumerate(options):
            option_map[normalize_text(option)] = valid_letters[idx]
        if normalized_pred in option_map:
            return option_map[normalized_pred]

    return None


def extract_yes_no(pred_text):
    if pred_text is None:
        return None
    text = strip_answer_prefix(pred_text)
    normalized = normalize_text(text)
    if normalized in {"yes", "yeah", "y", "true"}:
        return "YES"
    if normalized in {"no", "n", "false"}:
        return "NO"

    upper_text = str(pred_text).strip().upper()
    if re.search(r"\bYES\b", upper_text):
        return "YES"
    if re.search(r"\bNO\b", upper_text):
        return "NO"
    return None


def ordered_items(stats, preferred_order):
    seen = set()
    for key in preferred_order:
        if key in stats:
            seen.add(key)
            yield key, stats[key]
    for key in sorted(stats):
        if key in seen:
            continue
        yield key, stats[key]


def bucket_summary(correct, total):
    return {"correct": correct, "total": total, "accuracy": safe_div(correct, total)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate NaturalBench predictions.")
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
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    by_question_group = defaultdict(lambda: {"correct": 0, "total": 0})
    by_image_group = defaultdict(lambda: {"correct": 0, "total": 0})
    by_sample_group = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        pred_text = pred_by_qid.get(qid)
        if pred_text is None:
            missing += 1

        question_type = str(sample.get("question_type", "yes_no")).strip() or "yes_no"
        options = sample.get("options") or []
        gold = str(sample.get("label", "")).strip().upper()

        if question_type == "multiple_choice":
            option_chars = sample.get("option_chars") or list(string.ascii_uppercase[: max(len(options), 1)])
            pred_norm = extract_option_letter(pred_text, option_chars, options=options)
        else:
            pred_norm = extract_yes_no(pred_text)

        if pred_text is not None and pred_norm is None:
            unparsable += 1

        is_correct = pred_norm == gold
        total += 1
        correct += int(is_correct)

        source = str(sample.get("source", "unknown")).strip() or "unknown"
        by_question_type[question_type]["total"] += 1
        by_question_type[question_type]["correct"] += int(is_correct)
        by_source[source]["total"] += 1
        by_source[source]["correct"] += int(is_correct)

        question_group_id = str(sample.get("question_group_id", qid))
        image_group_id = str(sample.get("image_group_id", qid))
        sample_id = str(sample.get("sample_id", qid))
        by_question_group[question_group_id]["total"] += 1
        by_question_group[question_group_id]["correct"] += int(is_correct)
        by_image_group[image_group_id]["total"] += 1
        by_image_group[image_group_id]["correct"] += int(is_correct)
        by_sample_group[sample_id]["total"] += 1
        by_sample_group[sample_id]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "sample_id": sample_id,
                    "question_group_id": question_group_id,
                    "image_group_id": image_group_id,
                    "question_type": question_type,
                    "source": source,
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                    "unparsable": int(pred_text is not None and pred_norm is None),
                }
            )

    question_group_correct = sum(
        1 for stats in by_question_group.values() if stats["total"] > 0 and stats["correct"] == stats["total"]
    )
    image_group_correct = sum(
        1 for stats in by_image_group.values() if stats["total"] > 0 and stats["correct"] == stats["total"]
    )
    sample_group_correct = sum(
        1 for stats in by_sample_group.values() if stats["total"] > 0 and stats["correct"] == stats["total"]
    )

    summary = {
        "dataset": "naturalbench",
        "metric": "accuracy",
        "overall": bucket_summary(correct, total),
        "leaderboard": {
            "Acc": safe_div(correct, total),
            "Q-Acc": safe_div(question_group_correct, len(by_question_group)),
            "I-Acc": safe_div(image_group_correct, len(by_image_group)),
            "G-Acc": safe_div(sample_group_correct, len(by_sample_group)),
        },
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_question_type": {
            question_type: bucket_summary(stats["correct"], stats["total"])
            for question_type, stats in ordered_items(by_question_type, KNOWN_QUESTION_TYPES)
        },
        "by_source": {
            source: bucket_summary(stats["correct"], stats["total"])
            for source, stats in ordered_items(by_source, KNOWN_SOURCES)
        },
    }

    print(f"Acc: {summary['leaderboard']['Acc']:.4f}")
    print(f"Q-Acc: {summary['leaderboard']['Q-Acc']:.4f}")
    print(f"I-Acc: {summary['leaderboard']['I-Acc']:.4f}")
    print(f"G-Acc: {summary['leaderboard']['G-Acc']:.4f}")
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")

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
