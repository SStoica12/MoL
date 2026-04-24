import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path


KNOWN_SOURCES = ["ADE20K", "COCO", "Omni3D"]
KNOWN_TYPES = ["2D", "3D"]


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


def safe_mean(values):
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered)) / float(len(filtered))


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


def bucket_summary(correct, total, accuracy=None, **extra):
    payload = {
        "correct": correct,
        "total": total,
        "accuracy": safe_div(correct, total) if accuracy is None else float(accuracy),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate CV-Bench predictions.")
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
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type_raw = defaultdict(lambda: {"correct": 0, "total": 0})
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        pred_text = pred_by_qid.get(qid)
        if pred_text is None:
            missing += 1

        options = sample.get("options") or []
        option_chars = sample.get("option_chars") or list(string.ascii_uppercase[: max(len(options), 1)])
        gold = str(sample.get("label", "")).strip().upper()
        pred_norm = extract_option_letter(pred_text, option_chars, options=options)
        if pred_text is not None and pred_norm is None:
            unparsable += 1

        is_correct = pred_norm == gold
        total += 1
        correct += int(is_correct)

        source = str(sample.get("source", "unknown")).strip() or "unknown"
        task = str(sample.get("task", "unknown")).strip() or "unknown"
        type_name = str(sample.get("type", "unknown")).strip() or "unknown"

        by_source[source]["total"] += 1
        by_source[source]["correct"] += int(is_correct)
        by_type_raw[type_name]["total"] += 1
        by_type_raw[type_name]["correct"] += int(is_correct)
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "type": type_name,
                    "task": task,
                    "source": source,
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                    "unparsable": int(pred_text is not None and pred_norm is None),
                }
            )

    by_source_summary = {
        source: bucket_summary(stats["correct"], stats["total"])
        for source, stats in ordered_items(by_source, KNOWN_SOURCES)
    }
    ade20k_accuracy = by_source_summary.get("ADE20K", {}).get("accuracy")
    coco_accuracy = by_source_summary.get("COCO", {}).get("accuracy")
    omni3d_accuracy = by_source_summary.get("Omni3D", {}).get("accuracy")
    accuracy_2d = safe_mean([ade20k_accuracy, coco_accuracy])
    accuracy_3d = omni3d_accuracy

    official_accuracy = None
    if accuracy_2d is not None and accuracy_3d is not None:
        official_accuracy = (accuracy_2d + accuracy_3d) / 2.0
    elif accuracy_2d is not None:
        official_accuracy = accuracy_2d
    elif accuracy_3d is not None:
        official_accuracy = accuracy_3d
    else:
        official_accuracy = safe_div(correct, total)

    by_type_summary = {}
    for type_name, stats in ordered_items(by_type_raw, KNOWN_TYPES):
        calibrated_accuracy = None
        if type_name == "2D":
            calibrated_accuracy = accuracy_2d
        elif type_name == "3D":
            calibrated_accuracy = accuracy_3d
        by_type_summary[type_name] = bucket_summary(
            stats["correct"],
            stats["total"],
            accuracy=calibrated_accuracy,
            raw_accuracy=safe_div(stats["correct"], stats["total"]),
        )

    by_task_summary = {
        task: bucket_summary(stats["correct"], stats["total"])
        for task, stats in ordered_items(by_task, [])
    }

    raw_overall_accuracy = safe_div(correct, total)
    summary = {
        "dataset": "cvbench",
        "metric": "cvbench_accuracy",
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": official_accuracy,
            "raw_accuracy": raw_overall_accuracy,
        },
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "by_source": by_source_summary,
        "by_type": by_type_summary,
        "by_task": by_task_summary,
        "official_accuracy_components": {
            "ADE20K": ade20k_accuracy,
            "COCO": coco_accuracy,
            "Omni3D": omni3d_accuracy,
            "2D": accuracy_2d,
            "3D": accuracy_3d,
        },
    }

    print(f"CV-Bench official accuracy: {official_accuracy:.4f}")
    print(f"Raw sample accuracy: {raw_overall_accuracy:.4f}")
    if accuracy_2d is not None:
        print(f"2D accuracy: {accuracy_2d:.4f}")
    if accuracy_3d is not None:
        print(f"3D accuracy: {accuracy_3d:.4f}")
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")
    for source, stats in by_source_summary.items():
        print(f"{source}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

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
