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
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text.lstrip("+")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_answer_prefix(text):
    lowered = str(text).strip().lower()
    for prefix in ["answer:", "answer is", "the answer is", "option", "choice"]:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return lowered


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
            option_map[normalize_text(option)] = valid_letters[idx]
        if normalized_pred in option_map:
            return option_map[normalized_pred]

    return None


def open_answer_candidates(pred_text):
    if pred_text is None:
        return set()

    raw_text = str(pred_text).strip()
    if not raw_text:
        return set()

    raw_candidates = {
        raw_text,
        raw_text.split("\n", 1)[0],
        strip_answer_prefix(raw_text),
        strip_answer_prefix(raw_text.split("\n", 1)[0]),
    }

    split_patterns = [r"\n", r"\.\s", r";", r",", r"\bbecause\b", r"\btherefore\b", r"\bthus\b"]
    for pattern in split_patterns:
        next_candidates = set()
        for candidate in raw_candidates:
            next_candidates.add(re.split(pattern, candidate, maxsplit=1)[0].strip())
        raw_candidates.update(next_candidates)

    normalized_candidates = {normalize_text(candidate) for candidate in raw_candidates if candidate}
    numeric_candidates = {match.lstrip("+") for match in re.findall(r"[-+]?\d+(?:\.\d+)?", raw_text)}
    return {candidate for candidate in normalized_candidates.union(numeric_candidates) if candidate}


def update_bucket(bucket, key, is_correct):
    bucket[key]["total"] += 1
    bucket[key]["correct"] += int(is_correct)


def summary_from_bucket(bucket):
    return {
        key: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": safe_div(stats["correct"], stats["total"]),
        }
        for key, stats in sorted(bucket.items())
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMMU predictions.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())

    pred_by_qid = {}
    duplicates_ignored = 0
    for row in results:
        qid = str(row.get("question_id"))
        if qid in pred_by_qid:
            duplicates_ignored += 1
        pred_by_qid[qid] = row.get("text", "")

    total = 0
    correct = 0
    missing = 0
    unparsable = 0
    missing_gold = 0
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample.get("question_id"))
        question_type = sample.get("question_type", "open")
        subject = sample.get("subject", "unknown")
        domain = sample.get("domain", "unknown")
        gold = sample.get("label")
        pred_text = pred_by_qid.get(qid)

        if gold is None or str(gold).strip() == "":
            missing_gold += 1
            if args.detail_file:
                details.append(
                    {
                        "question_id": qid,
                        "subject": subject,
                        "domain": domain,
                        "question_type": question_type,
                        "gold": gold,
                        "pred_text": pred_text,
                        "correct": None,
                        "missing": int(pred_text is None),
                        "unparsable": None,
                        "has_gold": 0,
                    }
                )
            continue

        total += 1

        if pred_text is None:
            missing += 1
            pred_norm = None
            is_correct = False
            is_unparsable = False
        elif question_type == "multiple_choice":
            valid_letters = sample.get("option_chars") or list(
                string.ascii_uppercase[: max(len(sample.get("options", [])), 1)]
            )
            pred_norm = extract_option_letter(pred_text, valid_letters, options=sample.get("options"))
            is_correct = pred_norm == str(gold).strip().upper()
            is_unparsable = pred_norm is None
        else:
            pred_candidates = open_answer_candidates(pred_text)
            pred_norm = sorted(pred_candidates)[0] if pred_candidates else None
            gold_norm = normalize_text(gold)
            is_correct = gold_norm in pred_candidates
            is_unparsable = not pred_candidates

        correct += int(is_correct)
        unparsable += int(is_unparsable)
        update_bucket(by_subject, subject, is_correct)
        update_bucket(by_domain, domain, is_correct)
        update_bucket(by_question_type, question_type, is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "subject": subject,
                    "domain": domain,
                    "question_type": question_type,
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                    "unparsable": int(is_unparsable),
                    "has_gold": 1,
                }
            )

    if total == 0:
        summary = {
            "dataset": "mmmu",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "questions_without_ground_truth": missing_gold,
            "duplicates_ignored": duplicates_ignored,
            "note": "No gold labels found in question file; accuracy not computed.",
        }
    else:
        summary = {
            "dataset": "mmmu",
            "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
            "missing_predictions": missing,
            "unparsable_predictions": unparsable,
            "questions_without_ground_truth": missing_gold,
            "duplicates_ignored": duplicates_ignored,
            "by_subject": summary_from_bucket(by_subject),
            "by_domain": summary_from_bucket(by_domain),
            "by_question_type": summary_from_bucket(by_question_type),
        }

    if summary.get("no_ground_truth"):
        print("[mmmu]")
        print("Ground truth unavailable for this run; accuracy not computed.")
    else:
        print("[mmmu]")
        print(f"Overall: {correct}/{total} = {safe_div(correct, total):.4f}")
        print(f"Missing predictions: {missing}")
        print(f"Unparsable predictions: {unparsable}")

        for field in ["by_domain", "by_subject", "by_question_type"]:
            print(f"{field}:")
            for key, stats in summary[field].items():
                print(f"  {key}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

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