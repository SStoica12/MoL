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
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return lowered[len(prefix) :].strip()
    return lowered


def normalize_text(text):
    text = strip_answer_prefix(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(a: str, b: str):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def anls_score(gold_answers, prediction):
    if prediction is None:
        return 0.0, None

    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return 0.0, pred_norm

    scores = []
    for answer in gold_answers:
        gold_norm = normalize_text(answer)
        if not gold_norm:
            continue
        distance = levenshtein_distance(gold_norm, pred_norm)
        score = 1.0 - safe_div(distance, max(len(gold_norm), len(pred_norm)))
        if score < 0.5:
            score = 0.0
        scores.append(score)

    if not scores:
        return 0.0, pred_norm
    return max(scores), pred_norm


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocVQA predictions with ANLS.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(row.get("question_id")): row.get("text", "") for row in results}

    total = 0
    score_sum = 0.0
    missing = 0
    by_question_type = defaultdict(lambda: {"score_sum": 0.0, "total": 0})
    details = []

    for sample in questions:
        qid = str(sample["question_id"])
        pred_text = pred_by_qid.get(qid)
        if pred_text is None:
            missing += 1
        answers = sample.get("answers") or []
        score, pred_norm = anls_score(answers, pred_text)

        total += 1
        score_sum += score
        question_types = sample.get("question_types") or ["unknown"]
        for question_type in question_types:
            by_question_type[str(question_type)]["score_sum"] += score
            by_question_type[str(question_type)]["total"] += 1

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "answers": answers,
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "anls": score,
                    "missing": int(pred_text is None),
                    "question_types": question_types,
                }
            )

    summary = {
        "dataset": "docvqa",
        "metric": "anls",
        "overall": {"correct": score_sum, "total": total, "accuracy": safe_div(score_sum, total)},
        "missing_predictions": missing,
        "by_question_type": {
            question_type: {
                "correct": stats["score_sum"],
                "total": stats["total"],
                "accuracy": safe_div(stats["score_sum"], stats["total"]),
            }
            for question_type, stats in sorted(by_question_type.items())
        },
    }

    print(f"Overall ANLS: {summary['overall']['accuracy']:.4f}")
    print(f"Missing predictions: {missing}")
    for question_type, stats in summary["by_question_type"].items():
        print(f"{question_type}: {stats['accuracy']:.4f}")

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
