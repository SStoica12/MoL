import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from llava.eval.openai_compat import OpenAICompatError, chat_completion_with_retry, create_client


MMVET_PROMPT = """
Compare the ground truth and prediction from AI models, and determine the correctness score of the prediction.
Answer the question using a single floating-point value in the range [0.0, 1.0], where:
- 1.0 means the prediction is completely correct,
- 0.0 means the prediction is completely incorrect,
- and intermediate values reflect partial correctness.

Focus on whether the prediction captures the meaning of the ground truth, even if wording differs.
Do not explain your answer.

Question | {question}
Ground Truth | {answer}
Prediction | {prediction}
Correctness |
""".strip()


def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_grade_file(path: Path):
    if not path.exists():
        return {}
    data = json.load(open(path, "r"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def save_grade_file(path: Path, grades):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(grades, f, indent=2, sort_keys=True)


def parse_score(text):
    if text is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(text))
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return min(max(value, 0.0), 1.0)


def safe_mean(values):
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def normalize_capabilities(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def maybe_grade_predictions(args, annotation_by_qid, prediction_by_qid, grades):
    pending = []
    for qid in annotation_by_qid:
        entry = grades.get(qid, {})
        scores = entry.get("scores", [])
        if len(scores) < args.num_runs:
            pending.append(qid)

    if not pending:
        return None

    if not args.judge_model:
        return "No judge model configured; skipped MM-Vet grading."

    try:
        client = create_client(api_key=args.judge_api_key, base_url=args.judge_base_url)
    except OpenAICompatError as exc:
        return str(exc)

    for qid in pending:
        sample = annotation_by_qid[qid]
        pred_row = prediction_by_qid.get(qid, {})
        prediction = str(pred_row.get("text", ""))

        grade_entry = grades.setdefault(
            qid,
            {
                "scores": [],
                "reviews": [],
            },
        )

        while len(grade_entry["scores"]) < args.num_runs:
            prompt = MMVET_PROMPT.format(
                question=str(sample.get("question", "")).strip(),
                answer=str(sample.get("answer", "")).strip(),
                prediction=prediction.strip(),
            )
            try:
                review_text = chat_completion_with_retry(
                    client,
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    sleep_seconds=args.retry_sleep_seconds,
                )
            except OpenAICompatError as exc:
                return f"OpenAI grading failed for MM-Vet after {len(grade_entry['scores'])} run(s) on {qid}: {exc}"

            score = parse_score(review_text)
            if score is None:
                review_text = str(review_text).strip()
                grade_entry["reviews"].append(review_text)
                grade_entry["scores"].append(0.0)
            else:
                grade_entry["reviews"].append(str(review_text).strip())
                grade_entry["scores"].append(score)

            save_grade_file(args.grade_file_path, grades)

    return None


def build_summary_and_details(annotation_by_qid, prediction_by_qid, grades):
    total = len(annotation_by_qid)
    prediction_count = len(prediction_by_qid)
    missing_predictions = 0
    graded_questions = 0
    score_sums = []
    by_capability = defaultdict(lambda: {"scores": [], "total": 0, "graded": 0})
    by_capability_combo = defaultdict(lambda: {"scores": [], "total": 0, "graded": 0})
    details = []

    for qid, sample in annotation_by_qid.items():
        pred_row = prediction_by_qid.get(qid)
        if pred_row is None:
            missing_predictions += 1

        grade_entry = grades.get(qid, {})
        scores = [float(score) for score in grade_entry.get("scores", [])]
        mean_score = safe_mean(scores)
        capabilities = normalize_capabilities(sample.get("capability"))
        combo_key = ", ".join(capabilities) if capabilities else "unknown"

        if mean_score is not None:
            graded_questions += 1
            score_sums.append(mean_score)

        for capability in capabilities or ["unknown"]:
            by_capability[capability]["total"] += 1
            if mean_score is not None:
                by_capability[capability]["graded"] += 1
                by_capability[capability]["scores"].append(mean_score)

        by_capability_combo[combo_key]["total"] += 1
        if mean_score is not None:
            by_capability_combo[combo_key]["graded"] += 1
            by_capability_combo[combo_key]["scores"].append(mean_score)

        details.append(
            {
                "question_id": qid,
                "image": sample.get("imagename"),
                "question": sample.get("question"),
                "gold": sample.get("answer"),
                "pred_text": None if pred_row is None else pred_row.get("text"),
                "capabilities": capabilities,
                "reviewed": int(mean_score is not None),
                "score": mean_score,
                "scores": scores,
                "reviews": grade_entry.get("reviews", []),
            }
        )

    summary = {
        "dataset": "mmvet",
        "metric": "gpt_judge_score",
        "overall": {
            "total": total,
            "reviewed": graded_questions,
            "score": safe_mean(score_sums),
            "score_pct": None if safe_mean(score_sums) is None else safe_mean(score_sums) * 100.0,
        },
        "prediction_count": prediction_count,
        "missing_predictions": missing_predictions,
        "grade_count": graded_questions,
        "by_capability": {},
        "by_capability_combo": {},
    }

    for capability, stats in sorted(by_capability.items()):
        mean_score = safe_mean(stats["scores"])
        summary["by_capability"][capability] = {
            "total": stats["total"],
            "reviewed": stats["graded"],
            "score": mean_score,
            "score_pct": None if mean_score is None else mean_score * 100.0,
        }

    for combo, stats in sorted(by_capability_combo.items()):
        mean_score = safe_mean(stats["scores"])
        summary["by_capability_combo"][combo] = {
            "total": stats["total"],
            "reviewed": stats["graded"],
            "score": mean_score,
            "score_pct": None if mean_score is None else mean_score * 100.0,
        }

    if graded_questions == 0:
        summary["external_judge_required"] = True
        summary["note"] = (
            "No MM-Vet judge scores are available. Set OPENAI_API_KEY and --judge-model, "
            "or rerun with an existing grade file."
        )

    return summary, details


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MM-Vet predictions.")
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--grade-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    annotation_file = Path(args.annotation_file).expanduser().resolve()
    result_file = Path(args.result_file).expanduser().resolve()
    grade_file = Path(args.grade_file).expanduser().resolve()

    annotation_by_qid = json.load(open(annotation_file, "r"))
    if not isinstance(annotation_by_qid, dict):
        raise ValueError(f"Expected a JSON object in {annotation_file}")

    predictions = load_jsonl(result_file)
    prediction_by_qid = {str(row.get("question_id")): row for row in predictions}

    grades = load_grade_file(grade_file)
    args.grade_file_path = grade_file
    judge_error = maybe_grade_predictions(args, annotation_by_qid, prediction_by_qid, grades)
    save_grade_file(grade_file, grades)

    summary, details = build_summary_and_details(annotation_by_qid, prediction_by_qid, grades)
    summary["judge_model"] = args.judge_model or None
    summary["grade_file"] = str(grade_file)
    if judge_error:
        summary["judge_error"] = judge_error
        if summary.get("note"):
            summary["note"] = f"{summary['note']} {judge_error}"
        else:
            summary["note"] = judge_error

    if summary["overall"]["score_pct"] is not None:
        print(f"Overall MM-Vet score: {summary['overall']['score_pct']:.1f}")
    else:
        print("Overall MM-Vet score: unavailable")
    print(f"Missing predictions: {summary['missing_predictions']}")
    print(f"Reviewed questions: {summary['grade_count']}/{summary['overall']['total']}")
    for capability, stats in summary["by_capability"].items():
        if stats["score_pct"] is not None:
            print(f"{capability}: {stats['score_pct']:.1f}")
        else:
            print(f"{capability}: review unavailable")

    if args.output_file:
        save_json(Path(args.output_file).expanduser().resolve(), summary)
    if args.detail_file:
        save_jsonl(Path(args.detail_file).expanduser().resolve(), details)


if __name__ == "__main__":
    main()
