import argparse
import json
from collections import defaultdict
from pathlib import Path

from llava.eval.openai_compat import OpenAICompatError, chat_completion_with_retry, create_client


SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the quality of the answer."


def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_review_scores(review_text):
    if not review_text:
        return None
    first_line = str(review_text).strip().splitlines()[0]
    first_line = first_line.replace(",", " ")
    parts = [part for part in first_line.split() if part]
    if len(parts) < 2:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError:
        return None


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def normalize_caption(value):
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def build_review_prompt(question, context, reference_answer, model_answer, rule):
    role = rule["role"]
    prompt = rule["prompt"]
    return (
        f"[Context]\n{normalize_caption(context.get('caption', ''))}\n\n"
        f"[Question]\n{question['text']}\n\n"
        f"[{role} 1]\n{reference_answer}\n\n[End of {role} 1]\n\n"
        f"[{role} 2]\n{model_answer}\n\n[End of {role} 2]\n\n"
        f"[System]\n{prompt}\n\n"
    )


def load_existing_reviews(review_file: Path):
    if not review_file.exists():
        return {}

    reviews = {}
    for row in load_jsonl(review_file):
        qid = str(row.get("question_id"))
        reviews[qid] = row
    return reviews


def save_review_rows(review_file: Path, review_rows, questions):
    review_file.parent.mkdir(parents=True, exist_ok=True)
    ordered_qids = [str(question["question_id"]) for question in questions]
    with open(review_file, "w") as f:
        for qid in ordered_qids:
            row = review_rows.get(qid)
            if row is not None:
                f.write(json.dumps(row) + "\n")


def maybe_generate_reviews(args, questions, contexts, reference_answers, model_answers, rule_dict, review_rows):
    pending_questions = [question for question in questions if str(question["question_id"]) not in review_rows]
    if not pending_questions:
        return None

    if not args.judge_model:
        return "No judge model configured; skipped GPT review generation."

    try:
        client = create_client(api_key=args.judge_api_key, base_url=args.judge_base_url)
    except OpenAICompatError as exc:
        return str(exc)

    context_by_image = {str(row.get("image")): row for row in contexts}
    reference_by_qid = {str(row.get("question_id")): row for row in reference_answers}
    model_by_qid = {str(row.get("question_id")): row for row in model_answers}

    for index, question in enumerate(pending_questions, start=1):
        qid = str(question["question_id"])
        image_key = str(question.get("image"))
        context = context_by_image.get(image_key)
        if context is None:
            return f"Missing context for image {image_key}; cannot complete LLaVA-Bench review."

        reference_row = reference_by_qid.get(qid)
        if reference_row is None:
            return f"Missing reference answer for question {qid}; cannot complete LLaVA-Bench review."

        category = str(question.get("category", "unknown"))
        category_key = f"llava_bench_{category}"
        rule = rule_dict.get(category_key)
        if rule is None:
            return f"Missing review rule for category {category_key}."

        model_row = model_by_qid.get(qid, {})
        model_answer = str(model_row.get("text", ""))
        prompt = build_review_prompt(
            question=question,
            context=context,
            reference_answer=str(reference_row.get("text", "")),
            model_answer=model_answer,
            rule=rule,
        )

        try:
            review_text = chat_completion_with_retry(
                client,
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                sleep_seconds=args.retry_sleep_seconds,
            )
        except OpenAICompatError as exc:
            return f"OpenAI review failed after {index - 1} generated review(s): {exc}"

        review_rows[qid] = {
            "question_id": qid,
            "answer1_id": reference_row.get("answer_id", reference_row.get("question_id", qid)),
            "answer2_id": model_row.get("answer_id", model_row.get("question_id", qid)),
            "category": category_key,
            "content": review_text,
            "tuple": parse_review_scores(review_text),
        }

    return None


def build_summary_and_details(questions, model_answers, review_rows):
    model_by_qid = {str(row.get("question_id")): row for row in model_answers}
    total_questions = len(questions)
    prediction_count = len(model_by_qid)
    missing_predictions = 0
    reviewed_count = 0
    overall_ref_sum = 0.0
    overall_model_sum = 0.0
    by_category = defaultdict(lambda: {"reference_score_sum": 0.0, "model_score_sum": 0.0, "review_count": 0, "total": 0})
    details = []

    for question in questions:
        qid = str(question["question_id"])
        category = str(question.get("category", "unknown"))
        prediction_row = model_by_qid.get(qid)
        if prediction_row is None:
            missing_predictions += 1
        review_row = review_rows.get(qid)
        scores = None if review_row is None else review_row.get("tuple")
        if scores is None and review_row is not None:
            scores = parse_review_scores(review_row.get("content", ""))

        ref_score = None
        model_score = None
        relative_score_pct = None
        if scores is not None:
            ref_score, model_score = scores
            overall_ref_sum += ref_score
            overall_model_sum += model_score
            reviewed_count += 1
            by_category[category]["reference_score_sum"] += ref_score
            by_category[category]["model_score_sum"] += model_score
            by_category[category]["review_count"] += 1
            relative_score_pct = safe_div(model_score, ref_score)
            if relative_score_pct is not None:
                relative_score_pct *= 100.0

        by_category[category]["total"] += 1
        details.append(
            {
                "question_id": qid,
                "image": question.get("image"),
                "category": category,
                "question": question.get("text"),
                "pred_text": None if prediction_row is None else prediction_row.get("text"),
                "reviewed": int(scores is not None),
                "reference_score": ref_score,
                "model_score": model_score,
                "relative_score_pct": relative_score_pct,
                "review": None if review_row is None else review_row.get("content"),
            }
        )

    overall = {"total": total_questions, "reviewed": reviewed_count}
    if reviewed_count > 0:
        overall["reference_score"] = overall_ref_sum / reviewed_count
        overall["score"] = overall_model_sum / reviewed_count
        overall["relative_score_pct"] = safe_div(overall["score"], overall["reference_score"])
        if overall["relative_score_pct"] is not None:
            overall["relative_score_pct"] *= 100.0
    else:
        overall["reference_score"] = None
        overall["score"] = None
        overall["relative_score_pct"] = None

    summary = {
        "dataset": "llava-bench-in-the-wild",
        "metric": "gpt_review_relative_score_pct",
        "overall": overall,
        "prediction_count": prediction_count,
        "missing_predictions": missing_predictions,
        "review_count": reviewed_count,
        "by_category": {},
    }

    for category, stats in sorted(by_category.items()):
        row = {"total": stats["total"], "reviewed": stats["review_count"]}
        if stats["review_count"] > 0:
            row["reference_score"] = stats["reference_score_sum"] / stats["review_count"]
            row["score"] = stats["model_score_sum"] / stats["review_count"]
            row["relative_score_pct"] = safe_div(row["score"], row["reference_score"])
            if row["relative_score_pct"] is not None:
                row["relative_score_pct"] *= 100.0
        else:
            row["reference_score"] = None
            row["score"] = None
            row["relative_score_pct"] = None
        summary["by_category"][category] = row

    if reviewed_count == 0:
        summary["external_judge_required"] = True
        summary["note"] = (
            "No GPT review scores are available. Set OPENAI_API_KEY and --judge-model, "
            "or rerun with an existing review file."
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
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Bench-in-the-Wild predictions.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--context-file", type=str, required=True)
    parser.add_argument("--reference-answer-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--review-file", type=str, required=True)
    parser.add_argument("--rule-file", type=str, default="llava/eval/table/rule.json")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    question_file = Path(args.question_file).expanduser().resolve()
    context_file = Path(args.context_file).expanduser().resolve()
    reference_answer_file = Path(args.reference_answer_file).expanduser().resolve()
    result_file = Path(args.result_file).expanduser().resolve()
    review_file = Path(args.review_file).expanduser().resolve()
    rule_file = Path(args.rule_file).expanduser().resolve()

    questions = load_jsonl(question_file)
    contexts = load_jsonl(context_file)
    reference_answers = load_jsonl(reference_answer_file)
    model_answers = load_jsonl(result_file)
    rule_dict = json.load(open(rule_file, "r"))
    review_rows = load_existing_reviews(review_file)

    judge_error = maybe_generate_reviews(
        args=args,
        questions=questions,
        contexts=contexts,
        reference_answers=reference_answers,
        model_answers=model_answers,
        rule_dict=rule_dict,
        review_rows=review_rows,
    )
    save_review_rows(review_file, review_rows, questions)

    summary, details = build_summary_and_details(
        questions=questions,
        model_answers=model_answers,
        review_rows=review_rows,
    )
    summary["judge_model"] = args.judge_model or None
    summary["review_file"] = str(review_file)
    if judge_error:
        summary["judge_error"] = judge_error
        if summary.get("note"):
            summary["note"] = f"{summary['note']} {judge_error}"
        else:
            summary["note"] = judge_error

    if summary["overall"]["relative_score_pct"] is not None:
        print(
            f"Overall relative score: {summary['overall']['relative_score_pct']:.1f} "
            f"(score={summary['overall']['score']:.3f}, reference={summary['overall']['reference_score']:.3f})"
        )
    else:
        print("Overall relative score: unavailable")
    print(f"Missing predictions: {summary['missing_predictions']}")
    print(f"Reviewed questions: {summary['review_count']}/{summary['overall']['total']}")
    for category, stats in summary["by_category"].items():
        if stats["relative_score_pct"] is not None:
            print(
                f"{category}: {stats['relative_score_pct']:.1f} "
                f"(score={stats['score']:.3f}, reference={stats['reference_score']:.3f})"
            )
        else:
            print(f"{category}: review unavailable")

    if args.output_file:
        save_json(Path(args.output_file).expanduser().resolve(), summary)
    if args.detail_file:
        save_jsonl(Path(args.detail_file).expanduser().resolve(), details)


if __name__ == "__main__":
    main()
