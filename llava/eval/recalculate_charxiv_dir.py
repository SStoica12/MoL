import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from llava.eval.eval_charxiv import (
    CHARXIV_JUDGE_PROMPT,
    REASONING_JUDGE_RULES,
    build_summary_and_details,
    chat_completion_with_retry,
    create_client,
    load_grade_file,
    load_json_or_jsonl,
    parse_judge_response,
    reasoning_auto_grade,
    save_grade_file,
    save_json,
    save_jsonl,
)


def judge_one(client, args, qid, sample, pred_row):
    rule = REASONING_JUDGE_RULES.get(
        int(sample.get("instruction_category") or 0),
        "Give score 1 only if the prediction is semantically equivalent to the ground truth.",
    )
    prompt = CHARXIV_JUDGE_PROMPT.format(
        rule=rule,
        question=str(sample.get("raw_question") or sample.get("text") or "").strip(),
        answer=str(sample.get("label", "")).strip(),
        prediction=str(pred_row.get("text", "")).strip(),
    )
    review_text = chat_completion_with_retry(
        client,
        model=args.judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        sleep_seconds=args.retry_sleep_seconds,
    )
    extract_answer, score = parse_judge_response(review_text)
    if score is None:
        score = 0
    return qid, {
        "method": "judge",
        "reviewed": True,
        "score": int(score),
        "extract_answer": extract_answer,
        "review": str(review_text).strip(),
    }


def existing_accuracy(summary_file):
    if not summary_file.exists():
        return None
    summary = json.load(open(summary_file, "r"))
    return summary.get("overall", {}).get("accuracy")


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate one CharXiv model output directory when accuracy is null."
    )
    parser.add_argument("output_dir", help="Directory containing predictions.jsonl, summary.json, details.jsonl, grades.json.")
    parser.add_argument(
        "--question-file",
        default="playground/data/eval/CharXiv/validation.jsonl",
        help="CharXiv question jsonl used to create predictions.",
    )
    parser.add_argument("--predictions-file", default="predictions.jsonl")
    parser.add_argument("--summary-file", default="summary.json")
    parser.add_argument("--detail-file", default="details.jsonl")
    parser.add_argument("--grade-file", default="grades.json")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Rebuild files even when summary accuracy is already present.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    question_file = Path(args.question_file).expanduser().resolve()
    predictions_file = output_dir / args.predictions_file
    summary_file = output_dir / args.summary_file
    detail_file = output_dir / args.detail_file
    grade_file = output_dir / args.grade_file

    if not predictions_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_file}")

    current_accuracy = existing_accuracy(summary_file)
    if current_accuracy is not None and not args.force:
        print(f"CharXiv accuracy already present: {current_accuracy * 100.0:.2f}")
        print(f"Summary: {summary_file}")
        return

    questions = load_json_or_jsonl(question_file)
    results = load_json_or_jsonl(predictions_file)
    question_by_qid = {str(row.get("question_id")): row for row in questions}
    prediction_by_qid = {str(row.get("question_id")): row for row in results}
    grades = load_grade_file(grade_file)

    pending = []
    for qid, sample in question_by_qid.items():
        if sample.get("mode") != "reasoning":
            continue
        grade_entry = grades.get(qid)
        if grade_entry and grade_entry.get("reviewed"):
            continue

        pred_row = prediction_by_qid.get(qid, {})
        auto_score, extract_answer = reasoning_auto_grade(sample, pred_row.get("text", ""))
        if auto_score is not None:
            grades[qid] = {
                "method": "auto",
                "reviewed": True,
                "score": int(auto_score),
                "extract_answer": extract_answer,
                "review": "",
            }
        else:
            pending.append((qid, sample, pred_row))

    if pending:
        client = create_client()
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(judge_one, client, args, qid, sample, pred_row) for qid, sample, pred_row in pending]
            for future in as_completed(futures):
                qid, grade = future.result()
                grades[qid] = grade
                completed += 1
                if completed % args.save_every == 0:
                    save_grade_file(grade_file, grades)
                    print(f"Judged {completed}/{len(pending)}", flush=True)

    save_grade_file(grade_file, grades)
    summary, details = build_summary_and_details(question_by_qid, prediction_by_qid, grades)
    summary["judge_model"] = args.judge_model
    summary["grade_file"] = str(grade_file)
    save_json(summary_file, summary)
    save_jsonl(detail_file, details)

    overall = summary["overall"]
    accuracy = overall.get("accuracy")
    if accuracy is None:
        print(f"CharXiv accuracy is still unavailable: reviewed {overall['reviewed']}/{overall['total']}")
    else:
        print(f"CharXiv accuracy: {accuracy * 100.0:.2f}")
    print(f"Reviewed questions: {overall['reviewed']}/{overall['total']}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
