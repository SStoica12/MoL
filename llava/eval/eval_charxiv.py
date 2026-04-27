import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import sys
import os
# import pdb; pdb.set_trace()

try:
    from llava.eval.openai_compat import OpenAICompatError, chat_completion_with_retry, create_client
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from llava.eval.openai_compat import OpenAICompatError, chat_completion_with_retry, create_client

NOT_APPLICABLE_PATTERNS = (
    "not applicable",
    "n/a",
    "na",
    "none",
    "no title",
    "no x-axis label",
    "no y-axis label",
    "no legend",
)

DESCRIPTIVE_FAMILY_BY_QID = {
    1: "title",
    2: "ocr",
    3: "ocr",
    4: "ocr",
    5: "ocr",
    6: "ocr",
    7: "ocr",
    8: "quant",
    9: "quant",
    10: "counting",
    11: "bool",
    12: "counting",
    13: "enum",
    14: "quant",
    15: "quant",
    16: "trend",
    17: "counting",
    18: "layout",
    19: "counting",
}

REASONING_FAMILY_BY_INST = {
    1: "text_grounded",
    2: "semantic",
    3: "numeric",
    4: "counting",
}

REASONING_JUDGE_RULES = {
    1: (
        "Give score 1 only if the prediction identifies the same label, term, subplot, curve, "
        "or entity as the ground truth. Equivalent notation is acceptable."
    ),
    2: (
        "Give score 1 only if the prediction has the same semantic meaning as the ground truth. "
        "If the question provides explicit options, the chosen option must exactly match."
    ),
    3: "Give score 1 only if the numeric value matches the ground truth value exactly.",
    4: "Give score 1 only if the numeric value or count matches the ground truth value exactly.",
}

CHARXIV_JUDGE_PROMPT = """
You are grading a chart-understanding answer.
Return a JSON object with exactly two keys:
- "extract_answer": the final answer you extracted from the prediction
- "score": 0 or 1

Scoring rule:
{rule}

Question | {question}
Ground Truth | {answer}
Prediction | {prediction}
""".strip()


def load_json_or_jsonl(path: Path):
    if path.suffix == ".json":
        data = json.load(open(path, "r"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
        raise ValueError(f"Unsupported JSON structure in {path}")

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
    payload = json.load(open(path, "r"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_grade_file(path: Path, grades):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(grades, f, indent=2, sort_keys=True)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def safe_div(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def strip_answer_prefix(text):
    normalized = str(text).strip()
    patterns = [
        r"^\s*(?:answer|final answer)\s*[:\-]\s*",
        r"^\s*(?:the answer is|it is|it's)\s+",
    ]
    for pattern in patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if lines:
        normalized = lines[-1]
    return normalized.strip()


def normalize_text(text):
    text = strip_answer_prefix(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def canonical_string(text):
    text = normalize_text(text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_not_applicable(text):
    normalized = canonical_string(text)
    if not normalized:
        return False
    if normalized in NOT_APPLICABLE_PATTERNS:
        return True
    return any(pattern in normalized for pattern in NOT_APPLICABLE_PATTERNS)


def normalize_yes_no(text):
    normalized = canonical_string(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return None


def parse_layout(text):
    normalized = canonical_string(text)
    match = re.search(r"(\d+)\s*(?:x|by)\s*(\d+)", normalized)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d+)\s*rows?\s*(?:and|,)\s*(\d+)\s*columns?", normalized)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_enum_items(text):
    normalized = canonical_string(text)
    if is_not_applicable(normalized):
        return ["not applicable"]
    parts = [part.strip() for part in re.split(r"\s*,\s*", normalized) if part.strip()]
    return parts


def normalize_trend(text):
    normalized = canonical_string(text)
    replacements = {
        "moving up": "increase",
        "goes up": "increase",
        "go up": "increase",
        "rises": "increase",
        "rising": "increase",
        "increasing": "increase",
        "moving down": "decrease",
        "goes down": "decrease",
        "go down": "decrease",
        "falls": "decrease",
        "falling": "decrease",
        "decreasing": "decrease",
        "stays constant": "stabilize",
        "stays the same": "stabilize",
        "remains constant": "stabilize",
        "remain constant": "stabilize",
        "plateaus": "stabilize",
        "plateau": "stabilize",
        "stabilizes": "stabilize",
        "stabilized": "stabilize",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("then", " then ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_numeric_value(text):
    normalized = canonical_string(text)
    if not normalized:
        return None

    normalized = normalized.replace(",", "")
    normalized = normalized.replace("%", "")
    normalized = normalized.replace("×", "x")
    normalized = normalized.strip()

    suffix_match = re.fullmatch(r"([-+]?\d*\.?\d+)\s*([kmb])", normalized)
    if suffix_match:
        scale = {"k": 1e3, "m": 1e6, "b": 1e9}[suffix_match.group(2)]
        return float(suffix_match.group(1)) * scale

    sci_match = re.fullmatch(r"([-+]?\d*\.?\d+)\s*[x*]\s*10\^?\(?([-+]?\d+)\)?", normalized)
    if sci_match:
        return float(sci_match.group(1)) * math.pow(10.0, int(sci_match.group(2)))

    power_match = re.fullmatch(r"10\^?\(?([-+]?\d+)\)?", normalized)
    if power_match:
        return math.pow(10.0, int(power_match.group(1)))

    plain_match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", normalized)
    if plain_match:
        try:
            return float(plain_match.group(0))
        except ValueError:
            return None
    return None


def numeric_equal(left, right):
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def descriptive_auto_grade(sample, pred_text):
    qid = int(sample.get("descriptive_qid"))
    family = DESCRIPTIVE_FAMILY_BY_QID.get(qid, "other")
    gold = str(sample.get("label", "")).strip()
    pred = "" if pred_text is None else str(pred_text).strip()

    if family in {"title", "ocr"}:
        if is_not_applicable(gold):
            return int(is_not_applicable(pred)), strip_answer_prefix(pred)
        gold_num = parse_numeric_value(gold)
        pred_num = parse_numeric_value(pred)
        if gold_num is not None and pred_num is not None:
            return int(numeric_equal(gold_num, pred_num)), strip_answer_prefix(pred)
        return int(canonical_string(gold) == canonical_string(pred)), strip_answer_prefix(pred)

    if family in {"quant", "counting"}:
        if is_not_applicable(gold):
            return int(is_not_applicable(pred)), strip_answer_prefix(pred)
        return int(numeric_equal(parse_numeric_value(gold), parse_numeric_value(pred))), strip_answer_prefix(pred)

    if family == "bool":
        if is_not_applicable(gold):
            return int(is_not_applicable(pred)), strip_answer_prefix(pred)
        return int(normalize_yes_no(gold) == normalize_yes_no(pred)), strip_answer_prefix(pred)

    if family == "enum":
        return int(parse_enum_items(gold) == parse_enum_items(pred)), strip_answer_prefix(pred)

    if family == "trend":
        return int(normalize_trend(gold) == normalize_trend(pred)), strip_answer_prefix(pred)

    if family == "layout":
        return int(parse_layout(gold) == parse_layout(pred)), strip_answer_prefix(pred)

    return int(canonical_string(gold) == canonical_string(pred)), strip_answer_prefix(pred)


def reasoning_auto_grade(sample, pred_text):
    gold = str(sample.get("label", "")).strip()
    pred = "" if pred_text is None else str(pred_text).strip()
    inst_category = int(sample.get("instruction_category") or 0)

    gold_bool = normalize_yes_no(gold)
    pred_bool = normalize_yes_no(pred)
    if gold_bool is not None and pred_bool is not None:
        return int(gold_bool == pred_bool), strip_answer_prefix(pred)

    gold_num = parse_numeric_value(gold)
    pred_num = parse_numeric_value(pred)
    if gold_num is not None and pred_num is not None:
        return int(numeric_equal(gold_num, pred_num)), strip_answer_prefix(pred)

    if canonical_string(gold) == canonical_string(pred):
        return 1, strip_answer_prefix(pred)

    if inst_category in {3, 4} and gold_num is not None and pred_num is None:
        return 0, strip_answer_prefix(pred)

    return None, strip_answer_prefix(pred)


def parse_judge_response(review_text):
    try:
        payload = json.loads(review_text)
        if isinstance(payload, dict):
            extract_answer = payload.get("extract_answer")
            score = payload.get("score")
            if isinstance(score, bool):
                score = int(score)
            if isinstance(score, (int, float)):
                return str(extract_answer).strip() if extract_answer is not None else "", int(score >= 0.5)
    except Exception:
        pass

    score_match = re.search(r"\b([01])\b", str(review_text))
    score = None if score_match is None else int(score_match.group(1))
    answer_match = re.search(r"extract_answer[\"']?\s*[:=]\s*[\"']?([^\"'\n}]+)", str(review_text), re.IGNORECASE)
    extract_answer = "" if answer_match is None else answer_match.group(1).strip()
    return extract_answer, score


def maybe_grade_reasoning(args, question_by_qid, prediction_by_qid, grades):
    pending = []
    for qid, sample in question_by_qid.items():
        if sample.get("mode") != "reasoning":
            continue

        grade_entry = grades.get(qid)
        if grade_entry and grade_entry.get("reviewed"):
            continue

        pred_text = None if qid not in prediction_by_qid else prediction_by_qid[qid].get("text", "")
        auto_score, extract_answer = reasoning_auto_grade(sample, pred_text)
        if auto_score is not None:
            grades[qid] = {
                "method": "auto",
                "reviewed": True,
                "score": int(auto_score),
                "extract_answer": extract_answer,
                "review": "",
            }
            continue

        pending.append(qid)

    if not pending:
        return None

    if not args.judge_model:
        return "No judge model configured; unresolved CharXiv reasoning questions were left unreviewed."

    try:
        client = create_client(api_key=args.judge_api_key, base_url=args.judge_base_url)
    except OpenAICompatError as exc:
        return str(exc)

    for qid in pending:
        sample = question_by_qid[qid]
        pred_row = prediction_by_qid.get(qid, {})
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
            return f"OpenAI grading failed for CharXiv on {qid}: {exc}"

        extract_answer, score = parse_judge_response(review_text)
        if score is None:
            score = 0

        grades[qid] = {
            "method": "judge",
            "reviewed": True,
            "score": int(score),
            "extract_answer": extract_answer,
            "review": str(review_text).strip(),
        }
        save_grade_file(args.grade_file_path, grades)

    return None


def bucket_template():
    return {"total": 0, "reviewed": 0, "correct": 0}


def finalize_bucket(stats):
    print("reviewed less than total: ", stats["reviewed"] < stats["total"], ", not that reviewed is ", stats["reviewed"], "; total is ", stats["total"])
    accuracy = None if stats["reviewed"] < stats["total"] else safe_div(stats["correct"], stats["total"])
    reviewed_accuracy = None if stats["reviewed"] == 0 else safe_div(stats["correct"], stats["reviewed"])
    return {
        "total": stats["total"],
        "reviewed": stats["reviewed"],
        "correct": stats["correct"],
        "accuracy": accuracy,
        "reviewed_accuracy": reviewed_accuracy,
    }


def build_summary_and_details(question_by_qid, prediction_by_qid, grades):
    total = 0
    reviewed = 0
    correct = 0
    missing_predictions = 0
    auto_graded = 0
    judge_graded = 0
    by_category = defaultdict(bucket_template)
    by_subcategory = defaultdict(bucket_template)
    details = []

    for qid, sample in question_by_qid.items():
        total += 1
        pred_row = prediction_by_qid.get(qid)
        pred_text = None if pred_row is None else pred_row.get("text")
        if pred_row is None:
            missing_predictions += 1

        if sample.get("mode") == "descriptive":
            score, extract_answer = descriptive_auto_grade(sample, pred_text)
            grade_entry = {
                "method": "auto",
                "reviewed": True,
                "score": int(score),
                "extract_answer": extract_answer,
                "review": "",
            }
            grades[qid] = grade_entry
        else:
            grade_entry = grades.get(qid, {})

        mode = str(sample.get("mode", "unknown"))
        if mode == "descriptive":
            subcategory = f"descriptive/{DESCRIPTIVE_FAMILY_BY_QID.get(int(sample.get('descriptive_qid')), 'other')}"
        else:
            subcategory = f"reasoning/{REASONING_FAMILY_BY_INST.get(int(sample.get('instruction_category') or 0), 'other')}"

        by_category[mode]["total"] += 1
        by_subcategory[subcategory]["total"] += 1

        reviewed_flag = bool(grade_entry.get("reviewed"))
        score_value = int(grade_entry.get("score", 0)) if reviewed_flag else None
        if reviewed_flag:
            reviewed += 1
            correct += int(score_value)
            by_category[mode]["reviewed"] += 1
            by_category[mode]["correct"] += int(score_value)
            by_subcategory[subcategory]["reviewed"] += 1
            by_subcategory[subcategory]["correct"] += int(score_value)

            if grade_entry.get("method") == "auto":
                auto_graded += 1
            elif grade_entry.get("method") == "judge":
                judge_graded += 1

        details.append(
            {
                "question_id": qid,
                "figure_id": sample.get("figure_id"),
                "mode": mode,
                "subcategory": subcategory,
                "gold": sample.get("label"),
                "pred_text": pred_text,
                "reviewed": int(reviewed_flag),
                "score": score_value,
                "grading_method": grade_entry.get("method"),
                "extract_answer": grade_entry.get("extract_answer"),
                "review": grade_entry.get("review"),
            }
        )

    summary = {
        "dataset": "charxiv",
        "metric": "accuracy",
        "overall": finalize_bucket({"total": total, "reviewed": reviewed, "correct": correct}),
        "prediction_count": len(prediction_by_qid),
        "missing_predictions": missing_predictions,
        "grade_count": reviewed,
        "auto_graded_count": auto_graded,
        "judge_graded_count": judge_graded,
        "by_category": {key: finalize_bucket(stats) for key, stats in sorted(by_category.items())},
        "by_subcategory": {key: finalize_bucket(stats) for key, stats in sorted(by_subcategory.items())},
    }

    if reviewed < total:
        summary["external_judge_required"] = True
        summary["note"] = (
            "Some CharXiv reasoning questions remain unreviewed. "
            "Set OPENAI_API_KEY and --judge-model, or rerun with an existing grade file."
        )

    return summary, details


def main():
    parser = argparse.ArgumentParser(description="Evaluate CharXiv predictions.")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--grade-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    question_file = Path(args.question_file).expanduser().resolve()
    result_file = Path(args.result_file).expanduser().resolve()
    grade_file = Path(args.grade_file).expanduser().resolve()

    questions = load_json_or_jsonl(question_file)
    results = load_json_or_jsonl(result_file)
    question_by_qid = {str(row.get("question_id")): row for row in questions}
    prediction_by_qid = {str(row.get("question_id")): row for row in results}

    grades = load_grade_file(grade_file)
    args.grade_file_path = grade_file
    judge_error = maybe_grade_reasoning(args, question_by_qid, prediction_by_qid, grades)
    save_grade_file(grade_file, grades)

    summary, details = build_summary_and_details(question_by_qid, prediction_by_qid, grades)
    summary["judge_model"] = args.judge_model or None
    summary["grade_file"] = str(grade_file)
    if judge_error:
        summary["judge_error"] = judge_error
        if summary.get("note"):
            summary["note"] = f"{summary['note']} {judge_error}"
        else:
            summary["note"] = judge_error

    overall_accuracy = summary["overall"].get("accuracy")
    if overall_accuracy is None:
        print("Overall CharXiv accuracy: unavailable")
    else:
        print(f"Overall CharXiv accuracy: {overall_accuracy * 100.0:.2f}")
    print(f"Reviewed questions: {summary['grade_count']}/{summary['overall']['total']}")
    print(f"Missing predictions: {summary['missing_predictions']}")
    for category, stats in summary["by_category"].items():
        if stats["accuracy"] is not None:
            print(f"{category}: {stats['accuracy'] * 100.0:.2f}")
        else:
            print(f"{category}: review incomplete")

    if args.output_file:
        save_json(Path(args.output_file).expanduser().resolve(), summary)
    if args.detail_file:
        save_jsonl(Path(args.detail_file).expanduser().resolve(), details)


if __name__ == "__main__":
    main()