import argparse
import csv
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path

from m4c_evaluator import TextVQAAccuracyEvaluator


def safe_div(numerator, denominator):
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_or_jsonl(path: Path):
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    return load_jsonl(path)


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def normalize_text_loose(text):
    text = str(text).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        norm_pred = normalize_text(text)
        option_map = {}
        for idx, option in enumerate(options):
            letter = valid_letters[idx]
            option_map[normalize_text(option)] = letter
        if norm_pred in option_map:
            return option_map[norm_pred]

    return None


def extract_first_line_question(text):
    return str(text).strip().split("\n")[0].strip().lower()


def normalize_gqa_answer(text):
    return str(text).rstrip(".").strip().lower()


def normalize_pope_prediction(text):
    if text is None:
        return None
    text = str(text)
    if "." in text:
        text = text.split(".", 1)[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def print_summary(summary):
    dataset = summary.get("dataset", "unknown")
    overall = summary.get("overall", {})
    print(f"[{dataset}]")
    if overall:
        if "accuracy" in overall:
            acc = overall["accuracy"]
            if isinstance(acc, (int, float)):
                print(
                    f"Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} = {acc:.4f}"
                )
            else:
                print(f"Overall: {overall}")
        else:
            print(f"Overall: {overall}")

    if "missing_predictions" in summary:
        print(f"Missing predictions: {summary['missing_predictions']}")
    if "unparsable_predictions" in summary:
        print(f"Unparsable predictions: {summary['unparsable_predictions']}")
    if summary.get("no_ground_truth"):
        print("Ground truth unavailable for this run; accuracy not computed.")

    for field in [
        "by_category",
        "by_structural_type",
        "by_semantic_type",
        "by_question_type",
        "by_data_type",
        "by_answer_type",
        "by_split",
    ]:
        bucket = summary.get(field)
        if not bucket:
            continue
        print(f"{field}:")
        for key, stats in sorted(bucket.items()):
            if isinstance(stats, dict) and "accuracy" in stats:
                print(
                    f"  {key}: {stats.get('correct', 0)}/{stats.get('total', 0)} = {stats['accuracy']:.4f}"
                )
            else:
                print(f"  {key}: {stats}")


def save_outputs(summary, details, output_file=None, detail_file=None):
    if output_file:
        output_path = Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {output_path}")

    if detail_file and details is not None:
        detail_path = Path(detail_file).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            for row in details:
                f.write(json.dumps(row) + "\n")
        print(f"Saved details to {detail_path}")


def eval_mmbench(args):
    ann = {}
    with open(args.question_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["index"])
            options = {}
            for letter in ["A", "B", "C", "D"]:
                value = row.get(letter, "")
                if value is None:
                    continue
                value = str(value).strip()
                if not value or value.lower() == "nan":
                    continue
                options[letter] = value
            gt_letter = str(row.get("answer", "")).strip().upper()
            ann[qid] = {
                "answer": gt_letter,
                "category": row.get("category", "unknown"),
                "options": options,
            }

    preds = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {}
    duplicates_ignored = 0
    for row in preds:
        qid = str(row.get("question_id"))
        round_id = row.get("round_id", 0)
        try:
            round_id = int(round_id)
        except (TypeError, ValueError):
            round_id = 0
        if qid in pred_by_qid:
            old_round = pred_by_qid[qid]["_round_id"]
            if round_id < old_round:
                duplicates_ignored += 1
                row["_round_id"] = round_id
                pred_by_qid[qid] = row
            else:
                duplicates_ignored += 1
        else:
            row["_round_id"] = round_id
            pred_by_qid[qid] = row

    total = 0
    correct = 0
    missing = 0
    unparsable = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for qid, q in ann.items():
        valid_letters = list(q["options"].keys())
        if not valid_letters:
            continue
        total += 1
        category = q["category"] or "unknown"
        by_category[category]["total"] += 1
        pred = pred_by_qid.get(qid)

        if pred is None:
            missing += 1
            if args.detail_file:
                details.append(
                    {
                        "question_id": qid,
                        "category": category,
                        "gold": q["answer"],
                        "pred_letter": None,
                        "pred_text": None,
                        "correct": 0,
                        "missing": 1,
                    }
                )
            continue

        pred_letter = None
        pred_text = pred.get("text", "")
        options_in_pred = pred.get("options", [])
        option_char_in_pred = pred.get("option_char", [])

        pred_letter_in_pred_space = extract_option_letter(
            pred_text, option_char_in_pred if option_char_in_pred else valid_letters, options=options_in_pred
        )
        if pred_letter_in_pred_space and option_char_in_pred and options_in_pred:
            try:
                idx = option_char_in_pred.index(pred_letter_in_pred_space)
                selected_option_text = options_in_pred[idx]
                selected_norm = normalize_text(selected_option_text)
                for letter, option in q["options"].items():
                    if normalize_text(option) == selected_norm:
                        pred_letter = letter
                        break
            except ValueError:
                pass

        if pred_letter is None:
            pred_letter = extract_option_letter(pred_text, valid_letters, options=list(q["options"].values()))

        if pred_letter is None and options_in_pred and option_char_in_pred:
            pred_norm = normalize_text(pred_text)
            for idx, option_text in enumerate(options_in_pred):
                if pred_norm == normalize_text(option_text):
                    selected_norm = normalize_text(option_text)
                    for letter, option in q["options"].items():
                        if normalize_text(option) == selected_norm:
                            pred_letter = letter
                            break
                    if pred_letter:
                        break

        is_correct = pred_letter == q["answer"]
        if pred_letter is None:
            unparsable += 1
        correct += int(is_correct)
        by_category[category]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "category": category,
                    "gold": q["answer"],
                    "pred_letter": pred_letter,
                    "pred_text": pred_text,
                    "correct": int(is_correct),
                    "missing": 0,
                }
            )

    summary = {
        "dataset": "mmbench",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "unparsable_predictions": unparsable,
        "duplicates_ignored": duplicates_ignored,
        "by_category": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_category.items())
        },
    }
    return summary, details if args.detail_file else None


def eval_gqa(args):
    questions_obj = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    if isinstance(questions_obj, list):
        questions = {str(x["question_id"]): x for x in questions_obj if "question_id" in x}
    else:
        questions = {str(k): v for k, v in questions_obj.items()}

    result_path = Path(args.result_file).expanduser().resolve()
    pred_by_qid = {}
    if result_path.suffix == ".json":
        rows = json.load(open(result_path, "r"))
        for row in rows:
            qid = str(row.get("questionId"))
            pred_by_qid[qid] = normalize_gqa_answer(row.get("prediction", ""))
    else:
        rows = load_jsonl(result_path)
        for row in rows:
            qid = str(row.get("question_id"))
            pred_by_qid[qid] = normalize_gqa_answer(row.get("text", ""))

    total = 0
    correct = 0
    missing = 0
    by_structural = defaultdict(lambda: {"correct": 0, "total": 0})
    by_semantic = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for qid, q in questions.items():
        if args.balanced_only and not q.get("isBalanced", False):
            continue

        total += 1
        gold = normalize_gqa_answer(q.get("answer", ""))
        pred = pred_by_qid.get(str(qid))
        struct = q.get("types", {}).get("structural", "unknown")
        sem = q.get("types", {}).get("semantic", "unknown")

        by_structural[struct]["total"] += 1
        by_semantic[sem]["total"] += 1

        if pred is None:
            missing += 1
            is_correct = False
        else:
            is_correct = pred == gold

        correct += int(is_correct)
        by_structural[struct]["correct"] += int(is_correct)
        by_semantic[sem]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(is_correct),
                    "missing": int(pred is None),
                    "structural_type": struct,
                    "semantic_type": sem,
                }
            )

    summary = {
        "dataset": "gqa",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "by_structural_type": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_structural.items())
        },
        "by_semantic_type": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_semantic.items())
        },
    }
    return summary, details if args.detail_file else None


def eval_seed(args):
    result_rows = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(r.get("question_id")): str(r.get("text", "")).strip() for r in result_rows}

    ann_path = Path(args.annotation_file).expanduser().resolve() if args.annotation_file else None
    if ann_path is None or not ann_path.exists():
        summary = {
            "dataset": "seed",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "Annotation file not found; accuracy not computed.",
        }
        return summary, None

    data = json.load(open(ann_path, "r"))
    questions = data.get("questions", [])
    if not questions:
        summary = {
            "dataset": "seed",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "No questions in annotation file; accuracy not computed.",
        }
        return summary, None

    type_id_to_name = {str(v): k for k, v in data.get("question_type", {}).items()}

    total = 0
    correct = 0
    missing = 0
    by_qtype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_dtype = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []

    for q in questions:
        if "answer" not in q:
            continue

        qid = str(q.get("question_id"))
        gold = str(q.get("answer", "")).strip()
        qtype_id = str(q.get("question_type_id", "unknown"))
        qtype = type_id_to_name.get(qtype_id, qtype_id)
        dtype = q.get("data_type", "unknown")

        pred = pred_by_qid.get(qid)
        if pred is None:
            missing += 1
            is_correct = False
        else:
            pred_norm = str(pred).strip()
            if len(gold) == 1 and gold.upper() in string.ascii_uppercase:
                parsed = extract_option_letter(pred_norm, list(string.ascii_uppercase[:5]))
                pred_norm = parsed if parsed else pred_norm
            is_correct = pred_norm == gold

        total += 1
        correct += int(is_correct)
        by_qtype[qtype]["total"] += 1
        by_qtype[qtype]["correct"] += int(is_correct)
        by_dtype[dtype]["total"] += 1
        by_dtype[dtype]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "data_type": dtype,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(is_correct),
                    "missing": int(pred is None),
                }
            )

    summary = {
        "dataset": "seed",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "by_question_type": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_qtype.items())
        },
        "by_data_type": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_dtype.items())
        },
    }
    return summary, details if args.detail_file else None


def _binary_metrics(tp, tn, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def eval_pope(args):
    questions = load_jsonl(Path(args.question_file).expanduser().resolve())
    pred_rows = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(r.get("question_id")): r.get("text", "") for r in pred_rows}

    labels_by_category = {}
    for filename in sorted(os.listdir(args.annotation_dir)):
        if not (filename.startswith("coco_pope_") and filename.endswith(".json")):
            continue
        category = filename[len("coco_pope_") : -len(".json")]
        rows = load_jsonl(Path(args.annotation_dir) / filename)
        labels_by_category[category] = {str(r.get("question_id")): str(r.get("label", "")).strip().lower() for r in rows}

    total = 0
    correct = 0
    missing = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "missing": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "yes": 0, "pred_count": 0})
    details = []

    for q in questions:
        qid = str(q.get("question_id"))
        category = q.get("category", "unknown")
        category_labels = labels_by_category.get(category, {})
        gold_raw = category_labels.get(qid)
        if gold_raw not in {"yes", "no"}:
            continue

        total += 1
        by_category[category]["total"] += 1
        gold = 1 if gold_raw == "yes" else 0

        pred_text = pred_by_qid.get(qid)
        pred_norm = normalize_pope_prediction(pred_text)
        if pred_text is None:
            missing += 1
            by_category[category]["missing"] += 1
        else:
            by_category[category]["pred_count"] += 1
            if pred_norm == "yes":
                by_category[category]["yes"] += 1

        pred = None
        if pred_norm == "yes":
            pred = 1
        elif pred_norm == "no":
            pred = 0

        is_correct = pred is not None and pred == gold
        correct += int(is_correct)
        by_category[category]["correct"] += int(is_correct)

        if pred is not None:
            if pred == 1 and gold == 1:
                by_category[category]["tp"] += 1
            elif pred == 1 and gold == 0:
                by_category[category]["fp"] += 1
            elif pred == 0 and gold == 0:
                by_category[category]["tn"] += 1
            elif pred == 0 and gold == 1:
                by_category[category]["fn"] += 1

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "category": category,
                    "gold": "yes" if gold == 1 else "no",
                    "pred_text": pred_text,
                    "pred_norm": pred_norm,
                    "correct": int(is_correct),
                    "missing": int(pred_text is None),
                }
            )

    overall_metrics = _binary_metrics(
        sum(v["tp"] for v in by_category.values()),
        sum(v["tn"] for v in by_category.values()),
        sum(v["fp"] for v in by_category.values()),
        sum(v["fn"] for v in by_category.values()),
    )

    category_summary = {}
    for category, stats in sorted(by_category.items()):
        m = _binary_metrics(stats["tp"], stats["tn"], stats["fp"], stats["fn"])
        m.update(
            {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": safe_div(stats["correct"], stats["total"]),
                "missing_predictions": stats["missing"],
                "yes_ratio": safe_div(stats["yes"], stats["pred_count"]),
            }
        )
        category_summary[category] = m

    summary = {
        "dataset": "pope",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "binary_metrics": overall_metrics,
        "by_category": category_summary,
    }
    return summary, details if args.detail_file else None


def textvqa_prompt_processor(prompt):
    prompt = str(prompt)
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip().lower()
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            return prompt.split("\n")[1].strip().lower()
        return prompt.split("\n")[0].strip().lower()
    elif len(prompt.split("\n")) >= 1:
        return prompt.split("\n")[0].strip().lower()
    return prompt.strip().lower()


def eval_textvqa(args):
    annotations = json.load(open(args.annotation_file, "r")).get("data", [])
    ann_by_key = {}
    for ann in annotations:
        key = (str(ann.get("image_id")), str(ann.get("question", "")).strip().lower())
        answers = [x.get("answer", "") if isinstance(x, dict) else str(x) for x in ann.get("answers", [])]
        ann_by_key[key] = answers

    preds = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_key = {}
    duplicates_ignored = 0
    for row in preds:
        key = (str(row.get("question_id")), textvqa_prompt_processor(row.get("prompt", "")))
        if key in pred_by_key:
            duplicates_ignored += 1
            continue
        pred_by_key[key] = row.get("text", "")

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    details = []
    missing = 0
    for key, gt_answers in ann_by_key.items():
        pred = pred_by_key.get(key)
        if pred is None:
            missing += 1
            pred = ""
        pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})
        if args.detail_file:
            pred_proc = evaluator.answer_processor(pred)
            unique_scores = evaluator._compute_answer_scores(gt_answers)
            score = unique_scores.get(pred_proc, 0.0)
            details.append(
                {
                    "question_id": key[0],
                    "question": key[1],
                    "pred_answer": pred,
                    "score": score,
                    "missing": int(pred_by_key.get(key) is None),
                }
            )

    accuracy = evaluator.eval_pred_list(pred_list) if pred_list else 0.0
    total = len(pred_list)
    score_sum = accuracy * total

    summary = {
        "dataset": "textvqa",
        "overall": {"correct": score_sum, "total": total, "accuracy": accuracy},
        "missing_predictions": missing,
        "duplicates_ignored": duplicates_ignored,
        "scoring": "vqa_soft",
    }
    return summary, details if args.detail_file else None


def eval_mme(args):
    qpath = Path(args.question_file).expanduser().resolve() if args.question_file else None
    if qpath is None or not qpath.exists():
        summary = {
            "dataset": "mme",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "note": "Question/annotation file not found; accuracy not computed.",
        }
        return summary, None

    questions = load_json_or_jsonl(qpath)
    if isinstance(questions, dict):
        questions = questions.get("questions", [])

    preds = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(r.get("question_id")): r.get("text", "") for r in preds}

    def get_gold(sample):
        for key in ["answer", "gt_answer", "label", "target"]:
            if key in sample and sample[key] is not None:
                return sample[key]
        return None

    total = 0
    correct = 0
    missing = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    details = []
    gold_count = 0

    for idx, q in enumerate(questions):
        qid = str(q.get("question_id", idx))
        gold = get_gold(q)
        if gold is None:
            continue
        gold_count += 1
        category = q.get("category") or q.get("task") or q.get("type") or "unknown"
        pred = pred_by_qid.get(qid)

        if pred is None:
            missing += 1
            is_correct = False
        else:
            gold_norm = normalize_text_loose(gold)
            if gold_norm in {"yes", "no"}:
                pred_norm = normalize_pope_prediction(pred)
                is_correct = pred_norm == gold_norm
            else:
                is_correct = normalize_text_loose(pred) == gold_norm

        total += 1
        correct += int(is_correct)
        by_category[category]["total"] += 1
        by_category[category]["correct"] += int(is_correct)

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "category": category,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(is_correct),
                    "missing": int(pred is None),
                }
            )

    if gold_count == 0:
        summary = {
            "dataset": "mme",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(preds),
            "note": "No gold-answer field found in question file; accuracy not computed.",
        }
        return summary, None

    summary = {
        "dataset": "mme",
        "overall": {"correct": correct, "total": total, "accuracy": safe_div(correct, total)},
        "missing_predictions": missing,
        "by_category": {
            k: {"correct": v["correct"], "total": v["total"], "accuracy": safe_div(v["correct"], v["total"])}
            for k, v in sorted(by_category.items())
        },
    }
    return summary, details if args.detail_file else None


def eval_vizwiz(args):
    preds = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(r.get("question_id")): r.get("text", "") for r in preds}

    ann_path = Path(args.annotation_file).expanduser().resolve() if args.annotation_file else None
    q_path = Path(args.question_file).expanduser().resolve() if args.question_file else None
    if ann_path is None or not ann_path.exists() or q_path is None or not q_path.exists():
        summary = {
            "dataset": "vizwiz",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "Annotation or question file not provided/found; accuracy not computed.",
        }
        return summary, None

    questions = load_jsonl(q_path)
    question_meta = {}
    for q in questions:
        qid = str(q.get("question_id"))
        question_meta[qid] = {
            "image": q.get("image"),
            "question": extract_first_line_question(q.get("text", "")),
        }

    annotations = json.load(open(ann_path, "r"))
    ann_by_key = {}
    for ann in annotations:
        key = (ann.get("image"), str(ann.get("question", "")).strip().lower())
        answers = [a.get("answer", "") if isinstance(a, dict) else str(a) for a in ann.get("answers", [])]
        ann_by_key[key] = {"answers": answers, "answer_type": ann.get("answer_type", "unknown")}

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    details = []
    missing = 0
    unmatched_questions = 0
    by_answer_type = defaultdict(lambda: {"score_sum": 0.0, "total": 0})

    for qid, meta in question_meta.items():
        key = (meta["image"], meta["question"])
        ann = ann_by_key.get(key)
        if ann is None:
            unmatched_questions += 1
            continue

        pred = pred_by_qid.get(qid)
        if pred is None:
            missing += 1
            pred = ""

        gt_answers = ann["answers"]
        pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})

        pred_proc = evaluator.answer_processor(pred)
        unique_scores = evaluator._compute_answer_scores(gt_answers)
        score = unique_scores.get(pred_proc, 0.0)
        answer_type = ann["answer_type"]
        by_answer_type[answer_type]["score_sum"] += score
        by_answer_type[answer_type]["total"] += 1

        if args.detail_file:
            details.append(
                {
                    "question_id": qid,
                    "image": meta["image"],
                    "question": meta["question"],
                    "pred_answer": pred,
                    "score": score,
                    "answer_type": answer_type,
                    "missing": int(pred_by_qid.get(qid) is None),
                }
            )

    if not pred_list:
        summary = {
            "dataset": "vizwiz",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "No overlap between question file and annotation file; accuracy not computed.",
        }
        return summary, None

    accuracy = evaluator.eval_pred_list(pred_list)
    total = len(pred_list)
    score_sum = accuracy * total

    summary = {
        "dataset": "vizwiz",
        "overall": {"correct": score_sum, "total": total, "accuracy": accuracy},
        "missing_predictions": missing,
        "unmatched_questions": unmatched_questions,
        "scoring": "vqa_soft",
        "by_answer_type": {
            k: {
                "correct": v["score_sum"],
                "total": v["total"],
                "accuracy": safe_div(v["score_sum"], v["total"]),
            }
            for k, v in sorted(by_answer_type.items())
        },
    }
    return summary, details if args.detail_file else None


def eval_vqav2(args):
    preds = load_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(r.get("question_id")): r.get("text", "") for r in preds}

    ann_path = Path(args.annotation_file).expanduser().resolve() if args.annotation_file else None
    if ann_path is None or not ann_path.exists():
        summary = {
            "dataset": "vqav2",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "Annotation file not provided/found; accuracy not computed.",
        }
        return summary, None

    ann_json = json.load(open(ann_path, "r"))
    if isinstance(ann_json, dict) and "annotations" in ann_json:
        annotations = ann_json["annotations"]
    elif isinstance(ann_json, list):
        annotations = ann_json
    else:
        summary = {
            "dataset": "vqav2",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "Unsupported annotation format; accuracy not computed.",
        }
        return summary, None

    gt_by_qid = {}
    for ann in annotations:
        qid = str(ann.get("question_id"))
        if "answers" in ann and isinstance(ann["answers"], list):
            gt_answers = [x.get("answer", "") if isinstance(x, dict) else str(x) for x in ann["answers"]]
        elif "answer" in ann:
            gt_answers = [ann["answer"]] * 10
        else:
            continue
        if len(gt_answers) < 10:
            gt_answers = gt_answers + gt_answers[:1] * (10 - len(gt_answers))
        gt_by_qid[qid] = gt_answers[:10]

    if not gt_by_qid:
        summary = {
            "dataset": "vqav2",
            "no_ground_truth": True,
            "overall": {"correct": 0, "total": 0, "accuracy": None},
            "prediction_count": len(pred_by_qid),
            "note": "No valid gold answers found in annotation file.",
        }
        return summary, None

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    details = []
    missing = 0
    for qid, gt_answers in gt_by_qid.items():
        pred = pred_by_qid.get(qid)
        if pred is None:
            missing += 1
            pred = ""
        pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})
        if args.detail_file:
            pred_proc = evaluator.answer_processor(pred)
            unique_scores = evaluator._compute_answer_scores(gt_answers)
            score = unique_scores.get(pred_proc, 0.0)
            details.append(
                {
                    "question_id": qid,
                    "pred_answer": pred,
                    "score": score,
                    "missing": int(pred_by_qid.get(qid) is None),
                }
            )

    accuracy = evaluator.eval_pred_list(pred_list)
    total = len(pred_list)
    score_sum = accuracy * total
    summary = {
        "dataset": "vqav2",
        "overall": {"correct": score_sum, "total": total, "accuracy": accuracy},
        "missing_predictions": missing,
        "scoring": "vqa_soft",
    }
    return summary, details if args.detail_file else None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute benchmark summary.json for multiple eval datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mmbench", "gqa", "seed", "pope", "textvqa", "mme", "vizwiz", "vqav2"])
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--annotation-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--detail-file", type=str, default=None)
    parser.add_argument("--balanced-only", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "mmbench":
        if not args.question_file:
            raise ValueError("--question-file is required for mmbench")
        summary, details = eval_mmbench(args)
    elif args.dataset == "gqa":
        if not args.question_file:
            raise ValueError("--question-file is required for gqa")
        if not args.balanced_only:
            args.balanced_only = True
        summary, details = eval_gqa(args)
    elif args.dataset == "seed":
        summary, details = eval_seed(args)
    elif args.dataset == "pope":
        if not args.question_file:
            raise ValueError("--question-file is required for pope")
        if not args.annotation_dir:
            raise ValueError("--annotation-dir is required for pope")
        summary, details = eval_pope(args)
    elif args.dataset == "textvqa":
        if not args.annotation_file:
            raise ValueError("--annotation-file is required for textvqa")
        summary, details = eval_textvqa(args)
    elif args.dataset == "mme":
        summary, details = eval_mme(args)
    elif args.dataset == "vizwiz":
        summary, details = eval_vizwiz(args)
    elif args.dataset == "vqav2":
        summary, details = eval_vqav2(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print_summary(summary)
    save_outputs(summary, details, output_file=args.output_file, detail_file=args.detail_file)


if __name__ == "__main__":
    main()
