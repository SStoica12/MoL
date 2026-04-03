import argparse
import json
import re
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
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_prediction_label(pred_text):
    if pred_text is None:
        return None

    text = str(pred_text).strip()
    if not text:
        return None

    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    if compact in {"0", "1", "2"}:
        return compact

    prefix_match = re.search(
        r"^\s*(?:the\s+answer\s+is|answer|response|prediction|output)?\s*[:\-]?\s*"
        r"(yes|no|uncertain|unknown|unsure|unclear|true|false)\b",
        text,
        flags=re.IGNORECASE,
    )
    if prefix_match:
        token = prefix_match.group(1).lower()
        if token in {"yes", "true"}:
            return "1"
        if token in {"no", "false"}:
            return "0"
        return "2"

    uncertain_phrases = [
        "cannot determine",
        "cant determine",
        "can t determine",
        "cannot be determined",
        "not enough information",
        "not enough info",
        "impossible to tell",
        "unable to determine",
        "hard to tell",
        "not sure",
        "unsure",
        "uncertain",
        "unclear",
        "unknown",
    ]
    if normalized in uncertain_phrases or any(phrase in normalized for phrase in uncertain_phrases):
        return "2"

    yes_match = re.search(r"\byes\b", normalized)
    no_match = re.search(r"\bno\b", normalized)
    if yes_match and not no_match:
        return "1"
    if no_match and not yes_match:
        return "0"

    return None


def pair_key(sample):
    return "_".join(
        [
            str(sample["category"]),
            str(sample["subcategory"]),
            str(sample["set_id"]),
            str(sample["raw_question_id"]),
        ]
    )


def figure_key(sample):
    return "_".join(
        [
            str(sample["category"]),
            str(sample["subcategory"]),
            str(sample["set_id"]),
            str(sample["figure_id"]),
        ]
    )


def question_key(sample):
    return "_".join(
        [
            str(sample["category"]),
            str(sample["subcategory"]),
            str(sample["set_id"]),
            str(sample["figure_id"]),
            str(sample["raw_question_id"]),
        ]
    )


def assign_correctness(samples):
    for sample in samples:
        correctness = str(sample["correctness"])
        if sample["category"] == "VS" and int(sample["figure_id"]) == 0:
            sample["correct"] = 1 if correctness in {"1", "2"} else 0
        else:
            sample["correct"] = 1 if correctness == "1" else 0
    return samples


def get_eval_fig(data):
    eval_fig_dict = {}
    for row in data:
        if row["category"] == "VS" and str(row["figure_id"]) == "0":
            continue
        key = figure_key(row)
        correct, total = eval_fig_dict.get(key, (0, 0))
        eval_fig_dict[key] = (correct + row["correct"], total + 1)

    stats = {
        "total": len(eval_fig_dict),
        "correct": 0,
        "wrong": 0,
        "inconsistent": 0,
        "score": 0.0,
    }
    for correct, total in eval_fig_dict.values():
        if correct == total:
            stats["correct"] += 1
        elif correct == 0:
            stats["wrong"] += 1
        else:
            stats["inconsistent"] += 1
        stats["score"] += float(correct) / float(total)
    stats["score"] = safe_div(stats["score"], stats["total"])
    return stats


def get_eval_all(data):
    eval_all_dict = {}
    eval_all_stat = {"LH": 0, "VI": 0, "Mix": 0}

    for row in data:
        eval_all_dict[question_key(row)] = row["correct"]

        if str(row["category"]) == "VD":
            if str(row["figure_id"]) == "0":
                if str(row["correctness"]) in {"0", "2"}:
                    eval_all_stat["VI"] += 1
            else:
                if str(row["correctness"]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(row["correctness"]) == "2":
                    eval_all_stat["VI"] += 1
        else:
            if str(row["visual_input"]) == "0":
                if str(row["correctness"]) == "0":
                    eval_all_stat["LH"] += 1
            else:
                if str(row["correctness"]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(row["correctness"]) == "2":
                    eval_all_stat["VI"] += 1

    eval_all_stat["total"] = len(eval_all_dict)
    eval_all_stat["correct"] = sum(eval_all_dict.values())
    eval_all_stat["wrong"] = eval_all_stat["total"] - eval_all_stat["correct"]
    return eval_all_stat


def get_eval_pair_all(data):
    orig_correctness = {}
    for row in data:
        if str(row["figure_id"]) == "0":
            orig_correctness[pair_key(row)] = row["correctness"]

    eval_pair_dict = {}
    analysis_pair_dict = {}
    total_q = 0
    lh_counter = 0
    vi_counter = 0
    both_counter = 0

    for row in data:
        key = pair_key(row)
        correct, total = eval_pair_dict.get(key, (0, 0))
        eval_pair_dict[key] = (correct + row["correct"], total + 1)
        total_q += 1

        analysis = (0, 0)
        if str(row["figure_id"]) == "0":
            if str(row["category"]) == "VD":
                if str(row["correctness"]) in {"0", "2"}:
                    analysis = (0, 1)
            else:
                if str(row["correctness"]) == "0":
                    analysis = (1, 0)
        else:
            orig_c = orig_correctness[key]
            if str(row["category"]) == "VD":
                if str(orig_c) == "1" and str(row["correctness"]) == "0":
                    if str(row["same"]) == "1":
                        analysis = (1, 1)
                    else:
                        analysis = (0, 1)
                elif str(orig_c) == "1" and str(row["correctness"]) == "2":
                    analysis = (0, 1)
                elif str(row["correctness"]) in {"0", "2"}:
                    analysis = (0, 1)
            else:
                if str(orig_c) == "0":
                    if str(row["correctness"]) == "0" and str(row["same"]) == "1":
                        analysis = (1, 0)
                    elif str(row["correctness"]) == "0":
                        analysis = (1, 1)
                    elif str(row["correctness"]) == "2":
                        analysis = (1, 1)
                elif str(orig_c) == "2":
                    if str(row["correctness"]) in {"0", "2"}:
                        analysis = (0, 1)
                else:
                    if str(row["correctness"]) == "2":
                        analysis = (0, 1)
                    elif str(row["correctness"]) == "0":
                        if str(row["visual_input"]) == "1":
                            analysis = (0, 1)
                        elif str(row["visual_input"]) == "2":
                            if str(row["same"]) == "1":
                                analysis = (1, 0)
                            else:
                                analysis = (0, 1)
                        else:
                            raise AssertionError("Unexpected visual_input value")

        if analysis[0] > 0 and analysis[1] > 0:
            both_counter += 1
        elif analysis[0] > 0:
            lh_counter += 1
        elif analysis[1] > 0:
            vi_counter += 1

        prev = analysis_pair_dict.get(key, (0, 0))
        analysis_pair_dict[key] = (prev[0] + analysis[0], prev[1] + analysis[1])

    stats = {
        "total": len(eval_pair_dict),
        "total_q": total_q,
        "correct": 0,
        "wrong": 0,
        "LH": 0,
        "VI": 0,
        "Mix": 0,
        "LH_cg": lh_counter,
        "VI_cg": vi_counter,
        "Mix_cg": both_counter,
    }

    for key, value in eval_pair_dict.items():
        analysis = analysis_pair_dict[key]
        if value[0] == value[1]:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
        if analysis[0] > 0 and analysis[1] > 0:
            stats["Mix"] += 1
        elif analysis[0] > 0:
            stats["LH"] += 1
        elif analysis[1] > 0:
            stats["VI"] += 1

    return stats


def get_eval_pair_subset(data, hard: bool):
    pair_stats = {}
    total_q = 0
    target_visual_input = "2" if hard else None
    for row in data:
        if hard and str(row["visual_input"]) != "2":
            continue
        if not hard and str(row["visual_input"]) == "2":
            continue
        key = pair_key(row)
        correct, total = pair_stats.get(key, (0, 0))
        pair_stats[key] = (correct + row["correct"], total + 1)
        total_q += 1

    stats = {"total": len(pair_stats), "total_q": total_q, "correct": 0, "wrong": 0}
    for correct, total in pair_stats.values():
        if correct == total:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
    return stats


def yes_ratio_stats(data):
    yes_gt = [int(row["label"]) for row in data]
    yes_pred = [int(int(row["correct"]) == int(row["label"])) for row in data]
    false_samples = [row for row in data if int(row["correct"]) == 0]
    fp = [int(row["label"]) for row in false_samples]
    return {
        "diff": safe_div(sum(yes_pred), len(yes_pred)) - safe_div(sum(yes_gt), len(yes_gt)),
        "fp": safe_div(len(fp) - sum(fp), len(fp)),
    }


def summarize(samples):
    data_vd = [row for row in samples if row["category"] == "VD"]
    data_vs = [row for row in samples if row["category"] == "VS"]

    all_questions = get_eval_all(samples)
    question_vd = get_eval_all(data_vd)
    question_vs = get_eval_all(data_vs)

    all_pairs = get_eval_pair_all(samples)
    all_pairs_vd = get_eval_pair_all(data_vd)
    all_pairs_vs = get_eval_pair_all(data_vs)

    easy_pairs = get_eval_pair_subset(samples, hard=False)
    hard_pairs = get_eval_pair_subset(samples, hard=True)
    easy_pairs_vd = get_eval_pair_subset(data_vd, hard=False)
    hard_pairs_vd = get_eval_pair_subset(data_vd, hard=True)
    easy_pairs_vs = get_eval_pair_subset(data_vs, hard=False)
    hard_pairs_vs = get_eval_pair_subset(data_vs, hard=True)

    fig_all = get_eval_fig(samples)
    fig_vd = get_eval_fig(data_vd)
    fig_vs = get_eval_fig(data_vs)

    bias = yes_ratio_stats(samples)

    summary = {
        "dataset": "hallusionbench",
        "overall": {
            "correct": all_questions["correct"],
            "total": all_questions["total"],
            "accuracy": safe_div(all_questions["correct"], all_questions["total"]),
        },
        "leaderboard": {
            "question_pair_accuracy": safe_div(all_pairs["correct"], all_pairs["total"]),
            "figure_accuracy": safe_div(fig_all["correct"], fig_all["total"]),
            "easy_question_accuracy": safe_div(easy_pairs["correct"], easy_pairs["total"]),
            "hard_question_accuracy": safe_div(hard_pairs["correct"], hard_pairs["total"]),
            "question_accuracy": safe_div(all_questions["correct"], all_questions["total"]),
        },
        "question_accuracy": {
            "overall": all_questions,
            "VD": question_vd,
            "VS": question_vs,
        },
        "question_pair_accuracy": {
            "overall": all_pairs,
            "easy": easy_pairs,
            "hard": hard_pairs,
            "VD": {
                "overall": all_pairs_vd,
                "easy": easy_pairs_vd,
                "hard": hard_pairs_vd,
            },
            "VS": {
                "overall": all_pairs_vs,
                "easy": easy_pairs_vs,
                "hard": hard_pairs_vs,
            },
        },
        "figure_accuracy": {
            "overall": fig_all,
            "VD": fig_vd,
            "VS": fig_vs,
        },
        "bias_stats": bias,
        "error_breakdown": {
            "LH_pct": safe_div(all_pairs["LH_cg"], all_pairs["LH_cg"] + all_pairs["VI_cg"] + all_pairs["Mix_cg"]),
            "VI_pct": safe_div(all_pairs["VI_cg"], all_pairs["LH_cg"] + all_pairs["VI_cg"] + all_pairs["Mix_cg"]),
            "Mix_pct": safe_div(all_pairs["Mix_cg"], all_pairs["LH_cg"] + all_pairs["VI_cg"] + all_pairs["Mix_cg"]),
        },
    }
    return summary


def print_summary(summary, missing, unparsable):
    leaderboard = summary["leaderboard"]
    print(
        "Question accuracy: "
        f"{summary['overall']['correct']}/{summary['overall']['total']} = {summary['overall']['accuracy']:.4f}"
    )
    print(f"Missing predictions: {missing}")
    print(f"Unparsable predictions: {unparsable}")
    print(
        "Leaderboard: "
        f"qAcc={leaderboard['question_pair_accuracy']:.4f}, "
        f"fAcc={leaderboard['figure_accuracy']:.4f}, "
        f"easy_aAcc={leaderboard['easy_question_accuracy']:.4f}, "
        f"hard_aAcc={leaderboard['hard_question_accuracy']:.4f}, "
        f"aAcc={leaderboard['question_accuracy']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate HallusionBench predictions.")
    parser.add_argument("--question-file", type=str, required=True, help="Converted HallusionBench jsonl file.")
    parser.add_argument("--result-file", type=str, required=True, help="Model prediction jsonl file.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional summary json output path.")
    parser.add_argument("--detail-file", type=str, default=None, help="Optional per-sample detail jsonl path.")
    parser.add_argument(
        "--export-result-file",
        type=str,
        default=None,
        help="Optional merged HallusionBench-style JSON output with model predictions.",
    )
    args = parser.parse_args()

    questions = load_json_or_jsonl(Path(args.question_file).expanduser().resolve())
    results = load_json_or_jsonl(Path(args.result_file).expanduser().resolve())
    pred_by_qid = {str(row.get("question_id")): row for row in results}

    merged = []
    details = []
    missing = 0
    unparsable = 0

    orig_predictions = {}
    for sample in questions:
        if str(sample["figure_id"]) == "0":
            row = pred_by_qid.get(str(sample["question_id"]))
            if row is None:
                continue
            parsed = parse_prediction_label(row.get("text", ""))
            orig_predictions[pair_key(sample)] = parsed

    for sample in questions:
        pred_row = pred_by_qid.get(str(sample["question_id"]))
        pred_text = "" if pred_row is None else pred_row.get("text", "")
        parsed_label = parse_prediction_label(pred_text)
        if pred_row is None:
            missing += 1
        elif parsed_label is None:
            unparsable += 1

        if pred_row is None:
            correctness = "0"
        elif parsed_label is None:
            correctness = "0"
        elif parsed_label == str(sample["label"]):
            correctness = "1"
        elif parsed_label == "2":
            correctness = "2"
        else:
            correctness = "0"

        orig_label = orig_predictions.get(pair_key(sample))
        if parsed_label is not None and orig_label is not None:
            same = "1" if parsed_label == orig_label else "0"
        else:
            same = "0"
            if str(sample["figure_id"]) == "0" and parsed_label is not None:
                same = "1"

        merged_row = dict(sample)
        merged_row["model_prediction"] = pred_text
        merged_row["prediction_label"] = parsed_label
        merged_row["correctness"] = correctness
        merged_row["same"] = same
        merged.append(merged_row)

    merged = assign_correctness(merged)
    summary = summarize(merged)
    summary["missing_predictions"] = missing
    summary["unparsable_predictions"] = unparsable

    if args.detail_file:
        for row in merged:
            details.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "visual_input": row["visual_input"],
                    "set_id": row["set_id"],
                    "figure_id": row["figure_id"],
                    "raw_question_id": row["raw_question_id"],
                    "label": row["label"],
                    "prediction_label": row["prediction_label"],
                    "correctness": row["correctness"],
                    "correct": row["correct"],
                    "same": row["same"],
                    "pred_text": row["model_prediction"],
                }
            )

    print_summary(summary, missing=missing, unparsable=unparsable)

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

    if args.export_result_file:
        export_path = Path(args.export_result_file).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_rows = []
        for row in merged:
            export_rows.append(
                {
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "visual_input": row["visual_input"],
                    "set_id": row["set_id"],
                    "figure_id": row["figure_id"],
                    "sample_note": row.get("sample_note"),
                    "question_id": row["raw_question_id"],
                    "question": row["question"],
                    "gt_answer_details": row.get("gt_answer_details"),
                    "gt_answer": row["label"],
                    "filename": row.get("filename"),
                    "model_prediction": row["model_prediction"],
                    "prediction_label": row["prediction_label"],
                    "correctness": row["correctness"],
                    "same": row["same"],
                    "correct": row["correct"],
                }
            )
        with open(export_path, "w") as f:
            json.dump(export_rows, f, indent=2)
        print(f"Saved merged results to {export_path}")


if __name__ == "__main__":
    main()