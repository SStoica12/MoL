import json
from pathlib import Path


DEFAULT_VSTAR_CATEGORIES = ["GPT4V-hard", "OCR", "direct_attributes", "relative_position"]
DEFAULT_ANALYSIS_FILE_NAME = "router_analysis_samples.jsonl"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_attention_family(model_name: str):
    if "text-cond-attn-layer-router" in model_name:
        return "attn-layer"
    if "text-cond-attn-token-router" in model_name:
        return "attn-token"
    if "text-cond-attn-patchlayer" in model_name:
        return "patch_layer"
    if "text-cond-attn-hybrid-router" in model_name:
        return "hybrid"
    return None


def first_router_metadata(predictions_path: Path):
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        return None
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metadata = (payload.get("metadata") or {}).get("router")
            if metadata:
                return metadata
    return None


def discover_vstar_attention_models(
    answers_root: Path,
    checkpoint_root: Path,
    analysis_file_name: str = DEFAULT_ANALYSIS_FILE_NAME,
):
    discovered = []
    appendix = []

    for answer_dir in sorted(path for path in answers_root.iterdir() if path.is_dir()):
        model_name = answer_dir.name
        family = infer_attention_family(model_name)
        if family is None:
            continue

        predictions_path = answer_dir / "predictions.jsonl"
        details_path = answer_dir / "details.jsonl"
        summary_path = answer_dir / "summary.json"
        checkpoint_path = checkpoint_root / model_name
        analysis_path = answer_dir / analysis_file_name
        metadata = first_router_metadata(predictions_path)

        issues = []
        if metadata is None:
            issues.append("missing_router_metadata")
        if not checkpoint_path.exists():
            issues.append("missing_checkpoint")
        if not details_path.exists():
            issues.append("missing_details")
        if not summary_path.exists():
            issues.append("missing_summary")

        record = {
            "model_name": model_name,
            "family": family,
            "answer_dir": answer_dir,
            "predictions_path": predictions_path,
            "details_path": details_path,
            "summary_path": summary_path,
            "analysis_path": analysis_path,
            "checkpoint_path": checkpoint_path,
            "selection_mode": metadata.get("selection_mode") if metadata else None,
            "reserve_enabled": bool(
                metadata
                and (
                    "reserve_layer_index" in metadata
                    or "reserve_layer_prob" in metadata
                    or "reserve_layer_prob_summary" in metadata
                )
            ),
            "has_analysis": analysis_path.exists(),
        }
        if issues:
            record["issues"] = issues
            appendix.append(record)
        else:
            discovered.append(record)

    return discovered, appendix

