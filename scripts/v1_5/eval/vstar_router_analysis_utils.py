import json
from pathlib import Path
import re

import torch

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)


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


def load_checkpoint_config(model_path: Path):
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    return load_json(config_path)


def _config_bool(config, key, default=False):
    value = config.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def infer_encoder_type(model_name: str, config: dict | None = None):
    config = config or {}
    if config.get("encoder_type"):
        return str(config["encoder_type"])

    lower = model_name.lower().replace("-", "_")
    if "clip_dino" in lower:
        return "clip_dinov2_txt"
    if "siglip" in lower and "dino" in lower:
        return "siglip_dinov2_txt"
    if "dino" in lower:
        return "dinov2_txt"
    if "siglip" in lower:
        return "siglip"
    return "clip"


def infer_router_kind(encoder_type: str):
    if "dino" in encoder_type:
        return "dino"
    if "siglip" in encoder_type:
        return "siglip"
    return "clip"


def infer_attn_level(model_name: str, config: dict | None = None):
    config = config or {}
    if config.get("attn_level"):
        return str(config["attn_level"])

    lower = model_name.lower().replace("-", "_")
    if "hybrid" in lower:
        return "hybrid"
    if "patch_layer" in lower or "patchlayer" in lower:
        return "patch_layer"
    if "text_cond_attn_layer_router" in lower:
        return "layer"
    return "token"


def infer_num_layers_to_choose(model_name: str, config: dict | None = None):
    config = config or {}
    value = config.get("num_layers_to_choose")
    if value is not None:
        return int(value)
    match = re.search(r"(?:^|[_-])k=(\d+)(?:$|[_-])", model_name)
    if match:
        return int(match.group(1))
    return 1


def router_loader_kwargs(model_path: Path, model_name: str):
    config = load_checkpoint_config(model_path)
    encoder_type = infer_encoder_type(model_name, config)
    text_conditioning = str(config.get("text_conditioning", "attention"))
    mm_use_text_enc = _config_bool(
        config,
        "mm_use_text_enc",
        default=("text-cond" in model_name or "txt" in encoder_type),
    )

    return {
        "encoder_type": encoder_type,
        "mm_use_text_enc": mm_use_text_enc,
        "text_pool": str(config.get("text_pool", "mean")),
        "text_scale": float(config.get("text_scale", 1.0)),
        "text_conditioning": text_conditioning,
        "text_attention_type": str(config.get("text_attention_type", "qk")),
        "text_attention_dim": int(config.get("text_attention_dim", 1024)),
        "attn_level": infer_attn_level(model_name, config),
        "force_reserve_layer": _config_bool(config, "force_reserve_layer", default=False),
        "reserve_layer": int(config.get("reserve_layer", -2)),
        "learnable_gate": _config_bool(config, "learnable_gate", default=("learnable-gate" in model_name)),
        "confidence_gate": _config_bool(config, "confidence_gate", default=False),
        "confidence_gate_threshold": float(config.get("confidence_gate_threshold", 0.6)),
        "confidence_gate_slope": float(config.get("confidence_gate_slope", 10.0)),
        "hybrid_reserve_mode": str(config.get("hybrid_reserve_mode", "local_gate")),
        "hybrid_disagreement_metric": str(config.get("hybrid_disagreement_metric", "kl_local_global")),
        "num_layers_to_choose": infer_num_layers_to_choose(model_name, config),
        "router_alpha": float(config.get("router_alpha", 0.01)),
        "fusion_type": str(config.get("fusion_type", "attn")),
    }


def router_vision_tower(model, router_kind: str):
    if router_kind == "dino":
        return model.get_dino_vision_tower()
    load_model = "siglip" if router_kind == "siglip" else "clip"
    return model.get_model().get_vision_tower(load_model=load_model)


def sanitize_router_text(text):
    if text is None:
        return ""
    text = text.replace(DEFAULT_IM_START_TOKEN, " ")
    text = text.replace(DEFAULT_IM_END_TOKEN, " ")
    text = text.replace(DEFAULT_IMAGE_PATCH_TOKEN, " ")
    return " ".join(text.split())


def prepare_router_text_context(model, input_ids, attention_mask, prompts, router_kind: str):
    vision_tower = router_vision_tower(model, router_kind)
    prompts = [sanitize_router_text(prompt) for prompt in prompts]

    if hasattr(vision_tower, "text_tokenizer") and hasattr(vision_tower, "encode_text"):
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if router_kind == "siglip":
            tokenizer_kwargs.update({"padding": "max_length", "max_length": 64})
        tokens = vision_tower.text_tokenizer(prompts, **tokenizer_kwargs)
        text_input_ids = tokens["input_ids"].to(device=vision_tower.device)
        text_attention_mask = tokens.get("attention_mask", None)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device=vision_tower.device)
        with torch.no_grad():
            text_embeds = vision_tower.encode_text(text_input_ids, text_attention_mask)
        return text_embeds, text_attention_mask

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    text_attention_mask = attention_mask.bool() & (input_ids != IMAGE_TOKEN_INDEX)
    pad_id = getattr(model.config, "pad_token_id", 0)
    safe_input_ids = input_ids.clone()
    safe_input_ids[safe_input_ids == IMAGE_TOKEN_INDEX] = pad_id
    text_embeds = model.get_model().embed_tokens(safe_input_ids)
    return text_embeds, text_attention_mask


def run_vision_tower_for_router_analysis(vision_tower_wrapper, image_inputs, router_kind: str):
    if router_kind == "dino":
        image_forward_outs = vision_tower_wrapper(
            image_inputs,
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden_states = torch.stack(list(image_forward_outs.hidden_states))
        return image_forward_outs, hidden_states

    image_forward_outs = vision_tower_wrapper.vision_tower(
        image_inputs,
        output_hidden_states=True,
        output_attentions=True,
    )
    hidden_states = torch.stack(list(image_forward_outs.hidden_states))[1:]
    return image_forward_outs, hidden_states


def infer_attention_family(model_name: str):
    lower = model_name.lower().replace("-", "_")
    if "dino" in lower:
        if "hybrid" in lower:
            return "hybrid"
        if "patch_layer" in lower or "patchlayer" in lower:
            return "patch_layer"
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
