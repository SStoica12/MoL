import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.eval.model_vqa_loader import create_data_loader
from llava.eval.router_logging import (
    clear_router_selection_metadata,
    get_router_selection_metadata,
)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

from vstar_router_analysis_utils import (
    DEFAULT_ANALYSIS_FILE_NAME,
    infer_attention_family,
    load_json,
    load_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-sample router receptive-field and reserve-shift metrics for V*Bench."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file",
        type=str,
        default="playground/data/eval/vstar_bench/test_questions.jsonl",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="playground/data/eval/vstar_bench",
    )
    parser.add_argument("--detail-file", type=str, required=True)
    parser.add_argument("--summary-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def move_images_to_device(image_tensor, device, dtype):
    if isinstance(image_tensor, list):
        return [image.to(device=device, dtype=dtype, non_blocking=True) for image in image_tensor]
    return image_tensor.to(device=device, dtype=dtype, non_blocking=True)


def build_distance_matrix(vision_tower_wrapper):
    side = int(vision_tower_wrapper.num_patches_per_side)
    patch_size = int(vision_tower_wrapper.config.patch_size)
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(side, dtype=torch.float32),
            torch.arange(side, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    return torch.cdist(coords, coords, p=2) * patch_size


def normalize_layer_prob_mass(metadata):
    num_layers = int(metadata.get("num_layers_available") or 0)
    probs = torch.zeros(num_layers, dtype=torch.float32)
    for layer_idx, value in (metadata.get("selected_layer_prob_mass") or {}).items():
        probs[int(layer_idx)] = float(value)
    total = float(probs.sum().item())
    if total > 0:
        probs /= total
    return probs


def compute_rf_by_layer(attentions, distance_matrix):
    # attentions: sequence of [B, H, N, N] vision self-attention maps, one per layer
    if attentions is None:
        raise ValueError("Expected vision self-attention maps, but got None.")
    if isinstance(attentions, torch.Tensor):
        layer_attentions = attentions.to(torch.float32)
    else:
        layer_attentions = torch.stack([attn.to(torch.float32) for attn in attentions], dim=0)

    if layer_attentions.ndim != 5:
        raise ValueError(
            "Expected stacked attention maps with shape [L, B, H, N, N], got "
            f"{tuple(layer_attentions.shape)}"
        )
    if layer_attentions.shape[-1] <= 1 or layer_attentions.shape[-2] <= 1:
        raise ValueError("Expected CLS + patch tokens in the attention maps.")

    attn = layer_attentions[..., 1:, 1:]
    if attn.shape[-1] != distance_matrix.shape[0] or attn.shape[-2] != distance_matrix.shape[1]:
        raise ValueError(
            "Vision self-attention size does not match the vision tower patch grid: "
            f"{tuple(attn.shape[-2:])} vs {tuple(distance_matrix.shape)}"
        )
    attn = attn + 1e-32
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-32)

    distance_matrix = distance_matrix.to(device=attn.device, dtype=attn.dtype).view(
        1,
        1,
        1,
        distance_matrix.shape[0],
        distance_matrix.shape[1],
    )
    mean_distances = attn * distance_matrix
    mean_distances = mean_distances.sum(dim=-1)
    mean_distances = mean_distances.mean(dim=-1)
    mean_distances = mean_distances.mean(dim=-1)

    if mean_distances.shape[1] != 1:
        raise ValueError(
            "Collector expects batch_size=1 when computing per-sample receptive fields, got "
            f"{mean_distances.shape[1]}"
        )
    return mean_distances[:, 0]


def reduce_optional_mean(value):
    if value is None:
        return None
    if isinstance(value, dict):
        maybe_mean = value.get("mean")
        return None if maybe_mean is None else float(maybe_mean)
    return float(value)


def select_hidden_tokens(hidden_states, strategy):
    if strategy in ("default", "cls_patch"):
        return hidden_states
    if strategy == "patch":
        return hidden_states[:, 1:, :]
    if strategy == "cls":
        return hidden_states[:, :1, :]
    raise ValueError(f"Unexpected vision feature select strategy: {strategy}")


def compute_reserve_metrics(hidden_states, analysis, layer_prob_mass, strategy):
    reserve_idx = analysis.get("reserve_layer_index")
    reserve_features = analysis.get("pre_projector_reserve_features")
    final_features = analysis.get("pre_projector_features")
    if reserve_idx is None or reserve_features is None or final_features is None:
        return {
            "reserve_shift_l2": None,
            "reserve_shift_ratio": None,
            "reserve_cosine": None,
            "reserve_displacement_by_layer": None,
            "contribution_weighted_layer_effect": None,
        }

    final_features = final_features[0].to(torch.float32)
    reserve_features = reserve_features[0].to(torch.float32)
    delta = (final_features - reserve_features).norm(dim=-1).mean()
    reserve_norm = reserve_features.norm(dim=-1).mean().clamp(min=1e-8)
    cosine = F.cosine_similarity(final_features, reserve_features, dim=-1).mean()

    hidden_states = hidden_states[:, 0].to(torch.float32)  # [L, N, H]
    selected_hidden_states = select_hidden_tokens(hidden_states, strategy)
    reserve_state = selected_hidden_states[reserve_idx]
    displacement = (selected_hidden_states - reserve_state.unsqueeze(0)).norm(dim=-1).mean(dim=-1)
    contribution = displacement * layer_prob_mass.to(displacement.device)

    return {
        "reserve_shift_l2": float(delta.item()),
        "reserve_shift_ratio": float((delta / reserve_norm).item()),
        "reserve_cosine": float(cosine.item()),
        "reserve_displacement_by_layer": displacement.cpu().tolist(),
        "contribution_weighted_layer_effect": contribution.cpu().tolist(),
    }


def main():
    args = parse_args()

    model_path = Path(args.model_path).expanduser()
    model_name = model_path.name
    family = infer_attention_family(model_name)
    if family is None:
        raise ValueError(f"Model is not an attention-router family: {model_name}")

    output_file = (
        Path(args.output_file).expanduser()
        if args.output_file
        else Path(args.detail_file).expanduser().resolve().parent / DEFAULT_ANALYSIS_FILE_NAME
    )
    summary_file = (
        Path(args.summary_file).expanduser()
        if args.summary_file
        else Path(args.detail_file).expanduser().resolve().parent / "summary.json"
    )

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        args.model_base,
        get_model_name_from_path(str(model_path)),
    )
    model.eval()

    router = getattr(model, "router", None)
    if router is None or not hasattr(router, "analyze"):
        raise ValueError(f"Model does not expose a compatible router analysis path: {model_name}")
    if getattr(router, "text_conditioning", None) not in ("attention", "attn"):
        raise ValueError(f"Model is not an attention-conditioned router: {model_name}")

    questions = load_jsonl(Path(args.question_file).expanduser().resolve())
    if args.max_samples is not None:
        questions = questions[: args.max_samples]
    details = {
        str(row["question_id"]): row
        for row in load_jsonl(Path(args.detail_file).expanduser().resolve())
    }
    summary = load_json(summary_file.resolve())

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        conv_mode=args.conv_mode,
        num_workers=args.num_workers,
    )

    vision_tower_wrapper = model.get_model().get_vision_tower()
    vision_tower = vision_tower_wrapper.vision_tower
    distance_matrix = build_distance_matrix(vision_tower_wrapper).to(
        device=vision_tower_wrapper.device,
        dtype=torch.float32,
    )
    feature_strategy = getattr(router, "vision_feature_select_strategy", "cls_patch")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        iterator = zip(data_loader, questions)
        for (input_ids, image_tensor, image_sizes, prompt), question in tqdm(
            iterator,
            total=len(questions),
            desc=f"Collecting router analysis for {model_name}",
        ):
            question_id = str(question["question_id"])
            detail = details.get(question_id)
            if detail is None:
                raise KeyError(f"Missing details row for question_id={question_id}")
            if image_tensor is None:
                raise ValueError(f"Expected an image for V*Bench sample {question_id}")

            input_ids = input_ids.to(device="cuda", non_blocking=True)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            with torch.inference_mode():
                clear_router_selection_metadata(model)
                text_embeds, text_attention_mask = model.prepare_router_text_context(
                    input_ids,
                    attention_mask=attention_mask,
                    clip_texts=[prompt],
                )
                image_inputs = move_images_to_device(
                    image_tensor,
                    vision_tower_wrapper.device,
                    vision_tower_wrapper.dtype,
                )
                image_forward_outs = vision_tower(
                    image_inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                hidden_states = torch.stack(list(image_forward_outs.hidden_states))[1:]
                analysis = router.analyze(
                    hidden_states,
                    text_embeds=text_embeds,
                    text_attention_mask=text_attention_mask,
                )
                metadata = get_router_selection_metadata(model).get("router", {})

            if not metadata:
                raise RuntimeError(f"Router analysis did not populate metadata for {question_id}")

            layer_prob_mass = normalize_layer_prob_mass(metadata)
            rf_by_layer = compute_rf_by_layer(
                image_forward_outs.attentions,
                distance_matrix,
            ).cpu()
            sampled_rf_contribution = layer_prob_mass * rf_by_layer
            rf_routed = float(torch.dot(layer_prob_mass, rf_by_layer).item())
            rf_uniform = float(rf_by_layer.mean().item())
            reserve_metrics = compute_reserve_metrics(
                hidden_states=hidden_states,
                analysis=analysis,
                layer_prob_mass=layer_prob_mass,
                strategy=feature_strategy,
            )

            category = detail["category"]
            category_summary = summary.get("by_category", {}).get(category, {})
            row = {
                "question_id": question_id,
                "category": category,
                "correct": int(detail["correct"]),
                "family": family,
                "model_name": model_name,
                "selection_mode": metadata.get("selection_mode"),
                "num_layers_available": metadata.get("num_layers_available"),
                "num_layers_to_choose": metadata.get("num_layers_to_choose"),
                "selected_layers_shape": metadata.get("selected_layers_shape"),
                "reserve_layer_index": metadata.get("reserve_layer_index"),
                "num_routed_patches": metadata.get("num_routed_patches"),
                "layer_prob_mass": layer_prob_mass.tolist(),
                "selected_layer_counts": metadata.get("selected_layer_counts", {}),
                "selected_layer_frequency": metadata.get("selected_layer_frequency", {}),
                "rf_by_layer": rf_by_layer.tolist(),
                "sampled_rf_contribution_by_layer": sampled_rf_contribution.tolist(),
                "rf_routed": rf_routed,
                "rf_uniform": rf_uniform,
                "rf_routed_minus_uniform": rf_routed - rf_uniform,
                "rf_metric": "cl_vs_mim_self_attention_distance",
                "reserve_mean": reduce_optional_mean(
                    metadata.get("reserve_layer_prob_summary", metadata.get("reserve_layer_prob"))
                ),
                "hybrid_disagreement_mean": reduce_optional_mean(
                    metadata.get("hybrid_disagreement_summary", metadata.get("hybrid_disagreement"))
                ),
                "model_overall_accuracy": summary.get("overall", {}).get("accuracy"),
                "model_category_accuracy": category_summary.get("accuracy"),
                "reserve_enabled": bool(
                    metadata.get("reserve_layer_index") is not None
                    or metadata.get("reserve_layer_prob") is not None
                    or metadata.get("reserve_layer_prob_summary") is not None
                ),
            }
            row.update(reserve_metrics)
            fout.write(json.dumps(row) + "\n")

    print(f"Saved router analysis to {output_file.resolve()}")


if __name__ == "__main__":
    main()
