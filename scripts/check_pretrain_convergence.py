"""
Quick convergence check: pick one random training example and run inference
through the pretrain checkpoint, then print the model output.
"""

import argparse
import json
import os
import random
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model_router, load_pretrained_model_router_not_fine_tuned
from llava.utils import disable_torch_init


DATA_PATH = "/shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER = "/shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/images"
MODEL_PATH = "/shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-pretrain_text-cond-attn-token-router_clip_dino_hybrid_k=1_normalized_attn"
MODEL_BASE = "lmsys/vicuna-13b-v1.5"


def load_example(data_path: str, seed: int | None = None) -> dict:
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    example = rng.choice(data)
    return example

def build_prompt(example: dict, model_config, conv_mode: str) -> tuple[str, str]:
    """Return (question_text, full_prompt)."""
    convs = example.get("conversations", [])
    if convs:
        human_turn = next((c["value"] for c in convs if c["from"] == "human"), "")
        question = human_turn
    else:
        question = example.get("text", "Describe this image.")

    image_file = example.get("image") or (example.get("images") or [None])[0]
    has_image = image_file is not None and str(image_file).strip() != ""

    if has_image:
        replace_token = DEFAULT_IMAGE_TOKEN
        if model_config.mm_use_im_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if DEFAULT_IMAGE_TOKEN not in question:
            question = replace_token + "\n" + question
        elif model_config.mm_use_im_start_end:
            question = question.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return question, conv.get_prompt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--image-folder", default=IMAGE_FOLDER)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--model-base", default=MODEL_BASE)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_path} ...")
    example = load_example(args.data_path, seed=args.seed)
    print(f"\n--- Sampled example id: {example.get('id', '?')} ---")

    image_file = example.get("image") or (example.get("images") or [None])[0]
    has_image = image_file is not None and str(image_file).strip() != ""
    if has_image:
        image_path = image_file if os.path.isabs(image_file) else os.path.join(args.image_folder, image_file)
        print(f"Image: {image_path}")
    else:
        print("Image: (none)")

    ground_truth = next(
        (c["value"] for c in example.get("conversations", []) if c["from"] == "gpt"),
        example.get("caption", "(no ground truth)"),
    )
    print(f"Ground truth: {ground_truth[:200]}")

    print("\nLoading model (this may take a while for 13B) ...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor_dino, image_processor_clip, _ = load_pretrained_model_router_not_fine_tuned(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        num_layers_to_choose=1,
        router_alpha=1e-2,
        encoder_type="clip-dinov2_txt",
        mm_use_text_enc=True,
        text_pool="mean",
        text_scale=1.0,
        text_conditioning="attention",
        text_attention_type="qk",
        text_attention_dim=1024,
        attn_level="hybrid",
        force_reserve_layer=True,
        reserve_layer=-2,
        learnable_gate=False,
        confidence_gate=False,
        confidence_gate_threshold=0.6,
        confidence_gate_slope=10.0,
        hybrid_disagreement_metric="kl_local_global",
        hybrid_reserve_mode="disagreement_gate",
        fusion_type="attn",
    )

    # verify_loaded_model(model, args.model_path)

    question, prompt = build_prompt(example, model.config, args.conv_mode)
    print(f"\nQuestion: {question[:300]}")

    if has_image:
        image = Image.open(image_path).convert("RGB")
        images = [image]
        image_sizes = [image.size]
        image_tensor_clip = process_images(images, image_processor_clip, model.config) if image_processor_clip else None
        image_tensor_dino = process_images(images, image_processor_dino, model.config) if image_processor_dino else None
    else:
        image_sizes = None
        image_tensor_clip = None
        image_tensor_dino = None

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if image_tensor_clip is not None:
        image_tensor_clip = image_tensor_clip.to(dtype=torch.float16, device=device)
    if image_tensor_dino is not None:
        image_tensor_dino = image_tensor_dino.to(dtype=torch.float16, device=device)

    print("\nRunning inference ...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor_clip,
            dino_images=image_tensor_dino,
            image_sizes=image_sizes,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            encoder_type="clip-dinov2_txt",
            prompts=[question],
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("\n" + "=" * 60)
    print(f"MODEL OUTPUT:\n{output}")
    print("=" * 60)
    print(f"GROUND TRUTH:\n{ground_truth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
