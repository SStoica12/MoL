import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import glob
import os

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def verify_loaded_model(
    model,
    model_path: str,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    check_router_dtype: torch.dtype = torch.bfloat16,
    strict_key_coverage: bool = True,
    max_print_mismatches: int = 5,
):
    """
    Verifies that in-memory model weights match safetensor shard weights.

    - Compare in disk dtype (cast live -> disk dtype).
    - CPU compare.
    - Optional: ensure all shard keys exist in model (strict_key_coverage).
    - Validate router.weights dtype (bf16 by default).
    """
    rank0_print(f"\n--- Starting Deep Verification against safetensor shards in: {model_path} ---")

    live_state_dict = model.state_dict()

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_files:
        rank0_print(f"⚠️  No .safetensors shards found in {model_path}. Skipping verification.")
        return {}

    # --- Router dtype check ---
    router_key_candidates = [k for k in live_state_dict.keys() if "router" in k]
    if len(router_key_candidates) == 0:
        rank0_print("⚠️  Router dtype check: could not find a key ending with 'router.weights' in model.state_dict().")
    else:
        for rk in router_key_candidates:
            dt = live_state_dict[rk].dtype
            if dt != check_router_dtype:
                rank0_print(f"❌ Router dtype mismatch for {rk}: live dtype={dt}, expected={check_router_dtype}")
            else:
                rank0_print(f"✅ Router dtype OK for {rk}: {dt}")

    total_keys_checked = 0
    mismatches = []
    missing_in_live = []

    for shard_path in shard_files:
        rank0_print(f"Checking shard: {shard_path} ...")
        from safetensors.torch import load_file
        disk_weights = load_file(shard_path)

        for key, disk_tensor in disk_weights.items():
            if key not in live_state_dict:
                if strict_key_coverage:
                    missing_in_live.append(key)
                continue

            live_tensor = live_state_dict[key]

            live_cpu = live_tensor.detach().cpu()
            disk_cpu = disk_tensor.detach().cpu()

            # Cast LIVE -> DISK dtype
            if live_cpu.dtype != disk_cpu.dtype:
                live_cpu = live_cpu.to(disk_cpu.dtype)

            ok = torch.allclose(live_cpu, disk_cpu, atol=atol, rtol=rtol)
            if not ok:
                mismatches.append(key)
                if len(mismatches) <= max_print_mismatches:
                    diff = (live_cpu.flatten()[:8] - disk_cpu.flatten()[:8]).abs()
                    rank0_print(f"❌ MISMATCH at {key}")
                    rank0_print(f"   disk dtype={disk_cpu.dtype}, live dtype(after cast)={live_cpu.dtype}, shape={tuple(disk_cpu.shape)}")
                    rank0_print(f"   Disk (First 3): {disk_cpu.flatten()[:3].tolist()}")
                    rank0_print(f"   Live (First 3): {live_cpu.flatten()[:3].tolist()}")
                    rank0_print(f"   |diff| (First 3): {diff[:3].tolist()}")

            total_keys_checked += 1

    rank0_print("-" * 60)

    if strict_key_coverage and missing_in_live:
        rank0_print(f"🚨 KEY COVERAGE FAILED: {len(missing_in_live)} shard keys not found in model.state_dict().")
        rank0_print("First 10 missing keys:")
        for k in missing_in_live[:10]:
            rank0_print(f"   - {k}")
        rank0_print("-" * 60)

    if mismatches:
        rank0_print(f"🚨 VALUE CHECK FAILED: {len(mismatches)} mismatched keys out of {total_keys_checked} checked keys.")
    else:
        rank0_print(f"✅ SUCCESS: Verified {total_keys_checked} keys. In-memory tensors match shard tensors within atol/rtol.")
    rank0_print("-" * 60)

    return {
        "total_keys_checked": total_keys_checked,
        "num_mismatches": len(mismatches),
        "mismatches": mismatches,
        "num_missing_in_live": len(missing_in_live),
        "missing_in_live": missing_in_live,
    }

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    import os
    rank0_print("args.use_router:", args.use_router)
    disable_torch_init()
    router_name = args.router_name # will dino, clip, or dino_clip
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    rank0_print("model_path:", model_path)
    rank0_print("model_name:", model_name)

    model_name = get_model_name_from_path(args.model_path)
    use_router = (str(args.use_router) == "True") or (args.use_router is True)
    if use_router:
        rank0_print("\n      we should be using router")
        tokenizer, model, image_processor_dino, image_processor_clip, context_len = load_pretrained_model_router(
                                                            model_path=model_path,
                                                            model_base=args.model_base,
                                                            model_name=model_name,
                                                            num_layers_to_choose=args.num_layers_to_choose,
                                                            router_alpha=args.router_alpha,
                                                            encoder_type=args.encoder_type,
                                                            mm_use_text_enc = args.mm_use_text_enc,
                                                            text_pool = args.text_pool,
                                                            text_scale = args.text_scale, # overall influence of text. Would this allow us to better modulate the features. hyperparam sweep
                                                            text_conditioning= args.text_conditioning, # "film" or "attention"
                                                            text_attention_type=args.text_attention_type, # "concat" or "qk"
                                                            text_attention_dim=args.text_attention_dim,
                                                            attn_level= args.attn_level, # "token" or "layer" (used when text_conditioning == "attention")
                                                            force_reserve_layer=args.force_reserve_layer,
                                                            reserve_layer= args.reserve_layer,
                                                            learnable_gate= args.learnable_gate,
                                                            confidence_gate=args.confidence_gate,
                                                            confidence_gate_threshold= args.confidence_gate_threshold,
                                                            confidence_gate_slope= args.confidence_gate_slope,  # will help us get the correct router
                                                        )
    else:
        rank0_print("\n      not using router")
        tokenizer, model, image_processor_dino, image_processor_clip, context_len = load_pretrained_model(model_path=model_path,
                                                                               model_base=args.model_base,
                                                                               model_name=model_name,
                                                                               encoder_type=args.encoder_type,)

    verify_loaded_model(model, model_path)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    image_tensor_clip = None
    image_tensor_dino = None
    if image_processor_clip is not None:
        image_tensor_clip = process_images(
            images,
            image_processor_clip,
            model.config
        ).to(model.device, dtype=torch.float16)
    if image_processor_dino is not None:
        image_tensor_dino = process_images(
            images,
            image_processor_dino,
            model.config
        ).to(model.device, dtype=torch.float16)


    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
                input_ids,
                images=image_tensor_clip.to(dtype=torch.float16, device=next(model.parameters()).device, non_blocking=True) if image_tensor_clip is not None else None, # for some reason input was on cuda device 2 while model weight and bias were on cuda 0
                dino_images=image_tensor_dino.to(dtype=torch.float16, device=next(model.parameters()).device, non_blocking=True) if image_tensor_dino is not None else None,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                encoder_type=args.encoder_type,
                prompts=[prompt] if not isinstance(prompt, list) else prompt,
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--get_masks", type=int, default=1)  # 1 for True and 0 otherwise
    parser.add_argument("--output_path_vis", type=str, default="./playground/visualizing_experts")
    parser.add_argument("--benchmark", type=str, default="textvqa")
    parser.add_argument("--encoder_type", type=str, default="dinov2_txt", help="Type of encoder to use for the router. Options: dinov2_txt, dinov2_vit, clip_vit, etc. Should match the encoder used during training.")
    parser.add_argument("--mm_use_text_enc", action="store_true", help="Whether to use the text encoder for multimodal routing. Should match the setting used during training.")
    parser.add_argument("--text_pool", type=str, default="mean", help="How to pool text features for routing when using text enc. Options: mean, cls, etc.")
    parser.add_argument("--text_scale", type=float, default=1.0, help="Overall influence of text features in routing decisions. This can be used to modulate the contribution of text features relative to vision features. A hyperparameter sweep over this value could reveal interesting insights about the importance of text cues in expert selection.")
    parser.add_argument("--text_conditioning", type=str, default="attention", help="How to condition on text features for routing when using text enc. Options: film, attention.")
    parser.add_argument("--text_attention_type", type=str, default="qk", help="How to merge features when using text attention conditioning. Options: concat (concatenate text and vision features before routing), qk (use text features to compute query/key for attention over vision features).")
    parser.add_argument("--text_attention_dim", type=int, default=1024, help="Dim of text attention")
    parser.add_argument("--attn_level", type=str, default="token", help="Used when text_conditiong is attention. Options: token or layer.")
    parser.add_argument("--force_reserve_layer", type=bool, default=True, help="Use reserve layer or not.")
    parser.add_argument("--reserve_layer", type=int, default=-2, help="The reserve layer index. Only used if force_reserve_layer is True. This layer will always be selected as the expert, and the router will choose (num_layers_to_choose - 1) experts from the remaining layers.")
    parser.add_argument("--learnable_gate", type=bool, default=True, help="Use the learnable gate for the weighting of the image encoder features.")
    parser.add_argument("--confidence_gate", type=bool, default=False, help="Whether to use a confidence gate for the weighting of the image encoder features.")
    parser.add_argument("--confidence_gate_threshold", type=float, default=0.6, help="Threshold for the confidence gate. If the router's confidence (max softmax probability) is below this threshold, the image encoder features will be downweighted.")
    parser.add_argument("--confidence_gate_slope", type=float, default=10.0, help="Slope for the confidence gate. Higher values make the gate more like a hard threshold, while lower values make it smoother.")

    args = parser.parse_args()

    eval_model(args)
