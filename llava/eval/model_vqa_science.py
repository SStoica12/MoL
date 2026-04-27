import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.router_logging import (
    clear_router_selection_metadata,
    get_router_selection_metadata,
)
from llava.model.builder import load_pretrained_model, load_pretrained_model_router
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

def eval_model(args):
    # Model
    rank0_print("args.use_router:", args.use_router)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    rank0_print("model_path:", model_path)
    rank0_print("model_name:", model_name)

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
                                                            confidence_gate_slope= args.confidence_gate_slope,
                                                            hybrid_disagreement_metric= args.hybrid_disagreement_metric,  
                                                            hybrid_reserve_mode= args.hybrid_reserve_mode, 
                                                        )
    else:
        rank0_print("\n      not using router")
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, 
                                                                               model_base=args.model_base, 
                                                                               model_name=model_name)

    verify_loaded_model(model, model_path)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            images_clip = None
            images_dino = None
            image_sizes_clip = None
            images_sizes_dino = None
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            if image_processor_clip:
                image_tensor_clip = process_images([image], image_processor_clip, model.config)[0]
                images_clip = image_tensor_clip.unsqueeze(0).half().cuda()
            if image_processor_dino:
                image_tensor_dino = process_images([image], image_processor_dino, model.config)[0]
                images_dino = image_tensor_dino.unsqueeze(0).half().cuda()
            image_sizes= [image.size]

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images_clip = None
            images_dino = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            clear_router_selection_metadata(model)
            output_ids = model.generate(
                input_ids,
                images=images_clip,
                dino_images=images_dino,
                image_sizes=image_sizes,
                clip_texts=[prompt] if images is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                encoder_type=args.encoder_type,
                prompts=[cur_prompt] if not isinstance(cur_prompt, list) else cur_prompt,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        metadata = get_router_selection_metadata(model)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": metadata}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
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
    parser.add_argument("--hybrid_reserve_mode", type=str, default="local_gate", help="When attn_level is hybrid, this determines how the reserve layer is used. Options: local_gate (use a local gate to combine the reserve layer with the router's selected experts), always_concat (always concatenate the reserve layer features with the vision features regardless of router selection), dynamic_concat (dynamically decide whether to concatenate the reserve layer based on the router's confidence or other criteria).")
    parser.add_argument("--hybrid_disagreement_metric", type=str, default="kl_local_global", help="When attn_level is hybrid, this determines how to measure disagreement between the local router (which chooses among the experts) and a global router (which has access to all features). Options: kl_local_global (compute KL divergence between the local router's distribution over experts and a global router's distribution that considers all features), confidence_gap (measure the gap in confidence between the local router's top choice and the global router's top choice), feature_distance (compute a distance metric between the features used by the local router and those used by the global router). This metric can be used to inform the hybrid reserve mode, for example by activating the reserve layer when disagreement is high.")
    args = parser.parse_args()

    eval_model(args)
