import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import glob
from safetensors.torch import load_file
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_pretrained_model_router
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math
import types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


all_options = ['A', 'B', 'C', 'D']


def setup_router_prob_capture(model):
    """Monkey-patch router_probs on each router to collect probs during eval.
    Returns a dict mapping router name -> list of [B, L] float32 tensors."""
    buffers = {}
    for router_name in ("router_dino", "router_clip"):
        if not hasattr(model, router_name):
            continue
        router = getattr(model, router_name)
        buf = []
        buffers[router_name] = buf
        orig_fn = router.router_probs.__func__

        def _make_patched(orig, b):
            def patched(self, *args, **kwargs):
                probs = orig(self, *args, **kwargs)
                b.append(probs.detach().cpu().float())
                return probs
            return patched

        router.router_probs = types.MethodType(_make_patched(orig_fn, buf), router)
    return buffers


def plot_router_probs(buffers, output_dir, benchmark_name):
    """Aggregate collected router probs and save plots to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for router_name, buf in buffers.items():
        if not buf:
            continue
        probs = torch.cat(buf, dim=0).numpy()  # [N, L]
        output_json_rp = os.path.join(output_dir, benchmark_name)
        np.save(output_json_rp + ".npy", probs)
        mean_p = probs.mean(0)
        std_p  = probs.std(0)
        L = mean_p.shape[0]
        layers = np.arange(L)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{benchmark_name} | {router_name} router prob distribution")

        axes[0].bar(layers, mean_p, yerr=std_p, capsize=3, color="steelblue")
        axes[0].set_xlabel("Layer index")
        axes[0].set_ylabel("Mean probability")
        axes[0].set_title("Mean ± std router prob per layer")
        axes[0].set_xticks(layers)

        n_show = min(500, probs.shape[0])
        im = axes[1].imshow(probs[:n_show].T, aspect="auto", cmap="hot", vmin=0)
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Layer index")
        axes[1].set_title(f"Router prob heatmap (first {n_show} samples)")
        fig.colorbar(im, ax=axes[1])

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"router_probs_{benchmark_name}_{router_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved router prob plot: {out_path}")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

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
    print(f"\n--- Starting Deep Verification against safetensor shards in: {model_path} ---")

    live_state_dict = model.state_dict()

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_files:
        print(f"⚠️  No .safetensors shards found in {model_path}. Skipping verification.")
        return {}

    # --- Router dtype check ---
    router_key_candidates = [k for k in live_state_dict.keys() if "router" in k]
    if len(router_key_candidates) == 0:
        print("⚠️  Router dtype check: could not find a key ending with 'router.weights' in model.state_dict().")
    else:
        for rk in router_key_candidates:
            dt = live_state_dict[rk].dtype
            if dt != check_router_dtype:
                print(f"❌ Router dtype mismatch for {rk}: live dtype={dt}, expected={check_router_dtype}")
            else:
                print(f"✅ Router dtype OK for {rk}: {dt}")

    total_keys_checked = 0
    mismatches = []
    missing_in_live = []

    for shard_path in shard_files:
        print(f"Checking shard: {shard_path} ...")
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
                    print(f"❌ MISMATCH at {key}")
                    print(f"   disk dtype={disk_cpu.dtype}, live dtype(after cast)={live_cpu.dtype}, shape={tuple(disk_cpu.shape)}")
                    print(f"   Disk (First 3): {disk_cpu.flatten()[:3].tolist()}")
                    print(f"   Live (First 3): {live_cpu.flatten()[:3].tolist()}")
                    print(f"   |diff| (First 3): {diff[:3].tolist()}")

            total_keys_checked += 1

    print("-" * 60)

    if strict_key_coverage and missing_in_live:
        print(f"🚨 KEY COVERAGE FAILED: {len(missing_in_live)} shard keys not found in model.state_dict().")
        print("First 10 missing keys:")
        for k in missing_in_live[:10]:
            print(f"   - {k}")
        print("-" * 60)

    if mismatches:
        print(f"🚨 VALUE CHECK FAILED: {len(mismatches)} mismatched keys out of {total_keys_checked} checked keys.")
    else:
        print(f"✅ SUCCESS: Verified {total_keys_checked} keys. In-memory tensors match shard tensors within atol/rtol.")
    print("-" * 60)

    return {
        "total_keys_checked": total_keys_checked,
        "num_mismatches": len(mismatches),
        "mismatches": mismatches,
        "num_missing_in_live": len(missing_in_live),
        "missing_in_live": missing_in_live,
    }

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    use_router = (str(args.use_router) == "True") or (args.use_router is True)
    
    if use_router:
        print("\n      we should be using router")
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
        print("\n      not using router")
        tokenizer, model, image_processor_dino, image_processor_clip, context_len = load_pretrained_model(model_path=model_path, 
                                                                               model_base=args.model_base, 
                                                                               model_name=model_name)

    verify_loaded_model(model, model_path)

    router_prob_buffers = setup_router_prob_capture(model) if use_router else {}

    #tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor_dino = None
            image_tensor_clip = None
            if "dino" in args.encoder_type:
                image_tensor_dino = process_images([image], image_processor_dino, model.config)[0] if image_processor_dino is not None else None
            if "clip" in args.encoder_type:
                image_tensor_clip = process_images([image], image_processor_clip, model.config)[0] if image_processor_clip is not None else None
            
            with torch.inference_mode():
                # import pdb; pdb.set_trace()
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor_clip.unsqueeze(0).half().cuda() if image_tensor_clip is not None else None,
                    dino_images=image_tensor_dino.unsqueeze(0).half().cuda() if image_tensor_dino is not None else None,
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    prompts=[question],                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    encoder_type=args.encoder_type,)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

    if router_prob_buffers:
        plot_router_probs(
            router_prob_buffers,
            output_dir="/shared/nas2/sstoica2/MoL/playground/router_distribution",
            benchmark_name=args.benchmark
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--use_router", type=str, default="True")
    parser.add_argument("--router_name", type=str, default="dino")
    parser.add_argument("--num_layers_to_choose", type=int, default=1)
    parser.add_argument("--router_alpha", type=int, default=10**-2)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--encoder_type", type=str, default="dinov2_txt", help="Type of encoder to use for the router. Options: dinov2_txt, dinov2_vit, clip_vit, etc. Should match the encoder used during training.")
    parser.add_argument("--benchmark", type=str, default="textvqa")
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
