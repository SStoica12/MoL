import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_pretrained_model_router
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


import os
import torch
from safetensors.torch import load_file
import glob
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

    Key fixes vs your version:
      - Compare in *disk dtype* (cast live -> disk dtype), not disk -> live dtype.
      - Use CPU tensors for both; avoids device mismatch.
      - Optional: ensure all keys in shards exist in model (strict_key_coverage).
      - Explicitly validate router.weights dtype (bf16 by default).
    """
    print(f"\n--- Starting Deep Verification against safetensor shards in: {model_path} ---")

    live_state_dict = model.state_dict()  # refs; ok

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors shards found in {model_path}")

    # --- Router dtype check (what you asked for) ---
    router_key_candidates = [k for k in live_state_dict.keys() if k.endswith("router.weights") or k == "router.weights"]
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
        disk_weights = load_file(shard_path)  # typically already on CPU

        for key, disk_tensor in disk_weights.items():
            if key not in live_state_dict:
                if strict_key_coverage:
                    missing_in_live.append(key)
                continue

            live_tensor = live_state_dict[key]

            # Move live to CPU
            try:
                live_cpu = live_tensor.detach().cpu()
            except:
                print("PROBLEMS WITH TENSOR")
                print("key : ", key)
                print("live_tensor : ", live_tensor)
                import sys; sys.exit(0)
            disk_cpu = disk_tensor.detach().cpu()

            # IMPORTANT: cast LIVE -> DISK dtype (most faithful to the checkpoint)
            if live_cpu.dtype != disk_cpu.dtype:
                live_cpu = live_cpu.to(disk_cpu.dtype)

            # Compare
            ok = torch.allclose(live_cpu, disk_cpu, atol=atol, rtol=rtol)
            if not ok:
                mismatches.append(key)
                if len(mismatches) <= max_print_mismatches:
                    # show a tiny diff summary
                    diff = (live_cpu.flatten()[:8] - disk_cpu.flatten()[:8]).abs()
                    print(f"❌ MISMATCH at {key}")
                    print(f"   disk dtype={disk_cpu.dtype}, live dtype(after cast)={live_cpu.dtype}, shape={tuple(disk_cpu.shape)}")
                    print(f"   Disk (First 3): {disk_cpu.flatten()[:3].tolist()}")
                    print(f"   Live (First 3): {live_cpu.flatten()[:3].tolist()}")
                    print(f"   |diff| (First 3): {diff[:3].tolist()}")

            total_keys_checked += 1

    print("-" * 60)

    if strict_key_coverage and missing_in_live:
        print(f"🚨 KEY COVERAGE FAILED: {len(missing_in_live)} keys exist in shards but not in model.state_dict().")
        print("   (This can be OK if you intentionally changed architecture / removed modules.)")
        print("   First 10 missing keys:")
        for k in missing_in_live[:10]:
            print(f"   - {k}")
        print("-" * 60)

    if mismatches:
        print(f"🚨 VALUE CHECK FAILED: Found {len(mismatches)} mismatched keys out of {total_keys_checked} compared keys.")
        # Helpful breakdowns
        vision_errors = [k for k in mismatches if "vision_tower" in k]
        router_errors = [k for k in mismatches if "router" in k]
        if vision_errors:
            print(f"   -> {len(vision_errors)} mismatches are in the vision tower.")
        if router_errors:
            print(f"   -> {len(router_errors)} mismatches are in the router.")
        print("   Note: small mismatches can still come from dtype/quantization differences;")
        print("   but if you cast live->disk dtype (as done here), these should be much rarer.")
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




all_options = ['A', 'B', 'C', 'D']


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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # import pdb; pdb.set_trace()
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    if args.use_router == "True":
        print(f"\n      we should be using router")
        tokenizer, model, image_processor, context_len = load_pretrained_model_router(model_path, args.model_base, model_name)
    else:
        print(f"\n      not using router")
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    verify_loaded_model(model, model_path)
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

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--use_router", type=str, default="True")
    parser.add_argument("--num_layers_to_choose", type=int, default=1)
    parser.add_argument("--router_alpha", type=int, default=10**-2)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument('--output_grouped_questions', default="gqa_grouped_questions.json", type=str, help="Output filename for grouped questions JSON.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
