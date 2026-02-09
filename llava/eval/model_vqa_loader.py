import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import glob
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_pretrained_model_router
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

from safetensors.torch import load_file


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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


def usage(layer_expert_masks: dict, total_num_layers: int, benchmark: str, out_path: str, chunk_idx: int):
    """
    layer_expert_masks: {qid: {"inter_layers": tensor [B,K] or [K], "inter_layers_probs": tensor [B,K] or [K], ...}}
    Produces a global unweighted usage bar plot (normalized).
    """
    os.makedirs(out_path, exist_ok=True)

    all_layers = np.concatenate(
        [v["inter_layers"].reshape(-1).cpu().numpy().astype(np.int64) for v in layer_expert_masks.values()],
        axis=0
    )

    usage_unweighted = np.bincount(all_layers, minlength=total_num_layers).astype(np.float64)
    usage_unweighted = usage_unweighted / max(usage_unweighted.sum(), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    x = np.arange(total_num_layers)

    axes[0].bar(x, usage_unweighted)
    axes[0].set_xticks(x) 
    axes[0].set_xticklabels(x)  
    axes[0].set_title("Unweighted Expert Usage (Hard Counts)")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Normalized usage")

    axes[1].axis("off")

    fig.suptitle(f"Layer Expert Usage Comparison — {benchmark}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_path, f"expert_usage_comparison_{benchmark}_{chunk_idx}.png")
    plt.savefig(path)
    print(f"saved overall usage bar plot to {path}")
    plt.close(fig)

def plot_router_probs_heatmap(
    layer_expert_masks: dict,
    total_num_layers: int,
    benchmark: str,
    out_path: str,
    chunk_idx: int,
    num_samples: int = 10,
):
    """
    (1) Randomly sample `num_samples` question ids
    (2) Grab router_probs for those questions
    (3) Plot heatmap: rows = question ids, cols = layers

    Expects:
      layer_expert_masks[qid]["router_probs"]  # array-like or tensor, shape [B, L] or [L]
    """
    os.makedirs(out_path, exist_ok=True)

    keys = np.array(list(layer_expert_masks.keys())) 
    qids = keys[np.random.randint(0, len(keys), size=num_samples)]

    rows = []
    used_qids = []
    for qid in qids:
        if "router_probs" not in layer_expert_masks[qid]:
            continue

        rp = layer_expert_masks[qid]["router_probs"]

        # tensor -> numpy
        if hasattr(rp, "detach"):
            rp = rp.detach().cpu().numpy()

        rp = np.asarray(rp, dtype=np.float32)

        # [B, L] -> average over batch, then [L]
        if rp.ndim == 2:
            rp = rp.mean(axis=0)
        elif rp.ndim != 1:
            rp = rp.reshape(-1)

        rows.append(rp)
        used_qids.append(qid)

    if len(rows) == 0:
        raise KeyError("None of the sampled qids contained 'router_probs'.")

    M = np.stack(rows, axis=0)  # [k, L]

    # ---- plot heatmap ----
    fig, ax = plt.subplots(figsize=(max(10, total_num_layers * 0.25), 0.55 * len(used_qids) + 2))

    im = ax.imshow(M, aspect="auto")  # no explicit colors per your preference

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Question id (sampled)")

    ax.set_xticks(np.arange(total_num_layers))
    ax.set_yticks(np.arange(len(used_qids)))
    ax.set_yticklabels([str(q) for q in used_qids])

    # If labels are dense, rotate x labels
    if total_num_layers > 20:
        ax.set_xticklabels([str(i) for i in range(total_num_layers)], rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([str(i) for i in range(total_num_layers)])

    ax.set_title(f"Router probs heatmap — {benchmark} — chunk {chunk_idx} (n={len(used_qids)})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("router_prob")

    plt.tight_layout()
    path = os.path.join(out_path, f"router_probs_heatmap_{benchmark}_chunk{chunk_idx}.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved heatmap to: {path}")

    return used_qids, M


def get_usage_per_category(grouped_questions: dict, layer_expert_masks: dict, total_num_layers: int, benchmark: str, out_path: str, chunk_idx: int):
    """
    grouped_questions:
      Either { category: { sub_category: [ {QID: ...}, ... ] } }
      Or     { category: [ {QID: ...}, ... ] }

    layer_expert_masks:
      { qid: {
          "inter_layers": torch.Tensor [B,K] or [K],
          "inter_layers_probs": torch.Tensor [B,K] or [K],
          "last_layer_prob": ...
        } }
    """
    os.makedirs(out_path, exist_ok=True)

    qid_to_layers = {}
    qid_to_probs = {}

    for qid, v in layer_expert_masks.items():
        if "inter_layers" not in v or "inter_layers_probs" not in v:
            continue
        layers = v["inter_layers"].detach().reshape(-1).cpu().long()
        probs = v["inter_layers_probs"].detach().reshape(-1).cpu().float()
        good = (layers >= 0) & (layers < total_num_layers)
        if good.any():
            qid_to_layers[qid] = layers[good]
            qid_to_probs[qid] = probs[good]

    for cat, value in grouped_questions.items():
        if isinstance(value, dict):
            subcat_items = value.items()
        else:
            subcat_items = [(None, value)]

        for subcat, items in subcat_items:
            qids = [it.get("QID") for it in items if it.get("QID") in qid_to_layers]
            if len(qids) == 0:
                continue

            all_layers = torch.cat([qid_to_layers[q] for q in qids], dim=0)
            all_probs = torch.cat([qid_to_probs[q] for q in qids], dim=0)

            usage_unweighted = torch.bincount(all_layers, minlength=total_num_layers).float()
            usage_weighted = torch.zeros(total_num_layers, dtype=torch.float32)
            usage_weighted.scatter_add_(0, all_layers, all_probs)

            if usage_unweighted.sum() > 0:
                usage_unweighted = usage_unweighted / usage_unweighted.sum()
            if usage_weighted.sum() > 0:
                usage_weighted = usage_weighted / usage_weighted.sum()

            x = torch.arange(total_num_layers).numpy()
            unweighted_np = usage_unweighted.numpy()
            weighted_np = usage_weighted.numpy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

            axes[0].bar(x, unweighted_np)
            axes[0].set_title("Unweighted (Hard Counts)")
            axes[0].set_xlabel("Layer index")
            axes[0].set_ylabel("Normalized usage")

            axes[1].bar(x, weighted_np)
            axes[1].set_title("Weighted (Sum of probs)")
            axes[1].set_xlabel("Layer index")

            title = f"{benchmark} — {cat}"
            if subcat is not None:
                title += f" / {subcat}"
            title += f" (n={len(qids)})"

            fig.suptitle(title, fontsize=13)
            plt.tight_layout()

            safe_cat = str(cat).replace("/", "_")
            safe_sub = "all" if subcat is None else str(subcat).replace("/", "_")
            fname = f"expert_usage_{benchmark}__{safe_cat}__{safe_sub}_{args.chunk_idx}.png"
            path = os.path.join(out_path, fname)
            plt.savefig(path)
            print(f"\nSaving to path: {path}")
            plt.close(fig)


# ============================================================
# NEW: Merge chunks and plot (ONLY affects merging/plotting)
# ============================================================
def merge_chunks_and_plot(args):
    """
    Merges all per-chunk masks files and plots usage over the entire benchmark.

    Assumptions (matches your current code):
      - Each chunk writes: <answers_file>.replace(".jsonl", ".masks.pt")
      - Your launcher sets answers-file like: .../<CHUNKS>_<IDX>.jsonl
      - Therefore masks are: .../<CHUNKS>_<IDX>.masks.pt

    This function:
      - Finds all *.masks.pt in the answers directory
      - Loads and merges dicts keyed by question_id
      - Infers total_num_layers from layer indices (max + 2)
      - Calls usage(merged, total_num_layers, args.benchmark, args.output_path_vis)
    """
    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)

    mask_files = sorted(glob.glob(os.path.join(answers_dir, "*.masks.pt")))
    if not mask_files:
        raise FileNotFoundError(f"No *.masks.pt files found in answers_dir={answers_dir}")

    print(f"[merge] answers_dir={answers_dir}")
    print(f"[merge] found {len(mask_files)} mask files:")
    for p in mask_files:
        print(f"  - {os.path.basename(p)}")

    merged = {}
    max_layer = -1

    for p in mask_files:
        d = torch.load(p, map_location="cpu")
        if not isinstance(d, dict):
            raise TypeError(f"Loaded object from {p} is not a dict. Got type={type(d)}")

        # Detect overlaps (should not happen if chunking is correct)
        overlap = set(merged).intersection(d)
        if overlap:
            raise RuntimeError(
                f"Duplicate QIDs while merging {p}. Example duplicates: {list(overlap)[:10]}"
            )

        # Keep tensors (usage expects tensors) but track max layer
        for qid, v in d.items():
            inter_layers = v["inter_layers"]
            if torch.is_tensor(inter_layers):
                if inter_layers.numel() > 0:
                    max_layer = max(max_layer, int(inter_layers.max().item()))
            else:
                inter_layers_np = np.asarray(inter_layers).reshape(-1)
                if inter_layers_np.size > 0:
                    max_layer = max(max_layer, int(inter_layers_np.max()))

            merged[qid] = v

        print(f"[merge] merged {os.path.basename(p)}  added={len(d)}  total={len(merged)}")

    if max_layer < 0:
        raise RuntimeError("[merge] No inter_layers indices found; merged dict seems empty.")

    # inter_layers came from router_probs[:, :-1], so total experts = (max inter idx) + 2
    total_num_layers = int(max_layer) + 2
    print(f"[merge] merged_questions={len(merged)} inferred_total_num_layers={total_num_layers}")

    # Plot global usage across ALL questions
    usage(merged, total_num_layers, args.benchmark, args.output_path_vis)

    # Save merged masks for later analysis
    merged_out = os.path.join(answers_dir, f"merged_masks_{args.benchmark}.pt")
    torch.save(merged, merged_out)
    print(f"[merge] saved merged masks to: {merged_out}")


def eval_model(args):
    print("args.use_router:", args.use_router)
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print("model_path:", model_path)
    print("model_name:", model_name)

    # Make router flag robust to string/bool
    use_router = (str(args.use_router) == "True") or (args.use_router is True)

    if use_router:
        print("\n      we should be using router")
        tokenizer, model, image_processor, context_len = load_pretrained_model_router(
                                                            model_path=model_path,
                                                            model_base=args.model_base,
                                                            model_name=model_name,
                                                            num_layers_to_choose=args.num_layers_to_choose,
                                                            router_alpha=args.router_alpha
                                                        )
    else:
        print("\n      not using router")
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, 
                                                                               model_base=args.model_base, 
                                                                               model_name=model_name)

    verify_loaded_model(model, model_path)

    if args.get_masks == 1:
        model.eval()

    # Load and chunk questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    masks_file = answers_file.replace(".jsonl", ".masks.pt")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems this is a plain model, but not using an mmtag prompt; auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        #batch_size=1,
        #num_workers=4,
    )

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if use_router and hasattr(model, "router"):
                model.router.current_qid = idx

            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")

    ans_file.close()

    # Save masks PER CHUNK so you can merge across GPUs
    if use_router and hasattr(model, "router"):
        layer_expert_masks = model.router.expert_masks_per_q
        print(f"\nChunk {args.chunk_idx}/{args.num_chunks}: layer_expert_masks={len(layer_expert_masks)} questions={len(questions)}")
        assert len(layer_expert_masks) == len(questions), "mask count != chunk question count"

        torch.save(layer_expert_masks, masks_file)
        print(f"Saved chunk masks to: {masks_file}")

        # keep per-chunk visualization (your existing behavior)
        usage(layer_expert_masks, 
              model.router.num_layers, 
              args.benchmark, 
              args.output_path_vis, 
              args.chunk_idx)
        plot_router_probs_heatmap(
            layer_expert_masks,
            model.router.num_layers,
            args.benchmark,
            args.output_path_vis,
            args.chunk_idx,
            num_samples = 10
        )
    # NEW: optionally merge & plot only when running the last chunk
    # (Does not change CLI; uses existing args.num_chunks and args.chunk_idx)
    if args.get_masks == 1: #and int(args.chunk_idx) == int(args.num_chunks) - 1:
        try:
            merge_chunks_and_plot(args)
        except Exception as e:
            print(f"[merge][WARN] merge_chunks_and_plot failed: {e}")


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
    parser.add_argument("--output_grouped_questions", default="gqa_grouped_questions.json", type=str,
                        help="Output filename for grouped questions JSON.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--get_masks", type=int, default=1)  # 1 for True and 0 otherwise
    parser.add_argument("--output_path_vis", type=str, default="./playground/visualizing_experts")
    parser.add_argument("--benchmark", type=str, default="textvqa")
    args = parser.parse_args()

    eval_model(args)
