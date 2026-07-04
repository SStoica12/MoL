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
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoConfig

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def _load_eval_config(model_path: str):
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        rank0_print(f"[eval][WARN] Could not load config from {model_path}: {exc}")
        return None


def _infer_encoder_type_from_config(config, fallback: str) -> str:
    if config is None:
        return fallback

    configured = getattr(config, "encoder_type", None)
    vision_tower = str(getattr(config, "mm_vision_tower", "") or getattr(config, "vision_tower", "")).lower()

    if "siglip" in vision_tower:
        return "siglip"
    if configured:
        return configured
    if "dino" in vision_tower:
        return "dinov2_txt"
    if "clip" in vision_tower or "openai" in vision_tower or "laion" in vision_tower:
        return "clip"
    return fallback


def _apply_checkpoint_eval_defaults(args):
    config = _load_eval_config(os.path.expanduser(args.model_path))
    if config is None:
        return

    model_type = str(getattr(config, "model_type", "")).lower()
    is_multimodal = (
        model_type.startswith("llava")
        or hasattr(config, "mm_vision_tower")
        or bool(getattr(config, "use_mm_proj", False))
    )
    has_router = (
        bool(getattr(config, "use_text_conditioned_router", False))
        or bool(getattr(config, "use_text_only_gating", False))
        or bool(getattr(config, "has_router", False))
    )

    if is_multimodal and has_router and not args.use_router:
        rank0_print("[eval] Auto-enabling router from checkpoint config.")
        args.use_router = True

    inferred_encoder_type = _infer_encoder_type_from_config(config, args.encoder_type)
    if inferred_encoder_type != args.encoder_type:
        rank0_print(f"[eval] Using encoder_type={inferred_encoder_type} inferred from checkpoint config.")
        args.encoder_type = inferred_encoder_type

    if args.use_router and ("clip" in args.encoder_type or "siglip" in args.encoder_type) and "dino" not in args.encoder_type:
        args.router_name = "clip"

    for attr in (
        "num_layers_to_choose",
        "force_reserve_layer",
        "reserve_layer",
        "learnable_gate",
        "confidence_gate",
        "confidence_gate_threshold",
        "confidence_gate_slope",
        "hybrid_reserve_mode",
        "hybrid_disagreement_metric",
        "text_pool",
        "text_scale",
        "text_conditioning",
        "text_attention_type",
        "text_attention_dim",
        "attn_level",
        "fusion_type",
    ):
        if hasattr(config, attr):
            if attr == "num_layers_to_choose" and args.num_layers_to_choose is not None:
                rank0_print(
                    f"[eval] Overriding checkpoint num_layers_to_choose={getattr(config, attr)} "
                    f"with CLI value {args.num_layers_to_choose}."
                )
                continue
            setattr(args, attr, getattr(config, attr))

    if bool(getattr(config, "mm_use_text_enc", False)):
        args.mm_use_text_enc = True

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


def plot_router_probs(buffers, output_dir, benchmark_name, chunk_idx=0):
    """Aggregate collected router probs and save plots to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for router_name, buf in buffers.items():
        if not buf:
            continue
        probs = torch.cat(buf, dim=0).numpy()  # [N, L]
        output_json_rp = os.path.join(output_dir, f"router_probs_{benchmark_name}_{router_name}_chunk{chunk_idx}.npy")
        np.save(output_json_rp, probs)
        mean_p = probs.mean(0)
        std_p  = probs.std(0)
        L = mean_p.shape[0]
        layers = np.arange(L)
        uniform = np.ones(L) / L

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{benchmark_name} | {router_name} | chunk {chunk_idx} | N={probs.shape[0]}")

        # Plot 1: mean ± std bar chart
        axes[0, 0].bar(layers, mean_p, yerr=std_p, capsize=3, color="steelblue", label="mean")
        axes[0, 0].axhline(1.0 / L, color="red", linestyle="--", label=f"uniform (1/{L})")
        axes[0, 0].set_xlabel("Layer index")
        axes[0, 0].set_ylabel("Mean probability")
        axes[0, 0].set_title("Mean ± std router prob per layer")
        axes[0, 0].set_xticks(layers)
        axes[0, 0].legend()

        # Plot 2: heatmap of first N samples
        n_show = min(500, probs.shape[0])
        im = axes[0, 1].imshow(probs[:n_show].T, aspect="auto", cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_xlabel("Sample index")
        axes[0, 1].set_ylabel("Layer index")
        axes[0, 1].set_title(f"Router prob heatmap (first {n_show} samples)")
        fig.colorbar(im, ax=axes[0, 1])

        # Plot 3: layer 22 prob distribution (histogram)
        layer22_idx = min(22, L - 1)
        axes[1, 0].hist(probs[:, layer22_idx], bins=50, color="orange", edgecolor="black")
        axes[1, 0].axvline(1.0 / L, color="red", linestyle="--", label=f"uniform (1/{L})")
        axes[1, 0].set_xlabel(f"Router prob for layer {layer22_idx}")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(f"Layer {layer22_idx} prob distribution (collapse check)")
        axes[1, 0].legend()

        # Plot 4: mean prob vs uniform deviation
        deviation = mean_p - uniform
        colors = ["tomato" if d > 0 else "steelblue" for d in deviation]
        axes[1, 1].bar(layers, deviation, color=colors)
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].set_xlabel("Layer index")
        axes[1, 1].set_ylabel("Mean prob − uniform")
        axes[1, 1].set_title("Deviation from uniform (red = above, blue = below)")
        axes[1, 1].set_xticks(layers)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"router_probs_{benchmark_name}_{router_name}_chunk{chunk_idx}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        rank0_print(f"\nSaved router prob plot: {out_path}")

from safetensors.torch import load_file


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor_dino,
        image_processor_clip,
        model_config,
        conv_mode,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor_dino = image_processor_dino
        self.image_processor_clip = image_processor_clip
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        line = self.questions[index]
        image_files = line.get("images")
        #print("type(image_files): ", type(image_files), "image_file: ", image_files)
        if image_files is None:
            image_file = line.get("image")
            image_files = [image_file] if image_file is not None and str(image_file).strip() != "" else []
        else:
            image_files = [image_file for image_file in image_files if image_file is not None and str(image_file).strip() != ""]
        has_image = len(image_files) > 0
        qs = line["text"]

        if has_image:
            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model_config, "mm_use_im_start_end", False):
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

            if IMAGE_PLACEHOLDER in qs:
                if qs.count(IMAGE_PLACEHOLDER) != len(image_files):
                    raise ValueError(
                        f"Question {line.get('question_id')} has {len(image_files)} images but "
                        f"{qs.count(IMAGE_PLACEHOLDER)} image placeholders."
                    )
                qs = qs.replace(IMAGE_PLACEHOLDER, replace_token)
            elif DEFAULT_IMAGE_TOKEN in qs:
                if qs.count(DEFAULT_IMAGE_TOKEN) != len(image_files):
                    raise ValueError(
                        f"Question {line.get('question_id')} has {len(image_files)} images but "
                        f"{qs.count(DEFAULT_IMAGE_TOKEN)} <image> tokens."
                    )
                if getattr(self.model_config, "mm_use_im_start_end", False):
                    qs = qs.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            else:
                qs = "\n".join([replace_token] * len(image_files)) + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if has_image:
            resolved_paths = [
                image_file if os.path.isabs(image_file) else os.path.join(self.image_folder, image_file)
                for image_file in image_files
            ]
            images = []
            for image_path in resolved_paths:
                try:
                    with Image.open(image_path) as image:
                        images.append(image.convert('RGB'))
                except Exception as exc:
                    raise OSError(
                        f"Failed to load image '{image_path}' for question_id={line.get('question_id')}: {exc}"
                    ) from exc

            image_tensor_clip = None
            image_tensor_dino = None
            image_sizes = [image.size for image in images]
            # print("\n self.image_processor_clip is None: ", self.image_processor_clip is None)
            # print("\n self.image_processor_dino is None: ", self.image_processor_dino is None)
            if self.image_processor_clip is not None:
                # rank0_print("\n processing image with CLIP processor")
                image_tensor_clip = process_images(images, self.image_processor_clip, self.model_config) #[0]
            if self.image_processor_dino is not None:
                # rank0_print("\n processing image with DINO processor")
                image_tensor_dino = process_images(images, self.image_processor_dino, self.model_config) #[0]

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            # image_tensor = process_images(images, self.image_processor, self.model_config)
            # image_sizes = [image.size for image in images]
            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        else:
            image_tensor_clip = None
            image_tensor_dino = None
            image_sizes = None
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids[0]

        """
        prompt :  A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
        What is the brand of this camera?
        Reference OCR token: DAKOTA, DIGITAL, Single-Use, Camera, Pire, digitat
        Answer the question using a single word or phrase. ASSISTANT:
        """
        # rank0_print("prompt : ", prompt, "; type: ", type(prompt))
        # rank0_print("=========================")
        # import sys; sys.exit(0)
        return input_ids, image_tensor_clip, image_tensor_dino, image_sizes

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    assert len(batch) == 1, "batch_size must be 1"
    input_ids, image_tensor_clip, image_tensor_dino, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    # he did not do this stacking that llava did. It seems like all data llava used
    # only had one image for eval, so the stacking worked. However, for datasets with multiple
    # images, like mmmlu, the stacking resulted in first dim being 1 rather than 2,
    # resulting in problems in llava_arch.
    make_image_size_false_clip = False
    make_image_size_false_dino = False
    if image_tensor_clip is not None:
        if hasattr(image_tensor_clip, "ndim") and image_tensor_clip.ndim == 3:
            image_tensor_clip = image_tensor_clip.unsqueeze(0)
    else:
        make_image_size_false_clip = True

    if image_tensor_dino is not None:
        if hasattr(image_tensor_dino, "ndim") and image_tensor_dino.ndim == 3:
            image_tensor_dino = image_tensor_dino.unsqueeze(0)
    else:
        make_image_size_false_dino = True
    if make_image_size_false_clip and make_image_size_false_dino:
        image_sizes = None
    return input_ids, image_tensor_clip, image_tensor_dino, image_sizes


def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor_dino,
    image_processor_clip,
    model_config,
    conv_mode,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor_dino,
        image_processor_clip,
        model_config,
        conv_mode=conv_mode,
    )
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
    rank0_print(f"\n--- Starting Deep Verification against safetensor shards in: {model_path} ---")

    live_state_dict = model.state_dict()

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shard_files:
        rank0_print(f"⚠️  No .safetensors shards found in {model_path}. Skipping verification.")
        return {}
    if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
        rank0_print("ℹ️  PEFT adapter checkpoint detected; skipping strict adapter-key coverage after LoRA merge.")
        strict_key_coverage = False

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
    rank0_print(f"saved overall usage bar plot to {path}")
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
    rank0_print(f"Saved heatmap to: {path}")

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
            rank0_print(f"\nSaving to path: {path}")
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

    rank0_print(f"[merge] answers_dir={answers_dir}")
    rank0_print(f"[merge] found {len(mask_files)} mask files:")
    for p in mask_files:
        rank0_print(f"  - {os.path.basename(p)}")

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

        rank0_print(f"[merge] merged {os.path.basename(p)}  added={len(d)}  total={len(merged)}")

    if max_layer < 0:
        raise RuntimeError("[merge] No inter_layers indices found; merged dict seems empty.")

    # inter_layers came from router_probs[:, :-1], so total experts = (max inter idx) + 2
    total_num_layers = int(max_layer) + 2
    rank0_print(f"[merge] merged_questions={len(merged)} inferred_total_num_layers={total_num_layers}")

    # Plot global usage across ALL questions
    usage(merged, total_num_layers, args.benchmark, args.output_path_vis)

    # Save merged masks for later analysis
    merged_out = os.path.join(answers_dir, f"merged_masks_{args.benchmark}.pt")
    torch.save(merged, merged_out)
    rank0_print(f"[merge] saved merged masks to: {merged_out}")


def _extract_router_metadata(model, router_name: str, batch_index: int = 0) -> dict:
    """Read the latest routing decision from router_dino or router_clip and return a JSON-serializable dict."""
    router = None
    decision_attr = None
    if "dino" in router_name and hasattr(model, "router_dino"):
        router = model.router_dino
        decision_attr = "_latest_routing_decision_dino"
    elif "clip" in router_name and hasattr(model, "router_clip"):
        router = model.router_clip
        decision_attr = "_latest_routing_decision_clip"
    if router is None or decision_attr is None:
        return {}

    payload = getattr(router, decision_attr, None)
    if not payload:
        return {}

    layers = payload.get("layers")
    probs = payload.get("probs")
    if layers is None or probs is None or batch_index >= layers.shape[0]:
        return {}

    sample_layers = layers[batch_index].to(torch.int64)
    sample_probs = probs[batch_index].to(torch.float32)
    num_layers_available = payload.get("num_layers_available")

    flat = sample_layers.reshape(-1)
    counts = torch.bincount(flat, minlength=num_layers_available or int(flat.max().item()) + 1)
    layer_counts = {str(i): int(c) for i, c in enumerate(counts) if int(c) > 0}
    total = sum(layer_counts.values()) or 1
    layer_freq = {k: v / total for k, v in layer_counts.items()}

    metadata = {
        "selection_mode": payload.get("selection_mode", "image_level"),
        "num_layers_available": num_layers_available,
        "num_layers_to_choose": payload.get("num_layers_to_choose"),
        "selected_layers": sample_layers.tolist(),
        "selected_layer_counts": layer_counts,
        "selected_layer_frequency": layer_freq,
        "selected_layer_probs": sample_probs.detach().cpu().tolist(),
    }

    reserve_layer_index = payload.get("reserve_layer_index")
    if reserve_layer_index is not None:
        metadata["reserve_layer_index"] = int(reserve_layer_index)

    reserve_layer_prob = payload.get("reserve_layer_prob")
    if reserve_layer_prob is not None and batch_index < reserve_layer_prob.shape[0]:
        s = reserve_layer_prob[batch_index].to(torch.float32)
        metadata["reserve_layer_prob"] = float(s.item()) if s.ndim == 0 else float(s.mean().item())

    return {"router": metadata}


def eval_model(args):
    _apply_checkpoint_eval_defaults(args)
    if args.num_layers_to_choose is None:
        args.num_layers_to_choose = 1
    rank0_print("args.use_router:", args.use_router)
    disable_torch_init()
    router_name = args.router_name # will dino, clip, or dino_clip
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    rank0_print("model_path:", model_path)
    rank0_print("model_name:", model_name)
    print("\n =========== args: ", args)
    # import sys; sys.exit(0)

    # Make router flag robust to string/bool
    # import pdb; pdb.set_trace()
    use_router = args.use_router
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
                                                            fusion_type= args.fusion_type,
                                                        )
    else:
        rank0_print("\n      not using router")
        tokenizer, model, image_processor_dino, image_processor_clip, context_len = load_pretrained_model(model_path=model_path,
                                                                               model_base=args.model_base,
                                                                               model_name=model_name,
                                                                               encoder_type=args.encoder_type,)

    verify_loaded_model(model, model_path, check_router_dtype=next(model.parameters()).dtype)
    # import sys; sys.exit(0)
    # import pdb; pdb.set_trace()
    router_prob_buffers = setup_router_prob_capture(model) if use_router else {}
    # import sys; sys.exit(0)
    if args.get_masks == 1:
        model.eval()

    # Load and chunk questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # masks_file = answers_file.replace(".jsonl", ".masks.pt")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        rank0_print(f'It seems this is a plain model, but not using an mmtag prompt; auto switching to {args.conv_mode}.')
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor_dino,
        image_processor_clip,
        model.config,
        conv_mode=args.conv_mode,
    )
    for (input_ids, image_tensor_clip, image_tensor_dino, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if "dino" in router_name:
                if use_router and hasattr(model, "router_dino") :
                    model.router_dino.current_qid = idx
            if "clip" in router_name:
                model.router_clip.current_qid = idx
            # rank0_print("prompts: ", prompts)
            # rank0_print("===========")
            # import pdb; pdb.set_trace()
            if isinstance(image_tensor_clip, tuple) and len(image_tensor_clip) == 1:
                image_tensor_clip = image_tensor_clip[0]
            if isinstance(image_tensor_dino, tuple) and len(image_tensor_dino) == 1:
                image_tensor_dino = image_tensor_dino[0]

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
                prompts=[cur_prompt] if not isinstance(cur_prompt, list) else cur_prompt,
            ) # for dino hybrid input ids are tensor([[  1, 360,   2]], device='cuda:0')
        # import pdb; pdb.set_trace()
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()  # so this becomes D.
        metadata = _extract_router_metadata(model, router_name)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": metadata
        }) + "\n")

    ans_file.close()
    rank0_print("router_prob_buffers : ", router_prob_buffers)
    if router_prob_buffers:
        plot_router_probs(
            router_prob_buffers,
            output_dir=os.environ.get("ROUTER_DISTRIBUTION_DIR", "./playground/router_distribution"),
            benchmark_name=args.benchmark,
            chunk_idx=args.chunk_idx,
        )
    if use_router and hasattr(model, "router_dino") and model.router_dino.reserve_gate is not None:
        w = model.router_dino.reserve_gate.weight.data
        b = model.router_dino.reserve_gate.bias.data
        # Sample a few gate logits to see typical sigmoid output
        dummy = torch.zeros(1, w.shape[1], device=w.device, dtype=w.dtype)
        logit = (dummy @ w.T + b).item()
        rank0_print(f"[gate] bias logit (zero input): {logit:.3f}  sigmoid={torch.sigmoid(torch.tensor(logit)):.3f}")
        rank0_print(f"[gate] weight norm: {w.norm().item():.4f}")
        """
        [gate] bias logit (zero input): -0.131  sigmoid=0.467
        [gate] weight norm: 3.6992
        """

    # Save masks PER CHUNK so you can merge across GPUs
    # if "dino" in router_name:
    #     if use_router and hasattr(model, "router_dino"):
    #         layer_expert_masks = model.router_dino.expert_masks_per_q
    #         rank0_print(f"\nChunk {args.chunk_idx}/{args.num_chunks}: layer_expert_masks={len(layer_expert_masks)} questions={len(questions)}")
    #         assert len(layer_expert_masks) == len(questions), "mask count != chunk question count"

    #         torch.save(layer_expert_masks, masks_file)
    #         rank0_print(f"Saved chunk masks to: {masks_file}")

    #         # keep per-chunk visualization (your existing behavior)
    #         usage(layer_expert_masks,
    #             model.router_dino.num_layers,
    #             args.benchmark,
    #             args.output_path_vis,
    #             args.chunk_idx)
    #         plot_router_probs_heatmap(
    #             layer_expert_masks,
    #             model.router_dino.num_layers,
    #             args.benchmark,
    #             args.output_path_vis,
    #             args.chunk_idx,
    #             num_samples = 10
    #         )
    # if "clip" in router_name:
    #     if use_router and hasattr(model, "router_clip"):
    #         layer_expert_masks = model.router_clip.expert_masks_per_q
    #         rank0_print(f"\nChunk {args.chunk_idx}/{args.num_chunks}: layer_expert_masks={len(layer_expert_masks)} questions={len(questions)}")
    #         assert len(layer_expert_masks) == len(questions), "mask count != chunk question count"

    #         torch.save(layer_expert_masks, masks_file)
    #         rank0_print(f"Saved chunk masks to: {masks_file}")

    #         # keep per-chunk visualization (your existing behavior)
    #         usage(layer_expert_masks,
    #             model.router_clip.num_layers,
    #             args.benchmark,
    #             args.output_path_vis,
    #             args.chunk_idx)
    #         plot_router_probs_heatmap(
    #             layer_expert_masks,
    #             model.router_clip.num_layers,
    #             args.benchmark,
    #             args.output_path_vis,
    #             args.chunk_idx,
    #             num_samples = 10
    #         )
    # NEW: optionally merge & plot only when running the last chunk
    # (Does not change CLI; uses existing args.num_chunks and args.chunk_idx)
    if args.get_masks == 1: #and int(args.chunk_idx) == int(args.num_chunks) - 1:
        try:
            merge_chunks_and_plot(args)
        except Exception as e:
            rank0_print(f"[merge][WARN] merge_chunks_and_plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--use_router", action="store_true")
    parser.add_argument("--router_name", type=str, default="dino")
    parser.add_argument("--num_layers_to_choose", type=int, default=None)
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
    parser.add_argument("--encoder_type", type=str, default="dinov2_txt", help="Type of encoder to use for the router. Options: dinov2_txt, dinov2_vit, clip_vit, etc. Should match the encoder used during training.")
    parser.add_argument("--mm_use_text_enc", action="store_true", help="Whether to use the text encoder for multimodal routing. Should match the setting used during training.")
    parser.add_argument("--text_pool", type=str, default="mean", help="How to pool text features for routing when using text enc. Options: mean, cls, etc.")
    parser.add_argument("--text_scale", type=float, default=1.0, help="Overall influence of text features in routing decisions. This can be used to modulate the contribution of text features relative to vision features. A hyperparameter sweep over this value could reveal interesting insights about the importance of text cues in expert selection.")
    parser.add_argument("--text_conditioning", type=str, default="attention", help="How to condition on text features for routing when using text enc. Options: film, attention.")
    parser.add_argument("--text_attention_type", type=str, default="qk", help="How to merge features when using text attention conditioning. Options: concat (concatenate text and vision features before routing), qk (use text features to compute query/key for attention over vision features).")
    parser.add_argument("--text_attention_dim", type=int, default=1024, help="Dim of text attention")
    parser.add_argument("--attn_level", type=str, default="token", help="Used when text_conditiong is attention. Options: token, layer,  patch_layer, or hybrid.")
    parser.add_argument("--force_reserve_layer", action="store_true", help="Use reserve layer or not.")
    parser.add_argument("--reserve_layer", type=int, default=-2, help="The reserve layer index. Only used if force_reserve_layer is True. This layer will always be selected as the expert, and the router will choose (num_layers_to_choose - 1) experts from the remaining layers.")
    parser.add_argument("--learnable_gate", action="store_true", help="Use the learnable gate for the weighting of the image encoder features.")
    parser.add_argument("--confidence_gate", action="store_true", help="Whether to use a confidence gate for the weighting of the image encoder features.")
    parser.add_argument("--confidence_gate_threshold", type=float, default=0.6, help="Threshold for the confidence gate. If the router's confidence (max softmax probability) is below this threshold, the image encoder features will be downweighted.")
    parser.add_argument("--confidence_gate_slope", type=float, default=10.0, help="Slope for the confidence gate. Higher values make the gate more like a hard threshold, while lower values make it smoother.")
    parser.add_argument("--hybrid_reserve_mode", type=str, default="local_gate", help="When attn_level is hybrid, this determines how the reserve layer is used. Options: local_gate (use a local gate to combine the reserve layer with the router's selected experts), always_concat (always concatenate the reserve layer features with the vision features regardless of router selection), dynamic_concat (dynamically decide whether to concatenate the reserve layer based on the router's confidence or other criteria).")
    parser.add_argument("--hybrid_disagreement_metric", type=str, default="kl_local_global", help="When attn_level is hybrid, this determines how to measure disagreement between the local router (which chooses among the experts) and a global router (which has access to all features). Options: kl_local_global (compute KL divergence between the local router's distribution over experts and a global router's distribution that considers all features), confidence_gap (measure the gap in confidence between the local router's top choice and the global router's top choice), feature_distance (compute a distance metric between the features used by the local router and those used by the global router). This metric can be used to inform the hybrid reserve mode, for example by activating the reserve layer when disagreement is high.")
    parser.add_argument("--fusion_type", type=str, default="attn", help="How to fuse the image features with the text features. Options: mean_pooling, concate_upsample, attn")
    args = parser.parse_args()

    eval_model(args)
