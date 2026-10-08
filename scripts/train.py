#!/usr/bin/env python3
"""Portable launcher for the existing MoL trainer; no model code is changed."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = json.loads((ROOT / "configs/models.json").read_text())
MODEL_KEYS = (
    "model_name_or_path", "model_type", "encoder_type", "vision_tower",
    "attn_level", "num_layers_to_choose", "learnable_gate", "vision_tower_delta",
    "force_reserve_layer", "reserve_layer", "mm_use_text_enc", "text_attention_dim",
    "text_attention_type", "text_conditioning", "hybrid_reserve_mode",
    "hybrid_disagreement_metric", "fusion_type",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--image-folder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pretrain-dir", type=Path, help="MoL stage-1 output, including router weights")
    p.add_argument("--model-base", help="Override the base language model with a local path or Hub ID")
    p.add_argument("--gpus", type=int, default=8, help="Number of visible GPUs on this node")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--batch-size", type=int, default=4, help="Per-GPU microbatch; tune for available memory")
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--router-lr", type=float)
    p.add_argument("--k", type=int, help="Override num_layers_to_choose in the selected profile")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Print the command without loading models or data")
    return p


def command(args):
    profile = PROFILES[args.profile]
    if min(args.gpus, args.batch_size, args.gradient_accumulation, args.epochs) <= 0:
        raise ValueError("GPU count, batch size, accumulation, and epochs must be positive.")
    if args.k is not None and args.k <= 0:
        raise ValueError("--k must be positive.")
    pretrain = args.stage == "pretrain"
    if not pretrain and args.pretrain_dir is None:
        raise ValueError("Finetuning requires --pretrain-dir from MoL stage 1.")
    options = {key: profile[key] for key in MODEL_KEYS}
    if args.model_base:
        options["model_name_or_path"] = args.model_base
    if args.k is not None:
        options["num_layers_to_choose"] = args.k
    lr = args.learning_rate if args.learning_rate is not None else (1e-3 if pretrain else profile["finetune_lr"])
    router_lr = args.router_lr if args.router_lr is not None else (profile["pretrain_router_lr"] if pretrain else lr)
    options.update(
        version="plain" if pretrain else (
            "bunny" if profile["model_type"] == "phi-1.5" else
            "llama" if profile["model_type"] == "llama3-8b" else "v1"
        ),
        data_path=args.data_path.resolve(), image_folder=args.image_folder.resolve(),
        output_dir=args.output_dir.resolve(),
        deepspeed=ROOT / "scripts" / ("zero2.json" if pretrain or profile["lora_enable"] else "zero3.json"),
        mm_projector_type="mlp2x_gelu", mm_vision_select_layer=-2,
        mm_use_im_start_end=False, mm_use_im_patch_token=False,
        use_text_conditioned_router=True, freeze_backbone=pretrain,
        tune_mm_mlp_adapter=pretrain, save_router_stats=False,
        image_aspect_ratio="pad", group_by_modality_length=not pretrain,
        bf16=True, tf32=True, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_strategy="no", save_strategy="steps", save_steps=50000,
        save_total_limit=1, learning_rate=lr, mm_projector_lr=lr,
        dino_mm_projector_lr=lr, router_lr=router_lr, weight_decay=0.,
        warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=1,
        model_max_length=2048, gradient_checkpointing=True,
        dataloader_num_workers=args.workers, lazy_preprocess=True, report_to="none",
    )
    if not pretrain:
        folder = args.pretrain_dir.resolve()
        if "clip" in profile["encoder_type"] or "siglip" in profile["encoder_type"]:
            options["pretrain_mm_mlp_adapter"] = folder / "mm_projector.bin"
        if "dino" in profile["encoder_type"]:
            options["pretrain_dino_mm_mlp_adapter"] = folder / "dino_mm_projector.bin"
        if profile["lora_enable"]:
            options.update(lora_enable=True, lora_r=128, lora_alpha=256)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args.gpus),
           "--master_port", str(args.master_port), str(ROOT / "llava/train/train_mem_router.py")]
    for key, value in options.items():
        cmd.extend(["--" + key, str(value)])
    return cmd, options


def main():
    p = parser()
    args = p.parse_args()
    try:
        cmd, options = command(args)
        if not args.dry_run:
            if not args.data_path.is_file() or not args.image_folder.is_dir():
                raise ValueError("--data-path must be a JSON file and --image-folder an existing directory.")
            for key in ("pretrain_mm_mlp_adapter", "pretrain_dino_mm_mlp_adapter"):
                if key in options and not options[key].is_file():
                    raise ValueError(f"Missing stage-1 weights: {options[key]}")
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("--output-dir must be empty; use the underlying trainer for explicit resume.")
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            env = os.environ.copy()
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
