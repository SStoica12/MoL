#!/bin/bash

export WANDB_ENTITY="uiucnlp"
export WANDB_PROJECT="mixture-of-layers"
export WANDB_NAME="MoL_clip-dino-film_router_choose_layers_text-cond-attn-hybrid-router_disagreement-gate_lr5e-5-pretrain"
export WANDB_MODE="online"  # ensure it uploads
# export WANDB_DIR="/shared/nas2/jk100/MoL/scripts/v1_5/wandb" - where to store wandb logs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero2.json}"
DATA_PATH="${DATA_PATH:-/shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/blip_laion_cc_sbu_558k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/images}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-13b-moe-pretrain_5e-5_clip-dino-film_text-cond-attn-hybrid-router_pretrain_disagreement-gate}"
MASTER_PORT="${MASTER_PORT:-29600}"

deepspeed --master_port "${MASTER_PORT}" --include localhost:0,1,2,3 "${REPO_ROOT}/llava/train/train_mem_router.py" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --encoder_type clip_dino \
    --fusion_type film \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_residual_scaling_coef False \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_text_enc True \
    --num_layers_to_choose 1 \
    --use_text_conditioned_router True \
    --text_conditioning attention \
    --text_attention_type qk \
    --text_attention_dim 1024 \
    --attn_level "hybrid" \
    --hybrid_reserve_mode disagreement_gate \
    --hybrid_disagreement_metric kl_local_global \
    --use_text_only_gating False \
    --text_pool mean \
    --text_scale 1.0 \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 436 \
    --save_total_limit 100 \
    --mm_projector_lr 1e-3 \
    --router_lr 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --force_reserve_layer True \
    --reserve_layer -2 \
    --learnable_gate False \
    --confidence_gate False \
    --confidence_gate_threshold 0.6 \
    --confidence_gate_slope 10.0 \
    --save_router_stats True \
    --save_steps_router_stats 5 \
    --report_to wandb

    # --router_lr 2e-4 \ - originally

python "${REPO_ROOT}/scripts/v1_5/mol_train_gpu.py" --gpus 0,1,2,3
