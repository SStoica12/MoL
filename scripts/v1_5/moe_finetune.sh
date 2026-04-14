#!/bin/bash
export DS_BUILD_CPU_ADAM=0

export WANDB_ENTITY="uiucnlp"
export WANDB_PROJECT="mixture-of-layers"
export WANDB_NAME="MoL_router_choose_layers_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune"
export WANDB_MODE="online"  # ensure it uploads

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero3_offload.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero3.json}"
DATA_PATH="${DATA_PATH:-/shared/nas2/sstoica2/LLaVA/playground/data/eval/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/shared/nas2/sstoica2/LLaVA/playground/data/eval}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-13b-moe-pretrain_1e-5_text-cond-attn-hybrid-router_pretrain_disagreement-gate}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune}"
MASTER_PORT="${MASTER_PORT:-29600}"

deepspeed "${REPO_ROOT}/llava/train/train_mem_router.py" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_DIR}/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --freeze_residual_scaling_coef True \
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
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --router_lr 2e-5 \
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

    # --router_lr 2e-5 \ - originally

python "${REPO_ROOT}/scripts/v1_5/mol_train_gpu.py" --gpus 0,1,2,3
