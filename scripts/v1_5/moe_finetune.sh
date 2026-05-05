#!/bin/bash
export DS_BUILD_CPU_ADAM=0

export WANDB_ENTITY="uiucnlp"
export WANDB_PROJECT="mixture-of-layers"
export WANDB_NAME="MoL_router_choose_layers_finetune_text-cond-attn-token-router-gate_dino_layer_k=${NUM_LAYERS_TO_CHOOSE}"
export WANDB_MODE="online"  # ensure it uploads

NUM_LAYERS_TO_CHOOSE=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEEPSPEED_CONFIG=/shared/nas2/sstoica2/MoL/scripts/zero3_offload.json
# DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero3.json}"
DATA_PATH="${DATA_PATH:-/shared/nas2/sstoica2/LLaVA/playground/data/eval/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/shared/nas2/sstoica2/LLaVA/playground/data/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_dino_layer_k=${NUM_LAYERS_TO_CHOOSE}}"

# deepspeed --include localhost:1
#CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed
deepspeed --include localhost:0,1,2,3 --master_port 29501 "${REPO_ROOT}/llava/train/train_mem_router.py" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_dino_mm_mlp_adapter /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-pretrain_text-cond-attn-token-router_dino_layer_k=1/dino_mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --freeze_residual_scaling_coef True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_text_enc True \
    --num_layers_to_choose $NUM_LAYERS_TO_CHOOSE \
    --use_text_conditioned_router True \
    --text_conditioning attention \
    --text_attention_type qk \
    --text_attention_dim 1024 \
    --attn_level "token" \
    --hybrid_reserve_mode local_gate \
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
    --save_steps 2000 \
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
    --learnable_gate True \
    --confidence_gate False \
    --confidence_gate_threshold 0.6 \
    --confidence_gate_slope 10.0 \
    --save_router_stats True \
    --save_steps_router_stats 5 \
    --report_to wandb \
    --encoder_type "dinov2_txt"
curl -X POST -H 'Content-type: application/json' --data '{"text":"router done"}' https://hooks.slack.com/services/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
python /shared/nas2/jk100/MoL/scripts/v1_5/mol_train_gpu.py --gpus 0,1,2,3 --memory 70

#   --attn_level "patch_layer" \