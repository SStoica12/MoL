# #!/bin/bash

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path lmsys/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


#!/bin/bash
export DS_BUILD_CPU_ADAM=1

#!/bin/bash
export WANDB_ENTITY="uiucnlp"
export WANDB_PROJECT="mixture-of-layers"
export WANDB_NAME="MoL_no_router_finetune_dino"
export WANDB_MODE="online"  # ensure it uploads

NUM_LAYERS_TO_CHOOSE=2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEEPSPEED_CONFIG=/shared/nas2/sstoica2/MoL/scripts/zero3_offload.json
# DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero3.json}"
DATA_PATH="${DATA_PATH:-/shared/nas2/sstoica2/LLaVA/playground/data/eval/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/shared/nas2/sstoica2/LLaVA/playground/data/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/llava-v1.5-13b-moe-finetune_dino_no_router}"

# deepspeed --include localhost:1
#CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "${REPO_ROOT}/llava/train/train_mem.py" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_dino_mm_mlp_adapter /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-dino-pretrain-no_router/dino_mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --freeze_residual_scaling_coef True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_text_enc False \
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
    --save_steps 4000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --encoder_type "dinov2_txt"
curl -X POST -H 'Content-type: application/json' --data '{"text":"router done"}' https://hooks.slack.com/services/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
python /shared/nas2/jk100/MoL/scripts/v1_5/mol_train_gpu.py --gpus 0,1,2,3 --memory 70
