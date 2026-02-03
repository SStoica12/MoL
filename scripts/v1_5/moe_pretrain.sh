#!/bin/bash
export WANDB_NAME="MoL_router_choose_layers_pretrain"
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed /shared/nas2/sstoica2/MoL/llava/train/train_mem_router.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path /shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/blip_laion_cc_sbu_558k.json \
    --image_folder /shared/nas/data/m1/sstoica2/.cache/huggingface/.cache/huggingface/datasets--Blip/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_residual_scaling_coef False \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --num_layers_to_choose 1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-pretrain_2e-4_router-debuge \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 100 \
    --mm_projector_lr 1e-3 \
    --router_lr 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --force_reserve_layer False \
    --reserve_layer -2 \
    --save_router_stats True \
    --save_steps_router_stats 5 \
    --report_to wandb

python /shared/nas/data/m1/sstoica2/PAPO_copy/debug_gpu.py --gpus 0,1,2,3 --memory 70