#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --mm_use_text_enc \
    --benchmark pope \
    --use_route True \
    --attn_level "patch_layer" \
    --force_reserve_layer True \

python llava/eval/eval_pope.py \
    --annotation-dir /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/coco \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
