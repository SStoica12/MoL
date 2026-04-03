#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --mm_use_text_enc \
    --attn_level "patch_layer" \
    --benchmark textvqa \
    --use_route True \
    --force_reserve_layer True \
    --encoder_type dinov2_txt

python -m llava.eval.eval_textvqa \
    --annotation-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
