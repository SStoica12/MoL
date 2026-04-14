#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_dino_no_router \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --attn_level "patch_layer" \
    --benchmark textvqa \
    --use_route False \
    --num_layers_to_choose 2 \
    --force_reserve_layer True \
    --encoder_type dinov2_txt

python -m llava.eval.eval_textvqa \
    --annotation-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl