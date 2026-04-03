#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --mm_use_text_enc \
    --benchmark mmbench \
    --use_route True \
    --attn_level "patch_layer" \
    --force_reserve_layer True \

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /shared/nas2/sstoica2/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /shared/nas2/sstoica2/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
python /shared/nas/data/m1/sstoica2/PAPO_copy/debug_gpu.py --gpus 1 --memory 70
