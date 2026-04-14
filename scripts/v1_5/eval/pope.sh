#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /shared/nas2/sstoica2/MoL/checkpoints/llava-v1.5-13b-moe-finetune_dino_no_router \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --benchmark pope \
    --use_route False \

python llava/eval/eval_pope.py \
    --annotation-dir /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/coco \
    --question-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /shared/nas2/sstoica2/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
