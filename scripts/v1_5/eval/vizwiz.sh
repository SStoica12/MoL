#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
VIZWIZ_ROOT="${VIZWIZ_ROOT:-./playground/data/eval/vizwiz}"
VIZWIZ_OUT_ROOT="${VIZWIZ_OUT_ROOT:-./playground/data/eval/vizwiz_out}"
QUESTION_FILE="${QUESTION_FILE:-${VIZWIZ_ROOT}/llava_test.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${VIZWIZ_ROOT}/test}"
VIZWIZ_ANNOTATION_FILE="${VIZWIZ_ANNOTATION_FILE:-}"

# Optional batch mode:
# Paste checkpoint directories here to evaluate multiple models in one run.
# If non-empty, this list takes precedence over MODEL_PATH.
CHECKPOINT_ROOT="/shared/nas2/jk100/MoL/checkpoints"
MODEL_PATHS=(
    "liuhaotian/llava-v1.5-13b"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-layer-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-film-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k6"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_no-delta"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2_no-delta"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4_no-delta"
)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/model_paths_override.sh"
load_model_paths_override

run_eval_for_model() {
    local model_path="$1"
    local ckpt_name="$2"
    local answers_file="${VIZWIZ_OUT_ROOT}/answers/${ckpt_name}.jsonl"
    local upload_file="${VIZWIZ_OUT_ROOT}/answers_upload/${ckpt_name}.json"
    local summary_file="${VIZWIZ_OUT_ROOT}/answers/${ckpt_name}_summary.json"
    local detail_file="${VIZWIZ_OUT_ROOT}/answers/${ckpt_name}_details.jsonl"

    mkdir -p "$(dirname "${answers_file}")" "$(dirname "${upload_file}")"

    echo "=== Evaluating ${model_path} ==="

    python -m llava.eval.model_vqa_loader \
        --model-path "${model_path}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${answers_file}" \
        --temperature "${TEMPERATURE}" \
        --conv-mode "${CONV_MODE}"

    python scripts/convert_vizwiz_for_submission.py \
        --annotation-file "${QUESTION_FILE}" \
        --result-file "${answers_file}" \
        --result-upload-file "${upload_file}"

    eval_args=(
        --dataset vizwiz
        --question-file "${QUESTION_FILE}"
        --result-file "${answers_file}"
        --output-file "${summary_file}"
        --detail-file "${detail_file}"
    )
    if [[ -n "${VIZWIZ_ANNOTATION_FILE}" ]]; then
        eval_args+=(--annotation-file "${VIZWIZ_ANNOTATION_FILE}")
    fi
    python llava/eval/eval_benchmark_summary.py "${eval_args[@]}"
}

if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
    for model_path in "${MODEL_PATHS[@]}"; do
        ckpt_name="$(basename "${model_path%/}")"
        run_eval_for_model "${model_path}" "${ckpt_name}"
    done
else
    ckpt_name="$(basename "${MODEL_PATH%/}")"
    run_eval_for_model "${MODEL_PATH}" "${ckpt_name}"
fi
