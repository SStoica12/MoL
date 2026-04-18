#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
SPLIT="${SPLIT:-mmbench_dev_20230712}"
MMBENCH_ROOT="${MMBENCH_ROOT:-./playground/data/eval/mmbench}"
MMBENCH_OUT_ROOT="${MMBENCH_OUT_ROOT:-./playground/data/eval/mmbench_out}"

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

    local answer_dir="${MMBENCH_OUT_ROOT}/answers/${SPLIT}"
    local answers_file="${answer_dir}/${ckpt_name}.jsonl"
    local upload_dir="${MMBENCH_OUT_ROOT}/answers_upload/${SPLIT}"
    local summary_file="${answer_dir}/${ckpt_name}_summary.json"
    local detail_file="${answer_dir}/${ckpt_name}_details.jsonl"

    mkdir -p "${answer_dir}" "${upload_dir}"

    echo "=== Evaluating ${model_path} ==="

    python -m llava.eval.model_vqa_mmbench \
        --model-path "${model_path}" \
        --question-file "${MMBENCH_ROOT}/${SPLIT}.tsv" \
        --answers-file "${answers_file}" \
        --single-pred-prompt \
        --temperature "${TEMPERATURE}" \
        --conv-mode "${CONV_MODE}"

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file "${MMBENCH_ROOT}/${SPLIT}.tsv" \
        --result-dir "${answer_dir}" \
        --upload-dir "${upload_dir}" \
        --experiment "${ckpt_name}"

    python llava/eval/eval_benchmark_summary.py \
        --dataset mmbench \
        --question-file "${MMBENCH_ROOT}/${SPLIT}.tsv" \
        --result-file "${answers_file}" \
        --output-file "${summary_file}" \
        --detail-file "${detail_file}"
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
