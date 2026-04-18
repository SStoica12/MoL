#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED_ROOT="${SEED_ROOT:-./playground/data/eval/seed_bench}"
SEED_OUT_ROOT="${SEED_OUT_ROOT:-./playground/data/eval/seed_bench_out}"
QUESTION_FILE="${QUESTION_FILE:-${SEED_ROOT}/llava-seed-bench.jsonl}"
ANNOTATION_FILE="${ANNOTATION_FILE:-${SEED_ROOT}/SEED-Bench.json}"

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

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${gpu_list}"
CHUNKS=${#GPULIST[@]}

run_eval_for_model() {
    local model_path="$1"
    local ckpt_name="$2"
    local output_dir="${SEED_OUT_ROOT}/answers/${ckpt_name}"
    local output_file="${output_dir}/merge.jsonl"
    local upload_file="${SEED_OUT_ROOT}/answers_upload/${ckpt_name}.jsonl"
    local summary_file="${output_dir}/summary.json"
    local detail_file="${output_dir}/details.jsonl"

    mkdir -p "${output_dir}" "$(dirname "${upload_file}")"

    echo "=== Evaluating ${model_path} ==="

    for idx in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$idx]} python -m llava.eval.model_vqa_loader \
            --model-path "${model_path}" \
            --question-file "${QUESTION_FILE}" \
            --image-folder "${SEED_ROOT}" \
            --answers-file "${output_dir}/${CHUNKS}_${idx}.jsonl" \
            --num-chunks "${CHUNKS}" \
            --chunk-idx "${idx}" \
            --temperature "${TEMPERATURE}" \
            --conv-mode "${CONV_MODE}" &
    done

    wait

    : > "${output_file}"
    for idx in $(seq 0 $((CHUNKS-1))); do
        cat "${output_dir}/${CHUNKS}_${idx}.jsonl" >> "${output_file}"
    done

    python scripts/convert_seed_for_submission.py \
        --annotation-file "${ANNOTATION_FILE}" \
        --result-file "${output_file}" \
        --result-upload-file "${upload_file}"

    python llava/eval/eval_benchmark_summary.py \
        --dataset seed \
        --result-file "${output_file}" \
        --annotation-file "${ANNOTATION_FILE}" \
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
