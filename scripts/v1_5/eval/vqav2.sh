#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
SPLIT="${SPLIT:-llava_vqav2_mscoco_test-dev2015}"
VQAV2_ROOT="${VQAV2_ROOT:-./playground/data/eval/vqav2}"
VQAV2_OUT_ROOT="${VQAV2_OUT_ROOT:-./playground/data/eval/vqav2_out}"
QUESTION_FILE="${QUESTION_FILE:-${VQAV2_ROOT}/${SPLIT}.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${VQAV2_ROOT}/test2015}"
VQAV2_ANNOTATION_FILE="${VQAV2_ANNOTATION_FILE:-}"

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
)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${gpu_list}"
CHUNKS=${#GPULIST[@]}

run_eval_for_model() {
    local model_path="$1"
    local ckpt_name="$2"
    local output_dir="${VQAV2_OUT_ROOT}/answers/${SPLIT}/${ckpt_name}"
    local output_file="${output_dir}/merge.jsonl"
    local summary_file="${output_dir}/summary.json"
    local detail_file="${output_dir}/details.jsonl"

    mkdir -p "${output_dir}"

    echo "=== Evaluating ${model_path} ==="

    for idx in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$idx]} python -m llava.eval.model_vqa_loader \
            --model-path "${model_path}" \
            --question-file "${QUESTION_FILE}" \
            --image-folder "${IMAGE_FOLDER}" \
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

    if [[ ! -f "${VQAV2_OUT_ROOT}/llava_vqav2_mscoco_test2015.jsonl" ]]; then
        ln -sf "${VQAV2_ROOT}/llava_vqav2_mscoco_test2015.jsonl" "${VQAV2_OUT_ROOT}/llava_vqav2_mscoco_test2015.jsonl"
    fi
    python scripts/convert_vqav2_for_submission.py --split "${SPLIT}" --ckpt "${ckpt_name}" --dir "${VQAV2_OUT_ROOT}"

    eval_args=(
        --dataset vqav2
        --result-file "${output_file}"
        --output-file "${summary_file}"
        --detail-file "${detail_file}"
    )
    if [[ -n "${VQAV2_ANNOTATION_FILE}" ]]; then
        eval_args+=(--annotation-file "${VQAV2_ANNOTATION_FILE}")
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
