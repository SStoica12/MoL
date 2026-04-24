#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED_ROOT="${SEED_ROOT:-./playground/data/eval/seed_bench}"
SEED_OUT_ROOT="${SEED_OUT_ROOT:-./playground/data/eval/seed_bench_out}"
QUESTION_FILE="${QUESTION_FILE:-${SEED_ROOT}/llava-seed-bench.jsonl}"
ANNOTATION_FILE="${ANNOTATION_FILE:-}"
IMAGE_DIR="${IMAGE_DIR:-${SEED_ROOT}/SEED-Bench-image}"
VIDEO_FRAME_DIR="${VIDEO_FRAME_DIR:-${SEED_ROOT}/SEED-Bench-video-image}"
FORCE_REBUILD_QUESTIONS="${FORCE_REBUILD_QUESTIONS:-0}"
PREPARE_VIDEO_FRAMES="${PREPARE_VIDEO_FRAMES:-0}"
INFERENCE_ONLY="${INFERENCE_ONLY:-0}"
SOMETHING_SOMETHING_DIR="${SOMETHING_SOMETHING_DIR:-${SEED_ROOT}/videos/20bn-something-something-v2}"
EPIC_KITCHENS_DIR="${EPIC_KITCHENS_DIR:-${SEED_ROOT}/videos/EPIC-KITCHENS}"
BREAKFAST_DIR="${BREAKFAST_DIR:-${SEED_ROOT}/videos/BreakfastII_15fps_qvga_sync}"

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
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_no-delta"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2_no-delta"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4_no-delta"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune_k2"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune_k4"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_lr1e-5-pretrain_lr2e-5-finetune_no_reserve"
)

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/model_paths_override.sh"
load_model_paths_override

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "${gpu_list}"
CHUNKS=${#GPULIST[@]}

resolve_annotation_file() {
    if [[ -n "${ANNOTATION_FILE}" && -f "${ANNOTATION_FILE}" ]]; then
        printf '%s\n' "${ANNOTATION_FILE}"
        return 0
    fi

    local candidates=(
        "${SEED_ROOT}/SEED-Bench.json"
        "${SEED_ROOT}/SEED-Bench_v1_level1_2_3.json"
        "${SEED_ROOT}/SEED-Bench_v2_level1_2_3.json"
        "${SEED_ROOT}/SEED-Bench-2/SEED-Bench_v2_level1_2_3.json"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

ANNOTATION_FILE="$(resolve_annotation_file || true)"

if [[ "${PREPARE_VIDEO_FRAMES}" == "1" ]]; then
    if [[ -z "${ANNOTATION_FILE}" ]]; then
        echo "Cannot extract SEED-Bench video frames without an annotation file." >&2
        exit 1
    fi
    python scripts/extract_seed_video_frames.py \
        --annotation-file "${ANNOTATION_FILE}" \
        --output-dir "${VIDEO_FRAME_DIR}" \
        --something-something-dir "${SOMETHING_SOMETHING_DIR}" \
        --epic-kitchens-dir "${EPIC_KITCHENS_DIR}" \
        --breakfast-dir "${BREAKFAST_DIR}"
fi

if [[ "${FORCE_REBUILD_QUESTIONS}" == "1" || ! -s "${QUESTION_FILE}" ]]; then
    if [[ -z "${ANNOTATION_FILE}" ]]; then
        echo "Cannot build ${QUESTION_FILE} without a SEED-Bench annotation file." >&2
        exit 1
    fi
    python scripts/convert_seedbench_for_eval.py \
        --seed-root "${SEED_ROOT}" \
        --annotation-file "${ANNOTATION_FILE}" \
        --image-dir "${IMAGE_DIR}" \
        --video-frame-dir "${VIDEO_FRAME_DIR}" \
        --output-file "${QUESTION_FILE}"
else
    echo "Reusing existing SEED-Bench question file: ${QUESTION_FILE}"
fi

if [[ ! -s "${QUESTION_FILE}" ]]; then
    echo "SEED-Bench question file is missing or empty: ${QUESTION_FILE}" >&2
    exit 1
fi

CAN_EXPORT_AND_SCORE=1
if [[ -z "${ANNOTATION_FILE}" || ! -f "${ANNOTATION_FILE}" ]]; then
    if [[ "${INFERENCE_ONLY}" == "1" ]]; then
        CAN_EXPORT_AND_SCORE=0
        echo "SEED-Bench annotation file not found. Running inference only with existing question file." >&2
        echo "Submission export and scored summary will be skipped." >&2
    else
        echo "SEED-Bench annotation file is required for evaluation and submission export." >&2
        echo "Set INFERENCE_ONLY=1 to run inference from ${QUESTION_FILE} without summary/submission export." >&2
        exit 1
    fi
fi

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

    if [[ "${CAN_EXPORT_AND_SCORE}" == "1" ]]; then
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
    else
        python llava/eval/eval_benchmark_summary.py \
            --dataset seed \
            --result-file "${output_file}" \
            --output-file "${summary_file}"
    fi
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
