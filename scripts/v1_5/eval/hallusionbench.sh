#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Optional batch mode:
# Paste checkpoint directories here to evaluate multiple models in one run.
# If non-empty, this list takes precedence over --model-path / MODEL_PATH.
CHECKPOINT_ROOT="/shared/nas2/sstoica2/MoL/checkpoints"
MODEL_PATHS=(
    # "liuhaotian/llava-v1.5-13b"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_router_learnable-gate_scaled-residual"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-layer-router_learnable-gate_scaled-residual"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_scaled-residual"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-film-router_learnable-gate_scaled-residual"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k6"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_no-delta",
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_dino_patch_layer_corrected_k=2"
)

BENCHMARK_DIR="/shared/nas2/jk100/MoL/playground/data/eval/HallusionBench"
RAW_FILE="${RAW_FILE:-}"
QUESTION_FILE="${QUESTION_FILE:-}"

CKPT_NAME="${CKPT_NAME:-}"
OUTPUT_DIR="/shared/nas2/sstoica2/MoL/playground/data/eval"
ANSWERS_FILE="${ANSWERS_FILE:-}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
DETAIL_FILE="${DETAIL_FILE:-}"
EXPORT_RESULT_FILE="${EXPORT_RESULT_FILE:-}"

usage() {
    cat <<EOF
Usage: bash scripts/v1_5/eval/hallusionbench.sh [options]

Options:
  --model-path PATH
  (edit MODEL_PATHS array in this script for multi-checkpoint batch mode)
  --model-base PATH
  --conv-mode MODE
  --temperature FLOAT
  --top-p FLOAT
  --num-beams INT
  --max-new-tokens INT
  --benchmark-dir DIR
  --raw-file PATH
  --question-file PATH
  --ckpt-name NAME
  --output-dir DIR
  --answers-file PATH
  --summary-file PATH
  --detail-file PATH
  --export-result-file PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-base) MODEL_BASE="$2"; shift 2 ;;
        --conv-mode) CONV_MODE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --num-beams) NUM_BEAMS="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --raw-file) RAW_FILE="$2"; shift 2 ;;
        --question-file) QUESTION_FILE="$2"; shift 2 ;;
        --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --answers-file) ANSWERS_FILE="$2"; shift 2 ;;
        --summary-file) SUMMARY_FILE="$2"; shift 2 ;;
        --detail-file) DETAIL_FILE="$2"; shift 2 ;;
        --export-result-file) EXPORT_RESULT_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${RAW_FILE}" ]]; then
    RAW_FILE="${BENCHMARK_DIR}/HallusionBench.json"
fi

if [[ -z "${QUESTION_FILE}" ]]; then
    QUESTION_FILE="${BENCHMARK_DIR}/questions.jsonl"
fi

python scripts/convert_hallusionbench_for_eval.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --raw-file "${RAW_FILE}" \
    --output-file "${QUESTION_FILE}"

BASE_ARGS=()
if [[ -n "${MODEL_BASE}" ]]; then
    BASE_ARGS+=(--model-base "${MODEL_BASE}")
fi

TOP_P_ARGS=()
if [[ -n "${TOP_P}" ]]; then
    TOP_P_ARGS+=(--top_p "${TOP_P}")
fi

run_eval_for_model() {
    local model_path="$1"
    local answers_file="$2"
    local summary_file="$3"
    local detail_file="$4"
    local export_result_file="$5"

    local output_dir
    output_dir="$(dirname "${answers_file}")"
    mkdir -p "${output_dir}"

    echo "=== Evaluating ${model_path} ==="
    echo "Output dir: ${output_dir}"

    python -m llava.eval.model_vqa_loader \
        --model-path "${model_path}" \
        "${BASE_ARGS[@]}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${BENCHMARK_DIR}" \
        --answers-file "${answers_file}" \
        --temperature "${TEMPERATURE}" \
        "${TOP_P_ARGS[@]}" \
        --num_beams "${NUM_BEAMS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --conv-mode "${CONV_MODE}" \
        --benchmark hallusionbench \
        --mm_use_text_enc \
        --use_router True \
        --force_reserve_layer True \
        --attn_level "patch_layer" \
        --mm_use_text_enc \
        --num_layers_to_choose 2 \


    python llava/eval/eval_hallusionbench.py \
        --question-file "${QUESTION_FILE}" \
        --result-file "${answers_file}" \
        --output-file "${summary_file}" \
        --detail-file "${detail_file}" \
        --export-result-file "${export_result_file}"
}

if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
    output_root="${OUTPUT_DIR:-${BENCHMARK_DIR}/answers}"
    mkdir -p "${output_root}"
    for model_path in "${MODEL_PATHS[@]}"; do
        ckpt_name_local="$(basename "${model_path}")"
        answers_file_local="${output_root}/${ckpt_name_local}/predictions.jsonl"
        summary_file_local="${output_root}/${ckpt_name_local}/summary.json"
        detail_file_local="${output_root}/${ckpt_name_local}/details.jsonl"
        export_result_file_local="${output_root}/${ckpt_name_local}/HallusionBench_result.json"
        run_eval_for_model \
            "${model_path}" \
            "${answers_file_local}" \
            "${summary_file_local}" \
            "${detail_file_local}" \
            "${export_result_file_local}"
    done
else
    if [[ -z "${CKPT_NAME}" ]]; then
        CKPT_NAME="$(basename "${MODEL_PATH}")"
    fi
    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="${BENCHMARK_DIR}/answers/${CKPT_NAME}"
    fi
    if [[ -z "${ANSWERS_FILE}" ]]; then
        ANSWERS_FILE="${OUTPUT_DIR}/predictions.jsonl"
    fi
    if [[ -z "${SUMMARY_FILE}" ]]; then
        SUMMARY_FILE="${OUTPUT_DIR}/summary.json"
    fi
    if [[ -z "${DETAIL_FILE}" ]]; then
        DETAIL_FILE="${OUTPUT_DIR}/details.jsonl"
    fi
    if [[ -z "${EXPORT_RESULT_FILE}" ]]; then
        EXPORT_RESULT_FILE="${OUTPUT_DIR}/HallusionBench_result.json"
    fi
    run_eval_for_model \
        "${MODEL_PATH}" \
        "${ANSWERS_FILE}" \
        "${SUMMARY_FILE}" \
        "${DETAIL_FILE}" \
        "${EXPORT_RESULT_FILE}"
fi