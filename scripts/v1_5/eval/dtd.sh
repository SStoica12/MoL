#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# Optional batch mode:
# Paste checkpoint directories here to evaluate multiple models in one run.
# If non-empty, this list takes precedence over --model-path / MODEL_PATH.
CHECKPOINT_ROOT="/shared/nas2/jk100/MoL/checkpoints"
MODEL_PATHS=(
    "liuhaotian/llava-v1.5-13b"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-layer-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-film-router_confidence-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-film-router_learnable-gate_scaled-residual"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-patchlayer-softmix-router_learnable-gate_fixed-aux-loss_router-lr1e-5-pretrain_lr1e-5-finetune"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-patchlayer-softmix-router_learnable-gate_fixed-aux-loss_router-lr2e-5-pretrain_lr5e-6-finetune"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-patchlayer-softmix-router_learnable-gate_fixed-aux-loss"
)


DTD_ROOT="${DTD_ROOT:-./playground/data/eval/dtd/dtd}"
SPLIT="${SPLIT:-test}"
SPLIT_IDS="${SPLIT_IDS:-1}"   # space-separated, e.g. "1 2 3 4 5 6 7 8 9 10"
ALL_SPLITS="${ALL_SPLITS:-false}"

CKPT_NAME="${CKPT_NAME:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
QUESTION_FILE="${QUESTION_FILE:-}"
ANSWERS_FILE="${ANSWERS_FILE:-}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
DETAIL_FILE="${DETAIL_FILE:-}"

usage() {
    cat <<EOF
Usage: bash scripts/v1_5/eval/dtd.sh [options]

Options:
  --model-path PATH
  (edit MODEL_PATHS array in this script for multi-checkpoint batch mode)
  --model-base PATH
  --conv-mode MODE
  --temperature FLOAT
  --top-p FLOAT
  --num-beams INT
  --max-new-tokens INT
  --dtd-root DIR
  --split train|val|test
  --split-ids "1 2 3"     (quoted space-separated list)
  --all-splits            (use split ids 1..10)
  --ckpt-name NAME
  --run-name NAME
  --output-dir DIR
  --question-file PATH
  --answers-file PATH
  --summary-file PATH
  --detail-file PATH
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
        --dtd-root) DTD_ROOT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --split-ids) SPLIT_IDS="$2"; shift 2 ;;
        --all-splits) ALL_SPLITS="true"; shift 1 ;;
        --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --question-file) QUESTION_FILE="$2"; shift 2 ;;
        --answers-file) ANSWERS_FILE="$2"; shift 2 ;;
        --summary-file) SUMMARY_FILE="$2"; shift 2 ;;
        --detail-file) DETAIL_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${RUN_NAME}" ]]; then
    if [[ "${ALL_SPLITS}" == "true" ]]; then
        RUN_NAME="${SPLIT}_1-10"
    else
        RUN_NAME="${SPLIT}_$(echo "${SPLIT_IDS}" | tr ' ' '-')"
    fi
fi

if [[ -z "${QUESTION_FILE}" ]]; then
    QUESTION_FILE="${DTD_ROOT}/answers/${RUN_NAME}/questions.jsonl"
fi

mkdir -p "$(dirname "${QUESTION_FILE}")"

CONVERT_ARGS=(
    --dtd-root "${DTD_ROOT}"
    --split "${SPLIT}"
    --output-file "${QUESTION_FILE}"
)

if [[ "${ALL_SPLITS}" == "true" ]]; then
    CONVERT_ARGS+=(--all-splits)
else
    # shellcheck disable=SC2206
    SPLIT_IDS_ARR=(${SPLIT_IDS})
    CONVERT_ARGS+=(--split-ids "${SPLIT_IDS_ARR[@]}")
fi

python scripts/convert_dtd_for_eval.py "${CONVERT_ARGS[@]}"

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

    local output_dir
    output_dir="$(dirname "${answers_file}")"
    mkdir -p "${output_dir}"

    echo "=== Evaluating ${model_path} ==="
    echo "Output dir: ${output_dir}"

    python -m llava.eval.model_vqa_loader \
        --model-path "${model_path}" \
        "${BASE_ARGS[@]}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${DTD_ROOT}" \
        --answers-file "${answers_file}" \
        --temperature "${TEMPERATURE}" \
        "${TOP_P_ARGS[@]}" \
        --num_beams "${NUM_BEAMS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --conv-mode "${CONV_MODE}"

    python llava/eval/eval_dtd.py \
        --question-file "${QUESTION_FILE}" \
        --result-file "${answers_file}" \
        --output-file "${summary_file}" \
        --detail-file "${detail_file}"
}

if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
    output_root="${OUTPUT_DIR:-${DTD_ROOT}/answers/${RUN_NAME}}"
    mkdir -p "${output_root}"
    for model_path in "${MODEL_PATHS[@]}"; do
        ckpt_name_local="$(basename "${model_path}")"
        answers_file_local="${output_root}/${ckpt_name_local}/predictions.jsonl"
        summary_file_local="${output_root}/${ckpt_name_local}/summary.json"
        detail_file_local="${output_root}/${ckpt_name_local}/details.jsonl"
        run_eval_for_model "${model_path}" "${answers_file_local}" "${summary_file_local}" "${detail_file_local}"
    done
else
    if [[ -z "${CKPT_NAME}" ]]; then
        CKPT_NAME="$(basename "${MODEL_PATH}")"
    fi
    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="${DTD_ROOT}/answers/${RUN_NAME}/${CKPT_NAME}"
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
    run_eval_for_model "${MODEL_PATH}" "${ANSWERS_FILE}" "${SUMMARY_FILE}" "${DETAIL_FILE}"
fi
