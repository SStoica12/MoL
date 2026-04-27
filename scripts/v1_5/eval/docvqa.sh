#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
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
CHECKPOINT_ROOT="/shared/nas2/sstoica2/MoL/checkpoints"
MODEL_PATHS=(
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_dino_no_router"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_2e-4_text-cond-attn-token-router_dino_hybrid_corrected_k=2"
)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/model_paths_override.sh"
load_model_paths_override

BENCHMARK_DIR="/shared/nas2/jk100/MoL/playground/data/eval/DocVQA"
HF_DATASET="${HF_DATASET:-lmms-lab/DocVQA}"
HF_CONFIG="${HF_CONFIG:-DocVQA}"
HF_REVISION="${HF_REVISION:-main}"
HF_TOKEN="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"
PARQUET_DIR="${PARQUET_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
SPLIT="${SPLIT:-validation}"
QUESTION_FILE="${QUESTION_FILE:-}"

CKPT_NAME="${CKPT_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ANSWERS_FILE="${ANSWERS_FILE:-}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
DETAIL_FILE="${DETAIL_FILE:-}"

usage() {
    cat <<EOF
Usage: bash scripts/v1_5/eval/docvqa.sh [options]

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
  --hf-dataset REPO
  --hf-config NAME
  --hf-revision REV
  --hf-token TOKEN
  --parquet-dir DIR
  --cache-dir DIR
  --split NAME
  --question-file PATH
  --ckpt-name NAME
  --output-dir DIR
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
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --hf-dataset) HF_DATASET="$2"; shift 2 ;;
        --hf-config) HF_CONFIG="$2"; shift 2 ;;
        --hf-revision) HF_REVISION="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --parquet-dir) PARQUET_DIR="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --question-file) QUESTION_FILE="$2"; shift 2 ;;
        --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
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

if [[ -z "${QUESTION_FILE}" ]]; then
    QUESTION_FILE="${BENCHMARK_DIR}/${SPLIT}.jsonl"
fi

CONVERT_ARGS=(
    --benchmark-dir "${BENCHMARK_DIR}"
    --split "${SPLIT}"
    --output-file "${QUESTION_FILE}"
    --hf-config "${HF_CONFIG}"
)

if [[ -n "${PARQUET_DIR}" ]]; then
    CONVERT_ARGS+=(--parquet-dir "${PARQUET_DIR}")
else
    CONVERT_ARGS+=(--hf-dataset "${HF_DATASET}" --hf-revision "${HF_REVISION}")
    if [[ -n "${HF_TOKEN}" ]]; then
        CONVERT_ARGS+=(--hf-token "${HF_TOKEN}")
    fi
    if [[ -n "${CACHE_DIR}" ]]; then
        CONVERT_ARGS+=(--cache-dir "${CACHE_DIR}")
    fi
fi

python scripts/convert_docvqa_for_eval.py "${CONVERT_ARGS[@]}"

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
        --image-folder "${BENCHMARK_DIR}" \
        --answers-file "${answers_file}" \
        --temperature "${TEMPERATURE}" \
        "${TOP_P_ARGS[@]}" \
        --num_beams "${NUM_BEAMS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --conv-mode "${CONV_MODE}" \
         --mm_use_text_enc \
        --attn_level "hybrid" \
        --use_router \
        --hybrid_reserve_mode disagreement_gate \
        --hybrid_disagreement_metric kl_local_global \
        --num_layers_to_choose 2 \
        --force_reserve_layer \
        --encoder_type dinov2_txt \

    python llava/eval/eval_docvqa.py \
        --question-file "${QUESTION_FILE}" \
        --result-file "${answers_file}" \
        --output-file "${summary_file}" \
        --detail-file "${detail_file}"
}

if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
    output_root="${OUTPUT_DIR:-${BENCHMARK_DIR}/answers/${SPLIT}}"
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
        OUTPUT_DIR="${BENCHMARK_DIR}/answers/${SPLIT}/${CKPT_NAME}"
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
