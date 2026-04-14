#!/bin/bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# Optional batch mode:
# Paste checkpoint directories here to evaluate multiple models in one run.
# If non-empty, this list takes precedence over --model-path / MODEL_PATH.
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

BENCHMARK_DIR="${BENCHMARK_DIR:-./playground/data/eval/mmvet}"
LEGACY_BENCHMARK_DIR="./playground/data/eval/mm-vet"
ARCHIVE_FILE="${ARCHIVE_FILE:-}"
ANNOTATION_FILE="${ANNOTATION_FILE:-}"
HF_DATASET="${HF_DATASET:-lmms-lab/MMVet}"
HF_REVISION="${HF_REVISION:-main}"
HF_TOKEN="${HF_TOKEN:-}"
PARQUET_DIR="${PARQUET_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
SPLIT="${SPLIT:-test}"
QUESTION_FILE="${QUESTION_FILE:-}"

CKPT_NAME="${CKPT_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ANSWERS_FILE="${ANSWERS_FILE:-}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
DETAIL_FILE="${DETAIL_FILE:-}"
GRADE_FILE="${GRADE_FILE:-}"
RESULT_EXPORT_FILE="${RESULT_EXPORT_FILE:-}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4-0613}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_NUM_RUNS="${JUDGE_NUM_RUNS:-1}"

usage() {
    cat <<EOF
Usage: bash scripts/v1_5/eval/mmvet.sh [options]

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
  --archive-file PATH
  --annotation-file PATH
  --hf-dataset REPO
  --hf-revision REV
  --hf-token TOKEN
  --parquet-dir DIR
  --cache-dir DIR
  --split NAME
  --question-file PATH
  --judge-model MODEL
  --judge-api-key KEY
  --judge-base-url URL
  --judge-num-runs INT
  --ckpt-name NAME
  --output-dir DIR
  --answers-file PATH
  --summary-file PATH
  --detail-file PATH
  --grade-file PATH
  --result-export-file PATH
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
        --archive-file) ARCHIVE_FILE="$2"; shift 2 ;;
        --annotation-file) ANNOTATION_FILE="$2"; shift 2 ;;
        --hf-dataset) HF_DATASET="$2"; shift 2 ;;
        --hf-revision) HF_REVISION="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --parquet-dir) PARQUET_DIR="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --question-file) QUESTION_FILE="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-api-key) JUDGE_API_KEY="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --judge-num-runs) JUDGE_NUM_RUNS="$2"; shift 2 ;;
        --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --answers-file) ANSWERS_FILE="$2"; shift 2 ;;
        --summary-file) SUMMARY_FILE="$2"; shift 2 ;;
        --detail-file) DETAIL_FILE="$2"; shift 2 ;;
        --grade-file) GRADE_FILE="$2"; shift 2 ;;
        --result-export-file) RESULT_EXPORT_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${BENCHMARK_DIR}" == "./playground/data/eval/mmvet" && ! -d "${BENCHMARK_DIR}" && -d "${LEGACY_BENCHMARK_DIR}" ]]; then
    BENCHMARK_DIR="${LEGACY_BENCHMARK_DIR}"
fi

if [[ -z "${QUESTION_FILE}" ]]; then
    QUESTION_FILE="${BENCHMARK_DIR}/${SPLIT}.jsonl"
fi

prep_annotation_file="${ANNOTATION_FILE}"
if [[ -z "${prep_annotation_file}" ]]; then
    prep_annotation_file="${BENCHMARK_DIR}/mm-vet.json"
fi

PREP_ARGS=(
    --benchmark-dir "${BENCHMARK_DIR}"
    --output-file "${QUESTION_FILE}"
    --export-annotation-file "${prep_annotation_file}"
    --split "${SPLIT}"
)
if [[ -n "${ARCHIVE_FILE}" ]]; then
    PREP_ARGS+=(--archive-file "${ARCHIVE_FILE}")
fi
if [[ -n "${ANNOTATION_FILE}" ]]; then
    PREP_ARGS+=(--annotation-file "${ANNOTATION_FILE}")
fi
if [[ -n "${PARQUET_DIR}" ]]; then
    PREP_ARGS+=(--parquet-dir "${PARQUET_DIR}")
else
    PREP_ARGS+=(--hf-dataset "${HF_DATASET}" --hf-revision "${HF_REVISION}")
    if [[ -n "${HF_TOKEN}" ]]; then
        PREP_ARGS+=(--hf-token "${HF_TOKEN}")
    fi
    if [[ -n "${CACHE_DIR}" ]]; then
        PREP_ARGS+=(--cache-dir "${CACHE_DIR}")
    fi
fi
python scripts/prepare_mmvet_for_eval.py "${PREP_ARGS[@]}"

if [[ -z "${ANNOTATION_FILE}" ]]; then
    if [[ -f "${BENCHMARK_DIR}/mm-vet.json" ]]; then
        ANNOTATION_FILE="${BENCHMARK_DIR}/mm-vet.json"
    else
        ANNOTATION_FILE="$(find "${BENCHMARK_DIR}" -name 'mm-vet.json' | head -n 1)"
    fi
fi

BASE_ARGS=()
if [[ -n "${MODEL_BASE}" ]]; then
    BASE_ARGS+=(--model-base "${MODEL_BASE}")
fi

TOP_P_ARGS=()
if [[ -n "${TOP_P}" ]]; then
    TOP_P_ARGS+=(--top_p "${TOP_P}")
fi

JUDGE_ARGS=()
if [[ -n "${JUDGE_MODEL}" ]]; then
    JUDGE_ARGS+=(--judge-model "${JUDGE_MODEL}")
fi
if [[ -n "${JUDGE_API_KEY}" ]]; then
    JUDGE_ARGS+=(--judge-api-key "${JUDGE_API_KEY}")
fi
if [[ -n "${JUDGE_BASE_URL}" ]]; then
    JUDGE_ARGS+=(--judge-base-url "${JUDGE_BASE_URL}")
fi

run_eval_for_model() {
    local model_path="$1"
    local answers_file="$2"
    local summary_file="$3"
    local detail_file="$4"
    local grade_file="$5"
    local result_export_file="$6"

    local output_dir
    output_dir="$(dirname "${answers_file}")"
    mkdir -p "${output_dir}" "$(dirname "${result_export_file}")"

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
        --conv-mode "${CONV_MODE}"

    python scripts/convert_mmvet_for_eval.py \
        --src "${answers_file}" \
        --dst "${result_export_file}"

    python llava/eval/eval_mmvet.py \
        --annotation-file "${ANNOTATION_FILE}" \
        --result-file "${answers_file}" \
        --grade-file "${grade_file}" \
        --output-file "${summary_file}" \
        --detail-file "${detail_file}" \
        "${JUDGE_ARGS[@]}" \
        --num-runs "${JUDGE_NUM_RUNS}"
}

if [[ ${#MODEL_PATHS[@]} -gt 0 ]]; then
    output_root="${OUTPUT_DIR:-${BENCHMARK_DIR}/answers}"
    result_root="${BENCHMARK_DIR}/results"
    mkdir -p "${output_root}" "${result_root}"
    for model_path in "${MODEL_PATHS[@]}"; do
        ckpt_name_local="$(basename "${model_path}")"
        answers_file_local="${output_root}/${ckpt_name_local}/predictions.jsonl"
        summary_file_local="${output_root}/${ckpt_name_local}/summary.json"
        detail_file_local="${output_root}/${ckpt_name_local}/details.jsonl"
        grade_file_local="${output_root}/${ckpt_name_local}/grades.json"
        result_export_file_local="${result_root}/${ckpt_name_local}.json"
        run_eval_for_model \
            "${model_path}" \
            "${answers_file_local}" \
            "${summary_file_local}" \
            "${detail_file_local}" \
            "${grade_file_local}" \
            "${result_export_file_local}"
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
    if [[ -z "${GRADE_FILE}" ]]; then
        GRADE_FILE="${OUTPUT_DIR}/grades.json"
    fi
    if [[ -z "${RESULT_EXPORT_FILE}" ]]; then
        RESULT_EXPORT_FILE="${BENCHMARK_DIR}/results/${CKPT_NAME}.json"
    fi
    run_eval_for_model \
        "${MODEL_PATH}" \
        "${ANSWERS_FILE}" \
        "${SUMMARY_FILE}" \
        "${DETAIL_FILE}" \
        "${GRADE_FILE}" \
        "${RESULT_EXPORT_FILE}"
fi
