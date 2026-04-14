#!/bin/bash

###########################################################################################################
# This script summarizes benchmark results for multiple checkpoints and benchmarks.

# HOW TO USE
# # # Only GQA (your exact use case)
# bash scripts/v1_5/eval/summarize_benchmark_outputs.sh gqa

# # All supported benchmarks
# bash scripts/v1_5/eval/summarize_benchmark_outputs.sh all

# # Any subset
# bash scripts/v1_5/eval/summarize_benchmark_outputs.sh mmbench gqa seed

###########################################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-llava/eval/eval_benchmark_summary.py}"

# GQA
GQA_ROOT="${GQA_ROOT:-./playground/data/eval/gqa}"
GQA_OUT_ROOT="${GQA_OUT_ROOT:-./playground/data/eval/gqa_out}"
GQA_SPLIT="${GQA_SPLIT:-llava_gqa_testdev_balanced}"
GQA_TIER="${GQA_TIER:-testdev_balanced}"
GQA_QUESTION_FILE="${GQA_QUESTION_FILE:-${GQA_ROOT}/questions1.2/${GQA_TIER}_questions.json}"

# MMBench
MMBENCH_ROOT="${MMBENCH_ROOT:-./playground/data/eval/mmbench}"
MMBENCH_OUT_ROOT="${MMBENCH_OUT_ROOT:-./playground/data/eval/mmbench_out}"
MMBENCH_SPLIT="${MMBENCH_SPLIT:-mmbench_dev_20230712}"
MMBENCH_QUESTION_FILE="${MMBENCH_QUESTION_FILE:-${MMBENCH_ROOT}/${MMBENCH_SPLIT}.tsv}"

# SEED
SEED_ROOT="${SEED_ROOT:-./playground/data/eval/seed_bench}"
SEED_OUT_ROOT="${SEED_OUT_ROOT:-./playground/data/eval/seed_bench_out}"
SEED_ANNOTATION_FILE="${SEED_ANNOTATION_FILE:-${SEED_ROOT}/SEED-Bench.json}"

# POPE
POPE_ROOT="${POPE_ROOT:-./playground/data/eval/pope}"
POPE_OUT_ROOT="${POPE_OUT_ROOT:-./playground/data/eval/pope_out}"
POPE_QUESTION_FILE="${POPE_QUESTION_FILE:-${POPE_ROOT}/llava_pope_test.jsonl}"
POPE_ANNOTATION_DIR="${POPE_ANNOTATION_DIR:-${POPE_ROOT}/coco}"

# TextVQA
TEXTVQA_ROOT="${TEXTVQA_ROOT:-./playground/data/eval/textvqa}"
TEXTVQA_OUT_ROOT="${TEXTVQA_OUT_ROOT:-./playground/data/eval/textvqa_out}"
TEXTVQA_ANNOTATION_FILE="${TEXTVQA_ANNOTATION_FILE:-${TEXTVQA_ROOT}/TextVQA_0.5.1_val.json}"

# MME
MME_ROOT="${MME_ROOT:-./playground/data/eval/MME}"
MME_OUT_ROOT="${MME_OUT_ROOT:-./playground/data/eval/mme_out}"
MME_QUESTION_FILE="${MME_QUESTION_FILE:-${MME_ROOT}/llava_mme.jsonl}"

# VizWiz
VIZWIZ_ROOT="${VIZWIZ_ROOT:-./playground/data/eval/vizwiz}"
VIZWIZ_OUT_ROOT="${VIZWIZ_OUT_ROOT:-./playground/data/eval/vizwiz_out}"
VIZWIZ_QUESTION_FILE="${VIZWIZ_QUESTION_FILE:-${VIZWIZ_ROOT}/llava_test.jsonl}"
VIZWIZ_ANNOTATION_FILE="${VIZWIZ_ANNOTATION_FILE:-}"

# VQAv2
VQAV2_OUT_ROOT="${VQAV2_OUT_ROOT:-./playground/data/eval/vqav2_out}"
VQAV2_SPLIT="${VQAV2_SPLIT:-llava_vqav2_mscoco_test-dev2015}"
VQAV2_ANNOTATION_FILE="${VQAV2_ANNOTATION_FILE:-}"

# HallusionBench
HALLUSIONBENCH_ROOT="${HALLUSIONBENCH_ROOT:-./playground/data/eval/HallusionBench}"
HALLUSIONBENCH_OUT_ROOT="${HALLUSIONBENCH_OUT_ROOT:-${HALLUSIONBENCH_ROOT}/answers}"
HALLUSIONBENCH_QUESTION_FILE="${HALLUSIONBENCH_QUESTION_FILE:-${HALLUSIONBENCH_ROOT}/questions.jsonl}"
HALLUSIONBENCH_EVAL_SCRIPT="${HALLUSIONBENCH_EVAL_SCRIPT:-llava/eval/eval_hallusionbench.py}"

# RealWorldQA
REALWORLDQA_ROOT="${REALWORLDQA_ROOT:-./playground/data/eval/RealWorldQA}"
REALWORLDQA_SPLIT="${REALWORLDQA_SPLIT:-test}"
REALWORLDQA_OUT_ROOT="${REALWORLDQA_OUT_ROOT:-${REALWORLDQA_ROOT}/answers/${REALWORLDQA_SPLIT}}"
REALWORLDQA_QUESTION_FILE="${REALWORLDQA_QUESTION_FILE:-${REALWORLDQA_ROOT}/${REALWORLDQA_SPLIT}.jsonl}"
REALWORLDQA_EVAL_SCRIPT="${REALWORLDQA_EVAL_SCRIPT:-llava/eval/eval_realworldqa.py}"

# ChartQA
CHARTQA_ROOT="${CHARTQA_ROOT:-./playground/data/eval/ChartQA}"
CHARTQA_SPLIT="${CHARTQA_SPLIT:-test}"
CHARTQA_OUT_ROOT="${CHARTQA_OUT_ROOT:-${CHARTQA_ROOT}/answers/${CHARTQA_SPLIT}}"
CHARTQA_QUESTION_FILE="${CHARTQA_QUESTION_FILE:-${CHARTQA_ROOT}/${CHARTQA_SPLIT}.jsonl}"
CHARTQA_EVAL_SCRIPT="${CHARTQA_EVAL_SCRIPT:-llava/eval/eval_chartqa.py}"

# HRBench4K
HRBENCH4K_ROOT="${HRBENCH4K_ROOT:-./playground/data/eval/HRBench4K}"
HRBENCH4K_OUT_ROOT="${HRBENCH4K_OUT_ROOT:-${HRBENCH4K_ROOT}/answers}"
HRBENCH4K_QUESTION_FILE="${HRBENCH4K_QUESTION_FILE:-${HRBENCH4K_ROOT}/questions.jsonl}"
HRBENCH4K_EVAL_SCRIPT="${HRBENCH4K_EVAL_SCRIPT:-llava/eval/eval_hrbench.py}"

# HRBench8K
HRBENCH8K_ROOT="${HRBENCH8K_ROOT:-./playground/data/eval/HRBench8K}"
HRBENCH8K_OUT_ROOT="${HRBENCH8K_OUT_ROOT:-${HRBENCH8K_ROOT}/answers}"
HRBENCH8K_QUESTION_FILE="${HRBENCH8K_QUESTION_FILE:-${HRBENCH8K_ROOT}/questions.jsonl}"
HRBENCH8K_EVAL_SCRIPT="${HRBENCH8K_EVAL_SCRIPT:-llava/eval/eval_hrbench.py}"

# MMStar
MMSTAR_ROOT="${MMSTAR_ROOT:-./playground/data/eval/MMStar}"
MMSTAR_SPLIT="${MMSTAR_SPLIT:-val}"
MMSTAR_OUT_ROOT="${MMSTAR_OUT_ROOT:-${MMSTAR_ROOT}/answers/${MMSTAR_SPLIT}}"
MMSTAR_QUESTION_FILE="${MMSTAR_QUESTION_FILE:-${MMSTAR_ROOT}/${MMSTAR_SPLIT}.jsonl}"
MMSTAR_EVAL_SCRIPT="${MMSTAR_EVAL_SCRIPT:-llava/eval/eval_mmstar.py}"

# MMVet
MMVET_ROOT="${MMVET_ROOT:-./playground/data/eval/mmvet}"
MMVET_OUT_ROOT="${MMVET_OUT_ROOT:-${MMVET_ROOT}/answers}"
MMVET_ANNOTATION_FILE="${MMVET_ANNOTATION_FILE:-${MMVET_ROOT}/mm-vet.json}"
MMVET_EVAL_SCRIPT="${MMVET_EVAL_SCRIPT:-llava/eval/eval_mmvet.py}"
MMVET_JUDGE_MODEL="${MMVET_JUDGE_MODEL:-${JUDGE_MODEL:-}}"
MMVET_JUDGE_API_KEY="${MMVET_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
MMVET_JUDGE_BASE_URL="${MMVET_JUDGE_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"
MMVET_JUDGE_NUM_RUNS="${MMVET_JUDGE_NUM_RUNS:-1}"

# MMVP
MMVP_ROOT="${MMVP_ROOT:-./playground/data/eval/MMVP}"
MMVP_OUT_ROOT="${MMVP_OUT_ROOT:-${MMVP_ROOT}/answers}"
MMVP_QUESTION_FILE="${MMVP_QUESTION_FILE:-${MMVP_ROOT}/questions.jsonl}"
MMVP_EVAL_SCRIPT="${MMVP_EVAL_SCRIPT:-llava/eval/eval_mmvp.py}"

# V*Bench
VSTAR_ROOT="${VSTAR_ROOT:-./playground/data/eval/vstar_bench}"
VSTAR_OUT_ROOT="${VSTAR_OUT_ROOT:-${VSTAR_ROOT}/answers}"
VSTAR_QUESTION_FILE="${VSTAR_QUESTION_FILE:-${VSTAR_ROOT}/test_questions.jsonl}"
VSTAR_EVAL_SCRIPT="${VSTAR_EVAL_SCRIPT:-llava/eval/eval_vstar.py}"

# DTD
DTD_ROOT="${DTD_ROOT:-./playground/data/eval/dtd/dtd}"
DTD_RUN_NAME="${DTD_RUN_NAME:-test_1-10}"
DTD_OUT_ROOT="${DTD_OUT_ROOT:-${DTD_ROOT}/answers/${DTD_RUN_NAME}}"
DTD_QUESTION_FILE="${DTD_QUESTION_FILE:-${DTD_OUT_ROOT}/questions.jsonl}"
DTD_EVAL_SCRIPT="${DTD_EVAL_SCRIPT:-llava/eval/eval_dtd.py}"

ALL_BENCHMARKS=(mmbench gqa seed pope textvqa mme vizwiz vqav2 hallusionbench realworldqa chartqa hrbench4k hrbench8k mmstar mmvet mmvp vstar dtd)

success_count=0
skip_count=0
fail_count=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/v1_5/eval/summarize_benchmark_outputs.sh [all|benchmark ...]

Examples:
  bash scripts/v1_5/eval/summarize_benchmark_outputs.sh all
  bash scripts/v1_5/eval/summarize_benchmark_outputs.sh gqa
  bash scripts/v1_5/eval/summarize_benchmark_outputs.sh mmbench gqa seed hallusionbench realworldqa chartqa hrbench4k hrbench8k mmstar mmvet mmvp vstar dtd

Benchmarks:
  mmbench gqa seed pope textvqa mme vizwiz vqav2 hallusionbench realworldqa chartqa hrbench4k hrbench8k mmstar mmvet mmvp vstar dtd

Notes:
  - This script does NOT run inference. It only reads existing result jsonl files.
  - It writes summary/details next to each checkpoint's result file.
  - For VizWiz/VQAv2, provide annotation file via env var to compute accuracy:
      VIZWIZ_ANNOTATION_FILE=/path/to/ann.json
      VQAV2_ANNOTATION_FILE=/path/to/ann.json
  - HallusionBench expects ${HALLUSIONBENCH_QUESTION_FILE} to exist.
  - RealWorldQA expects ${REALWORLDQA_QUESTION_FILE} to exist.
  - ChartQA expects ${CHARTQA_QUESTION_FILE} to exist.
  - HRBench4K expects ${HRBENCH4K_QUESTION_FILE} to exist.
  - HRBench8K expects ${HRBENCH8K_QUESTION_FILE} to exist.
  - MMStar expects ${MMSTAR_QUESTION_FILE} to exist.
  - MMVet expects ${MMVET_ANNOTATION_FILE} to exist or judge-free summaries will remain incomplete.
  - MMVP expects ${MMVP_QUESTION_FILE} to exist.
  - V*Bench expects ${VSTAR_QUESTION_FILE} to exist.
  - DTD expects ${DTD_QUESTION_FILE} to exist. Override DTD_RUN_NAME/DTD_OUT_ROOT for non-default runs.
EOF
}

log() {
    printf '[summarize] %s\n' "$*"
}

run_python_script() {
    local label="$1"
    local script_path="$2"
    shift
    shift
    log "${label}"
    if PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" "$script_path" "$@"; then
        success_count=$((success_count + 1))
    else
        fail_count=$((fail_count + 1))
        log "FAILED: ${label}"
    fi
}

run_summary_cmd() {
    local label="$1"
    shift
    run_python_script "${label}" "${EVAL_SCRIPT}" "$@"
}

ensure_required_file() {
    local file_path="$1"
    local context="$2"
    if [[ ! -f "${file_path}" ]]; then
        log "SKIP ${context}: missing required file ${file_path}"
        skip_count=$((skip_count + 1))
        return 1
    fi
    return 0
}

summarize_gqa() {
    local base_dir="${GQA_OUT_ROOT}/answers/${GQA_SPLIT}"
    if [[ ! -d "${base_dir}" ]]; then
        log "SKIP gqa: ${base_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${GQA_QUESTION_FILE}" "gqa" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_summary_cmd \
            "gqa $(basename "${ckpt_dir}")" \
            --dataset gqa \
            --question-file "${GQA_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --balanced-only \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${base_dir}" -type f -name 'merge.jsonl' -print0 | sort -z)
}

summarize_mmbench() {
    local answer_dir="${MMBENCH_OUT_ROOT}/answers/${MMBENCH_SPLIT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP mmbench: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${MMBENCH_QUESTION_FILE}" "mmbench" || return

    while IFS= read -r -d '' result_file; do
        local stem
        stem="$(basename "${result_file}" .jsonl)"
        run_summary_cmd \
            "mmbench ${stem}" \
            --dataset mmbench \
            --question-file "${MMBENCH_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${answer_dir}/${stem}_summary.json" \
            --detail-file "${answer_dir}/${stem}_details.jsonl"
    done < <(find "${answer_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name '*_details.jsonl' -print0 | sort -z)
}

summarize_seed() {
    local answer_dir="${SEED_OUT_ROOT}/answers"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP seed: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"

        local args=(
            --dataset seed
            --result-file "${result_file}"
            --output-file "${ckpt_dir}/summary.json"
            --detail-file "${ckpt_dir}/details.jsonl"
        )
        if [[ -n "${SEED_ANNOTATION_FILE}" ]]; then
            args+=(--annotation-file "${SEED_ANNOTATION_FILE}")
        fi

        run_summary_cmd "seed $(basename "${ckpt_dir}")" "${args[@]}"
    done < <(find "${answer_dir}" -type f -name 'merge.jsonl' -print0 | sort -z)
}

summarize_pope() {
    local answer_dir="${POPE_OUT_ROOT}/answers"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP pope: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${POPE_QUESTION_FILE}" "pope" || return
    if [[ ! -d "${POPE_ANNOTATION_DIR}" ]]; then
        log "SKIP pope: missing annotation dir ${POPE_ANNOTATION_DIR}"
        skip_count=$((skip_count + 1))
        return
    fi

    while IFS= read -r -d '' result_file; do
        local stem
        stem="$(basename "${result_file}" .jsonl)"
        run_summary_cmd \
            "pope ${stem}" \
            --dataset pope \
            --question-file "${POPE_QUESTION_FILE}" \
            --annotation-dir "${POPE_ANNOTATION_DIR}" \
            --result-file "${result_file}" \
            --output-file "${answer_dir}/${stem}_summary.json" \
            --detail-file "${answer_dir}/${stem}_details.jsonl"
    done < <(find "${answer_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name '*_details.jsonl' -print0 | sort -z)
}

summarize_textvqa() {
    local answer_dir="${TEXTVQA_OUT_ROOT}/answers"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP textvqa: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${TEXTVQA_ANNOTATION_FILE}" "textvqa" || return

    while IFS= read -r -d '' result_file; do
        local stem
        stem="$(basename "${result_file}" .jsonl)"
        run_summary_cmd \
            "textvqa ${stem}" \
            --dataset textvqa \
            --annotation-file "${TEXTVQA_ANNOTATION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${answer_dir}/${stem}_summary.json" \
            --detail-file "${answer_dir}/${stem}_details.jsonl"
    done < <(find "${answer_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name '*_details.jsonl' -print0 | sort -z)
}

summarize_mme() {
    local answer_dir="${MME_OUT_ROOT}/answers"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP mme: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi

    while IFS= read -r -d '' result_file; do
        local stem
        stem="$(basename "${result_file}" .jsonl)"

        local args=(
            --dataset mme
            --result-file "${result_file}"
            --output-file "${answer_dir}/${stem}_summary.json"
            --detail-file "${answer_dir}/${stem}_details.jsonl"
        )
        if [[ -n "${MME_QUESTION_FILE}" ]]; then
            args+=(--question-file "${MME_QUESTION_FILE}")
        fi

        run_summary_cmd "mme ${stem}" "${args[@]}"
    done < <(find "${answer_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name '*_details.jsonl' -print0 | sort -z)
}

summarize_vizwiz() {
    local answer_dir="${VIZWIZ_OUT_ROOT}/answers"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP vizwiz: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi

    while IFS= read -r -d '' result_file; do
        local stem
        stem="$(basename "${result_file}" .jsonl)"

        local args=(
            --dataset vizwiz
            --question-file "${VIZWIZ_QUESTION_FILE}"
            --result-file "${result_file}"
            --output-file "${answer_dir}/${stem}_summary.json"
            --detail-file "${answer_dir}/${stem}_details.jsonl"
        )
        if [[ -n "${VIZWIZ_ANNOTATION_FILE}" ]]; then
            args+=(--annotation-file "${VIZWIZ_ANNOTATION_FILE}")
        fi

        run_summary_cmd "vizwiz ${stem}" "${args[@]}"
    done < <(find "${answer_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name '*_details.jsonl' -print0 | sort -z)
}

summarize_vqav2() {
    local base_dir="${VQAV2_OUT_ROOT}/answers/${VQAV2_SPLIT}"
    if [[ ! -d "${base_dir}" ]]; then
        log "SKIP vqav2: ${base_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"

        local args=(
            --dataset vqav2
            --result-file "${result_file}"
            --output-file "${ckpt_dir}/summary.json"
            --detail-file "${ckpt_dir}/details.jsonl"
        )
        if [[ -n "${VQAV2_ANNOTATION_FILE}" ]]; then
            args+=(--annotation-file "${VQAV2_ANNOTATION_FILE}")
        fi

        run_summary_cmd "vqav2 $(basename "${ckpt_dir}")" "${args[@]}"
    done < <(find "${base_dir}" -type f -name 'merge.jsonl' -print0 | sort -z)
}

summarize_hallusionbench() {
    local answer_dir="${HALLUSIONBENCH_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP hallusionbench: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${HALLUSIONBENCH_QUESTION_FILE}" "hallusionbench" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        local export_result_file
        ckpt_dir="$(dirname "${result_file}")"
        export_result_file="${ckpt_dir}/HallusionBench_result.json"
        run_python_script \
            "hallusionbench $(basename "${ckpt_dir}")" \
            "${HALLUSIONBENCH_EVAL_SCRIPT}" \
            --question-file "${HALLUSIONBENCH_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl" \
            --export-result-file "${export_result_file}"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_realworldqa() {
    local answer_dir="${REALWORLDQA_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP realworldqa: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${REALWORLDQA_QUESTION_FILE}" "realworldqa" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "realworldqa $(basename "${ckpt_dir}")" \
            "${REALWORLDQA_EVAL_SCRIPT}" \
            --question-file "${REALWORLDQA_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_chartqa() {
    local answer_dir="${CHARTQA_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP chartqa: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${CHARTQA_QUESTION_FILE}" "chartqa" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "chartqa $(basename "${ckpt_dir}")" \
            "${CHARTQA_EVAL_SCRIPT}" \
            --question-file "${CHARTQA_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_hrbench4k() {
    local answer_dir="${HRBENCH4K_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP hrbench4k: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${HRBENCH4K_QUESTION_FILE}" "hrbench4k" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "hrbench4k $(basename "${ckpt_dir}")" \
            "${HRBENCH4K_EVAL_SCRIPT}" \
            --question-file "${HRBENCH4K_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --dataset-name hrbench4k \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_hrbench8k() {
    local answer_dir="${HRBENCH8K_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP hrbench8k: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${HRBENCH8K_QUESTION_FILE}" "hrbench8k" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "hrbench8k $(basename "${ckpt_dir}")" \
            "${HRBENCH8K_EVAL_SCRIPT}" \
            --question-file "${HRBENCH8K_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --dataset-name hrbench8k \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_mmstar() {
    local answer_dir="${MMSTAR_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP mmstar: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${MMSTAR_QUESTION_FILE}" "mmstar" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "mmstar $(basename "${ckpt_dir}")" \
            "${MMSTAR_EVAL_SCRIPT}" \
            --question-file "${MMSTAR_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_mmvet() {
    local answer_dir="${MMVET_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP mmvet: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${MMVET_ANNOTATION_FILE}" "mmvet" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        local -a args
        ckpt_dir="$(dirname "${result_file}")"
        args=(
            --annotation-file "${MMVET_ANNOTATION_FILE}"
            --result-file "${result_file}"
            --grade-file "${ckpt_dir}/grades.json"
            --output-file "${ckpt_dir}/summary.json"
            --detail-file "${ckpt_dir}/details.jsonl"
            --num-runs "${MMVET_JUDGE_NUM_RUNS}"
        )
        if [[ -n "${MMVET_JUDGE_MODEL}" ]]; then
            args+=(--judge-model "${MMVET_JUDGE_MODEL}")
        fi
        if [[ -n "${MMVET_JUDGE_API_KEY}" ]]; then
            args+=(--judge-api-key "${MMVET_JUDGE_API_KEY}")
        fi
        if [[ -n "${MMVET_JUDGE_BASE_URL}" ]]; then
            args+=(--judge-base-url "${MMVET_JUDGE_BASE_URL}")
        fi

        run_python_script \
            "mmvet $(basename "${ckpt_dir}")" \
            "${MMVET_EVAL_SCRIPT}" \
            "${args[@]}"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_mmvp() {
    local answer_dir="${MMVP_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP mmvp: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${MMVP_QUESTION_FILE}" "mmvp" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "mmvp $(basename "${ckpt_dir}")" \
            "${MMVP_EVAL_SCRIPT}" \
            --question-file "${MMVP_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_vstar() {
    local answer_dir="${VSTAR_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP vstar: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${VSTAR_QUESTION_FILE}" "vstar" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "vstar $(basename "${ckpt_dir}")" \
            "${VSTAR_EVAL_SCRIPT}" \
            --question-file "${VSTAR_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

summarize_dtd() {
    local answer_dir="${DTD_OUT_ROOT}"
    if [[ ! -d "${answer_dir}" ]]; then
        log "SKIP dtd: ${answer_dir} not found"
        skip_count=$((skip_count + 1))
        return
    fi
    ensure_required_file "${DTD_QUESTION_FILE}" "dtd" || return

    while IFS= read -r -d '' result_file; do
        local ckpt_dir
        ckpt_dir="$(dirname "${result_file}")"
        run_python_script \
            "dtd $(basename "${ckpt_dir}")" \
            "${DTD_EVAL_SCRIPT}" \
            --question-file "${DTD_QUESTION_FILE}" \
            --result-file "${result_file}" \
            --output-file "${ckpt_dir}/summary.json" \
            --detail-file "${ckpt_dir}/details.jsonl"
    done < <(find "${answer_dir}" -type f -name 'predictions.jsonl' -print0 | sort -z)
}

run_one_benchmark() {
    local benchmark="$1"
    case "${benchmark}" in
        mmbench) summarize_mmbench ;;
        gqa) summarize_gqa ;;
        seed) summarize_seed ;;
        pope) summarize_pope ;;
        textvqa) summarize_textvqa ;;
        mme) summarize_mme ;;
        vizwiz) summarize_vizwiz ;;
        vqav2) summarize_vqav2 ;;
        hallusionbench) summarize_hallusionbench ;;
        realworldqa) summarize_realworldqa ;;
        chartqa) summarize_chartqa ;;
        hrbench4k) summarize_hrbench4k ;;
        hrbench8k) summarize_hrbench8k ;;
        mmstar) summarize_mmstar ;;
        mmvet) summarize_mmvet ;;
        mmvp) summarize_mmvp ;;
        vstar) summarize_vstar ;;
        dtd) summarize_dtd ;;
        *)
            log "Unknown benchmark: ${benchmark}"
            usage
            exit 2
            ;;
    esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

targets=()
if [[ $# -eq 0 ]]; then
    targets=("${ALL_BENCHMARKS[@]}")
elif [[ $# -eq 1 && "$1" == "all" ]]; then
    targets=("${ALL_BENCHMARKS[@]}")
else
    targets=("$@")
fi

for benchmark in "${targets[@]}"; do
    run_one_benchmark "${benchmark}"
done

log "Done. success=${success_count}, skipped=${skip_count}, failed=${fail_count}"
if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
