#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT_ROOT="/shared/nas2/jk100/MoL/checkpoints"
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
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-patchlayer-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k2_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_learnable-gate_lr1e-5-pretrain_lr2e-5-finetune_k4_no-delta"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune"
    "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune_k2"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_disagreement-gate_lr1e-5-pretrain_lr2e-5-finetune_k4"
    # "${CHECKPOINT_ROOT}/llava-v1.5-13b-moe-finetune_text-cond-attn-hybrid-router_lr1e-5-pretrain_lr2e-5-finetune_no_reserve"
)

declare -a FAILURES=()

timestamp() {
    date "+%F %T"
}

ckpt_name() {
    basename "${1%/}"
}

run() {
    local label="$1"
    shift

    echo
    echo "============================================================"
    echo "[$(timestamp)] START ${label}"
    echo "[$(timestamp)] CMD   $*"
    echo "============================================================"

    if "$@"; then
        echo "[$(timestamp)] OK    ${label}"
    else
        local status=$?
        echo "[$(timestamp)] FAIL  ${label} (exit ${status})"
        FAILURES+=("${label} (exit ${status})")
    fi
}

run_script_with_model_paths_override() {
    local script_path="$1"
    local model_paths_override=""
    if ((${#MODEL_PATHS[@]} > 0)); then
        printf -v model_paths_override '%s\n' "${MODEL_PATHS[@]}"
    fi
    MODEL_PATHS_OVERRIDE="${model_paths_override}" bash "${script_path}"
}

run_existing_script() {
    local script_name="$1"
    run "${script_name}" run_script_with_model_paths_override "${SCRIPT_DIR}/${script_name}"
}

run_llavabench_for_model() {
    local model_path="$1"
    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local answers_file="./playground/data/eval/llava-bench-in-the-wild/answers/${model_id}.jsonl"
    local review_file="./playground/data/eval/llava-bench-in-the-wild/reviews/${model_id}.jsonl"

    mkdir -p "./playground/data/eval/llava-bench-in-the-wild/answers"
    mkdir -p "./playground/data/eval/llava-bench-in-the-wild/reviews"

    python -m llava.eval.model_vqa \
        --model-path "${model_path}" \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file "${answers_file}" \
        --temperature 0 \
        --conv-mode vicuna_v1

    python llava/eval/eval_gpt_review_bench.py \
        --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
        --rule llava/eval/table/rule.json \
        --answer-list \
            playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
            "${answers_file}" \
        --output "${review_file}"

    python llava/eval/summarize_gpt_review.py -f "${review_file}"
}

run_mmbench_cn_for_model() {
    local model_path="$1"
    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local split="mmbench_dev_cn_20231003"
    local answer_dir="./playground/data/eval/mmbench_cn/answers/${split}"
    local upload_dir="./playground/data/eval/mmbench_cn/answers_upload/${split}"

    mkdir -p "${answer_dir}" "${upload_dir}"

    python -m llava.eval.model_vqa_mmbench \
        --model-path "${model_path}" \
        --question-file "./playground/data/eval/mmbench_cn/${split}.tsv" \
        --answers-file "${answer_dir}/${model_id}.jsonl" \
        --lang cn \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file "./playground/data/eval/mmbench_cn/${split}.tsv" \
        --result-dir "${answer_dir}" \
        --upload-dir "${upload_dir}" \
        --experiment "${model_id}"
}

run_mmvet_for_model() {
    local model_path="$1"
    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local answers_file="./playground/data/eval/mm-vet/answers/${model_id}.jsonl"
    local result_file="./playground/data/eval/mm-vet/results/${model_id}.json"

    mkdir -p "./playground/data/eval/mm-vet/answers"
    mkdir -p "./playground/data/eval/mm-vet/results"

    python -m llava.eval.model_vqa \
        --model-path "${model_path}" \
        --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder ./playground/data/eval/mm-vet/images \
        --answers-file "${answers_file}" \
        --temperature 0 \
        --conv-mode vicuna_v1

    python scripts/convert_mmvet_for_eval.py \
        --src "${answers_file}" \
        --dst "${result_file}"
}

run_qbench_for_model() {
    local model_path="$1"
    local split="$2"
    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local output_dir="./playground/data/eval/qbench/answers/${model_id}"

    mkdir -p "${output_dir}"

    python -m llava.eval.model_vqa_qbench \
        --model-path "${model_path}" \
        --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
        --questions-file "./playground/data/eval/qbench/llvisionqa_${split}.json" \
        --answers-file "${output_dir}/llvisionqa_${split}_answers.jsonl" \
        --conv-mode llava_v1 \
        --lang en
}

run_qbench_zh_for_model() {
    local model_path="$1"
    local split="$2"
    local zh_split
    case "${split}" in
        dev) zh_split="验证集" ;;
        test) zh_split="测试集" ;;
        *) echo "Unknown qbench_zh split: ${split}" >&2; return 1 ;;
    esac

    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local output_dir="./playground/data/eval/qbench/answers/${model_id}"

    mkdir -p "${output_dir}"

    python -m llava.eval.model_vqa_qbench \
        --model-path "${model_path}" \
        --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
        --questions-file "./playground/data/eval/qbench/质衡-问答-${zh_split}.json" \
        --answers-file "${output_dir}/llvisionqa_zh_${split}_answers.jsonl" \
        --conv-mode llava_v1 \
        --lang zh
}

run_sqa_for_model() {
    local model_path="$1"
    local model_id
    model_id="$(ckpt_name "${model_path}")"
    local answer_dir="./playground/data/eval/scienceqa/answers"
    local answers_file="${answer_dir}/${model_id}.jsonl"

    mkdir -p "${answer_dir}"

    python -m llava.eval.model_vqa_science \
        --model-path "${model_path}" \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file "${answers_file}" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1

    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file "${answers_file}" \
        --output-file "${answer_dir}/${model_id}_output.jsonl" \
        --output-result "${answer_dir}/${model_id}_result.json"
}

run_model_loop() {
    local benchmark_name="$1"
    shift
    local fn_name="$1"
    shift

    local model_path
    for model_path in "${MODEL_PATHS[@]}"; do
        local model_id
        model_id="$(ckpt_name "${model_path}")"
        run "${benchmark_name}:${model_id}" "${fn_name}" "${model_path}" "$@"
    done
}

# These scripts already contain the same batch-model loop internally.
run_existing_script "naturalbench.sh"
run_existing_script "docvqa.sh"
run_existing_script "cvbench.sh"
run_existing_script "mmstar.sh"
run_existing_script "mmvp.sh"
run_existing_script "mmvet.sh"
run_existing_script "vstar.sh"
run_existing_script "hallusionbench.sh"
run_existing_script "docvqa.sh"
run_existing_script "charxiv.sh"
run_existing_script "visualoverload.sh"
run_existing_script "realworldqa.sh"
run_existing_script "hrbench4k.sh"
run_existing_script "hrbench8k.sh"
run_existing_script "naturalbench.sh"
run_existing_script "mme_realworld.sh"
run_existing_script "mmmu.sh"
run_existing_script "mmbench.sh"
run_existing_script "pope.sh"
run_existing_script "chartqa.sh"
run_existing_script "countbenchqa.sh"
run_existing_script "docvqa.sh"
run_existing_script "gqa.sh"
run_existing_script "textvqa.sh"
# run_existing_script "llavabench.sh"
# run_existing_script "mme.sh"
# run_existing_script "seed.sh"
# run_existing_script "vizwiz.sh"
# run_existing_script "vqav2.sh"
# run_existing_script "dtd.sh"

# These launchers are single-model scripts, so apply the shared MODEL_PATHS here.
# run_model_loop "mmbench_cn" run_mmbench_cn_for_model
# run_model_loop "qbench-dev" run_qbench_for_model dev
# run_model_loop "qbench-test" run_qbench_for_model test
# run_model_loop "qbench_zh-dev" run_qbench_zh_for_model dev
# run_model_loop "qbench_zh-test" run_qbench_zh_for_model test
# run_model_loop "sqa" run_sqa_for_model

echo
echo "============================================================"
if ((${#FAILURES[@]} > 0)); then
    echo "Completed with failures:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi

echo "All benchmark runs completed successfully."
echo "============================================================"
