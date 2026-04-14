#!/usr/bin/env bash

load_model_paths_override() {
    local serialized="${MODEL_PATHS_OVERRIDE:-}"
    if [[ -z "${serialized}" ]]; then
        return
    fi

    mapfile -t MODEL_PATHS < <(printf '%s\n' "${serialized}")

    local filtered=()
    local model_path
    for model_path in "${MODEL_PATHS[@]}"; do
        if [[ -n "${model_path}" ]]; then
            filtered+=("${model_path}")
        fi
    done
    MODEL_PATHS=("${filtered[@]}")
}
