#!/bin/bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODE=""
METADATA_PATH="data/metadata/edit_metadata.jsonl"
SAVE_DIR=""
DEVICE="cuda:0"
MODEL="Qwen/Qwen-Image-Edit-2511"
SEED=0
LIMIT=""
SKIP_EXISTING=0
KEYS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --metadata_path) METADATA_PATH="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --keys) KEYS="$2"; shift 2 ;;
        --skip_existing) SKIP_EXISTING=1; shift 1 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Error: missing required argument --mode" >&2
    echo "       mode must be one of: visual, text_instruction, visual_with_org_tikz" >&2
    exit 1
fi

if [[ "${MODE}" != "visual" && "${MODE}" != "text_instruction" && "${MODE}" != "visual_with_org_tikz" ]]; then
    echo "Error: --mode must be one of: visual, text_instruction, visual_with_org_tikz" >&2
    exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
    if [[ "${MODE}" == "visual" ]]; then
        SAVE_DIR="work/edit/visual/w_org_image/qwen-image-edit-2511"
    elif [[ "${MODE}" == "text_instruction" ]]; then
        SAVE_DIR="work/edit/visual/w_org_image/qwen-image-edit-2511-text-instruction"
    else
        SAVE_DIR="work/edit/visual/w_org_image_w_org_tikz/qwen-image-edit-2511"
    fi
fi

echo "Starting inference (mode=${MODE})..."
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_ROOT}"
uv sync --extra diffusers

args=(
    --mode "${MODE}"
    --metadata_path "${METADATA_PATH}"
    --save_dir "${SAVE_DIR}"
    --device "${DEVICE}"
    --model "${MODEL}"
    --seed "${SEED}"
)

[[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")
[[ "${SKIP_EXISTING}" -eq 1 ]] && args+=(--skip_existing)

if [[ -n "${KEYS}" ]]; then
    read -r -a key_arr <<< "${KEYS}"
    args+=(--keys "${key_arr[@]}")
fi

uv run --extra diffusers python "${REPO_ROOT}/editing/diffusers/edit_images_by_qwen-image-edit-2511.py" "${args[@]}"

echo "Done."
