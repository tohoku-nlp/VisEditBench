#!/bin/bash

set -eu

# ==============================================================================
# Default values
# ==============================================================================

MODEL=""
GPU_ID="0,1,2,3"
HF_TOKEN=""
RUNTIME="apptainer"

PORT=8080
TENSOR_PARALLEL_SIZE=4
DTYPE="bfloat16"
SEED=42
GUIDED_DECODING="json"

# ==============================================================================
# Parse keyword arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)                MODEL="$2";                shift 2 ;;
        --gpu_id)               GPU_ID="$2";               shift 2 ;;
        --hf_token)             HF_TOKEN="$2";             shift 2 ;;
        --runtime)              RUNTIME="$2";              shift 2 ;;
        --port)                 PORT="$2";                 shift 2 ;;
        --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --dtype)                DTYPE="$2";                shift 2 ;;
        --seed)                 SEED="$2";                 shift 2 ;;
        --guided_decoding)      GUIDED_DECODING="$2";      shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ==============================================================================
# Validate required arguments
# ==============================================================================

if [[ -z "${MODEL}" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

if [[ "${RUNTIME}" != "singularity" && "${RUNTIME}" != "apptainer" ]]; then
    echo "Error: --runtime must be 'singularity' or 'apptainer', got '${RUNTIME}'" >&2
    exit 1
fi

# ==============================================================================
# Start vLLM server
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
CONTAINER_WORKDIR=/workspace
SIF_FILE_PATH="${PROJECT_ROOT}/apptainer/vllm.sif"

echo "Starting vLLM server (Qwen3.5)..."
echo "  Model:                ${MODEL}"
echo "  GPU:                  ${GPU_ID}"
echo "  Port:                 ${PORT}"
echo "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  Runtime:              ${RUNTIME}"

exec "${RUNTIME}" run \
    --nv \
    --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    --env "NCCL_CUMEM_ENABLE=1" \
    --env "VLLM_USE_V1=1" \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --home "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
    "${SIF_FILE_PATH}" \
        "${MODEL}" \
        "${TENSOR_PARALLEL_SIZE}" \
        "${DTYPE}" \
        "${SEED}" \
        "${PORT}"
