#!/bin/bash

set -eu

MODEL=$1
GPU_ID=$2
HF_TOKEN=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

SIF_FILE_PATH="${SCRIPT_DIR}/vllm.sif"
CONTAINER_WORKDIR=/work

TENSOR_PARALLEL_SIZE=4
DTYPE="bfloat16"
SEED=42
PORT=8080
GUIDED_DECODING="json"

apptainer run \
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
        "${PORT}" \
        "${GUIDED_DECODING}"
