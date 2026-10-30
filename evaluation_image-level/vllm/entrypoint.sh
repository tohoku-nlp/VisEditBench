#!/bin/sh

MODEL=$1
TENSOR_PARALLEL_SIZE=$2
DTYPE=$3
SEED=$4
PORT=$5

vllm serve "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --port "${PORT}" \
    --trust-remote-code
