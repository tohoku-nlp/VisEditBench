#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL=""
INPUT_PATH=""
OUTPUT_PATH=""
HOSTS="127.0.0.1"
PORT=8080
PORTS=""
GPU_ID="0"
HF_TOKEN=""
RUNTIME="apptainer"
TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
SEED=42
NUM_PROCESSES_PER_SERVER=1
MAX_TOKENS=128000
ENABLE_THINKING=0
SKIP_SERVER=0
WAIT_RETRIES=60
WAIT_INTERVAL=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --input_path) INPUT_PATH="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ports) PORTS="$2"; shift 2 ;;
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --hf_token) HF_TOKEN="$2"; shift 2 ;;
    --runtime) RUNTIME="$2"; shift 2 ;;
    --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --num_processes_per_server) NUM_PROCESSES_PER_SERVER="$2"; shift 2 ;;
    --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
    --enable_thinking) ENABLE_THINKING=1; shift 1 ;;
    --skip_server) SKIP_SERVER=1; shift 1 ;;
    --wait_retries) WAIT_RETRIES="$2"; shift 2 ;;
    --wait_interval) WAIT_INTERVAL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

missing=()
[[ -z "${MODEL}" ]] && missing+=("--model")
[[ -z "${INPUT_PATH}" ]] && missing+=("--input_path")
[[ -z "${OUTPUT_PATH}" ]] && missing+=("--output_path")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Error: missing required arguments: ${missing[*]}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Error: input JSONL not found: ${INPUT_PATH}" >&2
  exit 1
fi

if [[ -z "${PORTS}" ]]; then
  PORTS="${PORT}"
fi

wait_for_server() {
  local host="$1"
  local port="$2"
  for ((i=1; i<=WAIT_RETRIES; i++)); do
    if curl -sf "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${WAIT_INTERVAL}"
  done
  return 1
}

if [[ "${SKIP_SERVER}" -eq 0 ]]; then
  echo "Starting vLLM server..."
  bash "${SCRIPT_DIR}/run_vllm_server.sh" \
    --model "${MODEL}" \
    --gpu_id "${GPU_ID}" \
    --hf_token "${HF_TOKEN}" \
    --runtime "${RUNTIME}" \
    --port "${PORT}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" &
  SERVER_PID=$!
  trap 'echo "Stopping vLLM server..."; kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

  read -r -a host_arr <<< "${HOSTS}"
  read -r -a port_arr <<< "${PORTS}"
  wait_host="${host_arr[0]}"
  wait_port="${port_arr[0]}"

  echo "Waiting for server at ${wait_host}:${wait_port} ..."
  if ! wait_for_server "${wait_host}" "${wait_port}"; then
    echo "Error: server did not become ready in time." >&2
    exit 1
  fi
fi

echo "Running image-vllm evaluation..."
cd "${REPO_ROOT}"
cmd=(
  uv run python eval.py image-vllm
  --input-path "${INPUT_PATH}"
  --output-path "${OUTPUT_PATH}"
  --hosts ${HOSTS}
  --ports ${PORTS}
  --model "${MODEL}"
  --token "${HF_TOKEN}"
  --seed "${SEED}"
  --num-processes-per-server "${NUM_PROCESSES_PER_SERVER}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ "${ENABLE_THINKING}" -eq 1 ]]; then
  cmd+=(--enable-thinking)
fi

"${cmd[@]}"

echo "Done: ${OUTPUT_PATH}"
