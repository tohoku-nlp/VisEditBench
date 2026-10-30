#!/bin/bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODE=""
MODEL=""
GPU_ID="0,1,2,3"
HF_TOKEN=""
RUNTIME="singularity"

INPUT_PATH=""
OUTPUT_PATH=""

HOSTS="localhost"
PORT=8080
PORTS=""

TENSOR_PARALLEL_SIZE=4
DTYPE="bfloat16"
SEED=42
GUIDED_DECODING="json"

NUM_PROCESSES_PER_SERVER=1
MAX_TOKENS=128000
TEX_OUTPUT_DIR=""
ENABLE_THINKING=0

START_SERVER=1
WAIT_RETRIES=60
WAIT_INTERVAL=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)                     MODE="$2";                     shift 2 ;;
        --model)                    MODEL="$2";                    shift 2 ;;
        --gpu_id)                   GPU_ID="$2";                   shift 2 ;;
        --hf_token)                 HF_TOKEN="$2";                 shift 2 ;;
        --runtime)                  RUNTIME="$2";                  shift 2 ;;
        --input_path)               INPUT_PATH="$2";               shift 2 ;;
        --output_path)              OUTPUT_PATH="$2";              shift 2 ;;
        --hosts)                    HOSTS="$2";                    shift 2 ;;
        --port)                     PORT="$2";                     shift 2 ;;
        --ports)                    PORTS="$2";                    shift 2 ;;
        --tensor_parallel_size)     TENSOR_PARALLEL_SIZE="$2";     shift 2 ;;
        --dtype)                    DTYPE="$2";                    shift 2 ;;
        --seed)                     SEED="$2";                     shift 2 ;;
        --guided_decoding)          GUIDED_DECODING="$2";          shift 2 ;;
        --num_processes_per_server) NUM_PROCESSES_PER_SERVER="$2"; shift 2 ;;
        --max_tokens)               MAX_TOKENS="$2";               shift 2 ;;
        --tex_output_dir)           TEX_OUTPUT_DIR="$2";           shift 2 ;;
        --skip_server)              START_SERVER=0;                 shift 1 ;;
        --wait_retries)             WAIT_RETRIES="$2";             shift 2 ;;
        --wait_interval)            WAIT_INTERVAL="$2";            shift 2 ;;
        --enable_thinking)          ENABLE_THINKING=1;              shift 1 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

missing=()
[[ -z "${MODE}" ]]        && missing+=("--mode")
[[ -z "${MODEL}" ]]       && missing+=("--model")
[[ -z "${INPUT_PATH}" ]]  && missing+=("--input_path")
[[ -z "${OUTPUT_PATH}" ]] && missing+=("--output_path")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: missing required arguments: ${missing[*]}" >&2
    exit 1
fi

if [[ "${MODE}" != "visual" && "${MODE}" != "visual_w_org_tikz" && "${MODE}" != "text_w_org_tikz" ]]; then
    echo "Error: --mode must be one of: visual, visual_w_org_tikz, text_w_org_tikz" >&2
    exit 1
fi

if [[ "${RUNTIME}" != "singularity" && "${RUNTIME}" != "apptainer" ]]; then
    echo "Error: --runtime must be 'singularity' or 'apptainer', got '${RUNTIME}'" >&2
    exit 1
fi

if [[ -z "${PORTS}" ]]; then
    PORTS="${PORT}"
fi

wait_for_server() {
    local host="$1"
    local port="$2"

    if command -v curl >/dev/null 2>&1; then
        for ((i=1; i<=WAIT_RETRIES; i++)); do
            if curl -sf "http://${host}:${port}/v1/models" >/dev/null; then
                return 0
            fi
            sleep "${WAIT_INTERVAL}"
        done
        return 1
    fi

    sleep 10
    return 0
}

if [[ "${START_SERVER}" -eq 1 ]]; then
    echo "Starting vLLM server (Qwen3.5)..."
    SERVER_SCRIPT="${SCRIPT_DIR}/run_vllm_server.sh"
    chmod +x "${SERVER_SCRIPT}"
    "${SERVER_SCRIPT}" \
        --model "${MODEL}" \
        --gpu_id "${GPU_ID}" \
        --hf_token "${HF_TOKEN}" \
        --runtime "${RUNTIME}" \
        --port "${PORT}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --dtype "${DTYPE}" \
        --seed "${SEED}" \
        --guided_decoding "${GUIDED_DECODING}" \
        &
    SERVER_PID=$!
    trap 'echo "Stopping vLLM server..."; kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

    read -r -a host_arr <<< "${HOSTS}"
    read -r -a port_arr <<< "${PORTS}"
    wait_host="${host_arr[0]}"
    wait_port="${port_arr[0]}"

    echo "Waiting for server to be ready at ${wait_host}:${wait_port}..."
    if ! wait_for_server "${wait_host}" "${wait_port}"; then
        echo "Error: server did not become ready in time." >&2
        exit 1
    fi
fi

echo "Starting inference (mode=${MODE})..."
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_ROOT}"
uv sync

python_args=(
    --mode "${MODE}"
    --input_path "${INPUT_PATH}"
    --output_path "${OUTPUT_PATH}"
    --hosts ${HOSTS}
    --ports ${PORTS}
    --model "${MODEL}"
    --token "${HF_TOKEN}"
    --seed "${SEED}"
    --num_processes_per_server "${NUM_PROCESSES_PER_SERVER}"
    --max_tokens "${MAX_TOKENS}"
)

if [[ -n "${TEX_OUTPUT_DIR}" ]]; then
    python_args+=(--tex_output_dir "${TEX_OUTPUT_DIR}")
fi

[[ "${ENABLE_THINKING}" -eq 1 ]] && python_args+=(--enable_thinking)

uv run python "${REPO_ROOT}/editing/vllm/run_vllm_edit_inference.py" \
    "${python_args[@]}"

echo "Done."
