#!/bin/bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="${SCRIPT_DIR}/../../.."

# Avoid home directory quota issues by storing cache/tmp under project storage.
CACHE_ROOT="${PROJECT_ROOT}/.apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/cache"
export APPTAINER_TMPDIR="${CACHE_ROOT}/tmp"

# Backward compatibility for environments still honoring Singularity vars.
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

cd "${PROJECT_ROOT}"

apptainer build \
    --fakeroot \
    "${SCRIPT_DIR}/vllm.sif" \
    "${VLLM_DIR}/vllm.def"
