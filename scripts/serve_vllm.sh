#!/usr/bin/env bash
set -euo pipefail

WORLD_MEM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-${WORLD_MEM_ROOT}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORLD_MEM_ROOT}/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORLD_MEM_ROOT}/.cache/vllm}"

# Run this in a GPU environment. Clients may run on a different CPU machine.
exec vllm serve "${MODEL_ID:-Qwen/Qwen3-8B}" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3-8B}" \
  --host "${BIND_ADDRESS:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  "$@"
