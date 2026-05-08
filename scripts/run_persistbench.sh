#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPENCLAW_PORT="${OPENCLAW_PORT:-18790}"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/persistbench_openclaw_gpt4omini_results.json}"

export OPENCLAW_BIN

cd "${ROOT_DIR}"

ARGS=(
  --gateway-port "${OPENCLAW_PORT}" \
  --output "${OUTPUT_PATH}" \
  --failure-types "${FAILURE_TYPES:-cross_domain,sycophancy}" \
  --memory-mode "${MEMORY_MODE:-preindex}" \
  --memory-max-results "${MEMORY_MAX_RESULTS:-20}" \
  --memory-min-score "${MEMORY_MIN_SCORE:-0}" \
  --index-timeout "${INDEX_TIMEOUT:-1800}" \
  --sycophancy-failure-threshold "${SYCOPHANCY_FAILURE_THRESHOLD:-4}"
)

if [[ -n "${LIMIT_PER_TYPE+x}" ]]; then
  if [[ -n "${LIMIT_PER_TYPE}" ]]; then
    ARGS+=(--limit-per-type "${LIMIT_PER_TYPE}")
  fi
else
  ARGS+=(--limit-per-type 30)
fi

if [[ -n "${GENERATIONS+x}" ]]; then
  if [[ -n "${GENERATIONS}" ]]; then
    ARGS+=(--generations "${GENERATIONS}")
  fi
else
  ARGS+=(--generations 1)
fi

exec "${PYTHON_BIN}" persistbench_openclaw_eval.py "${ARGS[@]}"
