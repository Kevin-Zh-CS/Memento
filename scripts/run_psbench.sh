#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
STATE_DIR="${OPENCLAW_STATE_DIR:-$ROOT_DIR/.openclaw-state}"
WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-$ROOT_DIR/.openclaw-workspace}"
CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$STATE_DIR/openclaw.json}"
API_BASE="${OPENCLAW_API_BASE:-http://127.0.0.1:18789/v1}"

exec "$PYTHON_BIN" "$ROOT_DIR/psbench_openclaw_eval.py" \
  --history_path "$ROOT_DIR/data/processed/LoCoMo_ori/{persona}.json" \
  --harmful_dir "$ROOT_DIR/data/processed/Harmful_Query_Set" \
  --api_base "$API_BASE" \
  --openclaw_workspace_dir "$WORKSPACE_DIR" \
  --openclaw_config_path "$CONFIG_PATH" \
  --openclaw_state_dir "$STATE_DIR" \
  --openclaw_cli "$OPENCLAW_BIN" \
  "$@"
