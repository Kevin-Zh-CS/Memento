#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
OPENCLAW_PORT="${OPENCLAW_PORT:-18789}"
STATE_DIR="${OPENCLAW_STATE_DIR:-$ROOT_DIR/.openclaw-state}"
WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-$ROOT_DIR/.openclaw-workspace}"
CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$STATE_DIR/openclaw.json}"
LOG_DIR="$STATE_DIR/logs"

mkdir -p "$STATE_DIR" "$WORKSPACE_DIR" "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  sed "s#__MEMENTO_WORKSPACE__#$WORKSPACE_DIR#g" \
    "$ROOT_DIR/config/openclaw.example.json" > "$CONFIG_PATH"
  chmod 600 "$CONFIG_PATH"
fi

chmod 700 "$STATE_DIR"

exec env \
  OPENCLAW_CONFIG_PATH="$CONFIG_PATH" \
  OPENCLAW_STATE_DIR="$STATE_DIR" \
  "$OPENCLAW_BIN" gateway --port "$OPENCLAW_PORT" --verbose
