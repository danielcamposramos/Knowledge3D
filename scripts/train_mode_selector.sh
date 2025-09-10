#!/bin/bash
set -euo pipefail

echo "Training Mode Selector..."

OUTCOMES_PATH="docs/reports/training/mode_selector_outcomes.jsonl"
MODEL_PATH="../Knowledge3D.local/models/mode_selector.pkl"

if [[ ! -f "$OUTCOMES_PATH" ]]; then
  echo "No outcome logs found at $OUTCOMES_PATH"
  echo "Run live sessions with K3D_MODE_LOG_SIM=1 first"
  exit 1
fi

scripts/k3d_env.sh run python -m knowledge3d.models.mode_selector \
  "$OUTCOMES_PATH" \
  "$MODEL_PATH"

echo "Mode Selector training complete! Saved to $MODEL_PATH"

