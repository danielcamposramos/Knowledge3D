#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PATH="$HOME/miniconda3/bin:$PATH"

OUT_DIR="$ROOT_DIR//K3D/Knowledge3D.local/models/rlwhf_policy"
DATASET="$ROOT_DIR/docs/reports/training/rlwhf_dataset.jsonl"

echo "[RLWHF] Ensuring dataset at $DATASET..."
"$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.tools.train_rlwhf_policy \
  --dataset "$DATASET" \
  --out "$OUT_DIR" \
  --model distilgpt2 \
  --epochs 1 --batch 4 --max_len 384 --lr 5e-5

echo "[RLWHF] Trained policy saved to $OUT_DIR"

