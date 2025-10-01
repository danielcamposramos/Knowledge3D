#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Activate conda
if [ -f "/home/daniel/miniforge/etc/profile.d/conda.sh" ]; then
  . "/home/daniel/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# 1) Vision captions (small warmup, then 200 if non-empty)
"$ROOT_DIR/scripts/run_vision_captions.sh"

# 2) Training-only sequential chain (consistency 50+50, shapes 100, long-run 2x50)
"$ROOT_DIR/scripts/run_gpu_train_only.sh"

echo "[VISION+TRAIN] done"

