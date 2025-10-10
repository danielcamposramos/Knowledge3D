#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/gpu_train_only_${STAMP}.log"

echo "[GPU-TRAIN] starting @ $STAMP" | tee -a "$LOG"
if [ -f "/home/daniel/miniforge/etc/profile.d/conda.sh" ]; then
  . "/home/daniel/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export K3D_PTX_STRICT=1
export K3D_FORCE_PTX_FUSE=1
export K3D_RPN_BEAM=1
export K3D_RPN_BEAM_WIDTH=5

step () { echo; echo "[GPU-TRAIN] $1" | tee -a "$LOG"; }

step "RLWHF Policy Training (replaces phase25 trainers) — 100 epochs"
# DEPRECATED: phase25 consistency/shapes/long_run trainers consolidated into train_rlwhf_policy
python -m knowledge3d.tools.training_pipelines.train_rlwhf_policy \
  --epochs 100 --limit 5000 --lr 5e-4 | tee -a "$LOG"

# Old (deprecated) commands - phase25 modules were removed:
# python -m knowledge3d.tools.phase25.consistency_trainer --epochs 50 --limit 5000 --lr 1e-3
# python -m knowledge3d.tools.phase25.shapes_trainer --epochs 100 --limit 5000
# python -m knowledge3d.tools.phase25.long_run --epochs 50 --limit 300 --eval-every 5

echo "[GPU-TRAIN] complete" | tee -a "$LOG"

