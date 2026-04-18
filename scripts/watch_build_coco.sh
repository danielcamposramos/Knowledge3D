#!/usr/bin/env bash
# Watch for the COCO CSV and build the GLB once ready.
#
# Usage:
#   RAW=/home/daniel/K3D_llama_cpp/datasets BASE=/K3D/Knowledge3D.local/datasets \
#   scripts/watch_build_coco.sh
#
# Notes:
# - Respects K3D_CONDA_ENV (k3dml by default)
# - Writes logs to /home/daniel/K3D_llama_cpp/logs/coco_build.log
set -euo pipefail

RAW=${RAW:-/home/daniel/K3D_llama_cpp/datasets}
BASE=${BASE:-/K3D/Knowledge3D.local/datasets}
LOGS=${LOGS:-/home/daniel/K3D_llama_cpp/logs}
mkdir -p "$LOGS" "viewer/public"

CSV="$BASE/coco.train.clip.csv"
META="$BASE/coco.train.meta.json"
GLB="viewer/public/coco_50k.glb"

echo "[WATCH] Waiting for $CSV and $META ..." | tee -a "$LOGS/coco_build.log"
while [[ ! -s "$CSV" || ! -f "$META" ]]; do
  sleep 3
done
echo "[BUILD] Found inputs. Building $GLB" | tee -a "$LOGS/coco_build.log"
scripts/k3d_env.sh run python -m k3dgen "$CSV" --gltf "$GLB" --k 10 --reducer umap --metadata "$META" --emb-precision f16 2>&1 | tee -a "$LOGS/coco_build.log"
echo "[DONE] Built $GLB" | tee -a "$LOGS/coco_build.log"

