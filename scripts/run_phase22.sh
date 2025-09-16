#!/usr/bin/env bash
set -euo pipefail

if [ -x /home/daniel/miniforge/bin/conda ]; then
  eval "$('/home/daniel/miniforge/bin/conda' 'shell.bash' 'hook')"
else
  echo "Conda not found at /home/daniel/miniforge/bin/conda" >&2
  exit 1
fi

conda activate k3d-cranium
export PYTHONPATH=.

echo "[Phase22] Generating 1000 clusters (GPU-only torch KMeans)..."
python -m knowledge3d.tools.phase18.meaning_cluster_trainer --generate_clusters 1000

echo "[Phase22] Training all clusters (GPU-only AdaptedFusedHead)..."
python -m knowledge3d.tools.phase18.meaning_cluster_trainer --train_all_clusters

echo "[Phase22] Done. See logs/phase22_clusters.json and logs/phase22_scale_report.json"

