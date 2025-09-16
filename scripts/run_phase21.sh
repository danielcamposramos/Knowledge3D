#!/usr/bin/env bash
set -euo pipefail

# Activate conda k3d-cranium explicitly
if [ -x /home/daniel/miniforge/bin/conda ]; then
  eval "$('/home/daniel/miniforge/bin/conda' 'shell.bash' 'hook')"
else
  echo "Miniforge conda not found at /home/daniel/miniforge/bin/conda" >&2
  exit 1
fi

conda activate k3d-cranium

export PYTHONPATH=.

echo "[Phase21] Generating auto clusters..."
python -m knowledge3d.tools.phase18.meaning_cluster_trainer --gen_phase21

echo "[Phase21] Running training across clusters..."
python -m knowledge3d.tools.phase18.meaning_cluster_trainer --phase21_run

echo "[Phase21] Complete. See logs/phase21_prep_report.json"

