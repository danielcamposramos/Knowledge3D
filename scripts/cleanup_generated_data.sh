#!/usr/bin/env bash
set -euo pipefail
ROOT=$(pwd)
LOCAL=${K3D_LOCAL_DIR:-/K3D/Knowledge3D.local}
PUB="$ROOT/viewer/public"

# Remove large generated datasets in local folder (safe; outside repo)
if [ -d "$LOCAL/datasets" ]; then
  echo "[local] Removing generated datasets under $LOCAL/datasets (ai_* and knowledge_garden.*)"
  find "$LOCAL/datasets" -maxdepth 1 -type f \( -name 'ai_*' -o -name 'knowledge_garden.*' -o -name 'sample_*' \) -print -delete || true
fi

# Remove big public GLBs to free space; keep tiny demos
if [ -d "$PUB" ]; then
  echo "[public] Removing large generated GLBs from viewer/public"
  cd "$PUB"
  rm -f ai_compendium.*.glb ai_books_basic.*.glb knowledge_garden.*.glb k3d_foundation.*.glb || true
fi

echo "Cleanup complete."
