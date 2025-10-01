#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=.
export K3D_PTX_STRICT=${K3D_PTX_STRICT:-1}
export K3D_FORCE_PTX_FUSE=${K3D_FORCE_PTX_FUSE:-1}
export K3D_CONSISTENCY_FALLBACK_TEXT=${K3D_CONSISTENCY_FALLBACK_TEXT:-0}

epochs=${1:-50}
limit=${2:-5000}
lr=${3:-1e-3}

scripts/k3d_env.sh run python -m knowledge3d.tools.phase25.consistency_trainer --epochs "$epochs" --limit "$limit" --lr "$lr"

echo "Consistency trainer complete (fallback_text=$K3D_CONSISTENCY_FALLBACK_TEXT)."

