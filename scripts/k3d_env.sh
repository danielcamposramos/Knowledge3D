#!/usr/bin/env bash
set -euo pipefail

# K3D environment runner: prefers conda k3d-cranium, falls back to local venv
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Always attempt to source local conda profiles, then activate k3d-cranium
if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  . "$HOME/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi

# Fallback: ensure local venv bin is first on PATH
if [ -x "$ROOT_DIR/.venv_k3dml/bin/python" ]; then
  export PATH="$ROOT_DIR/.venv_k3dml/bin:$PATH"
fi

# Common env
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export K3D_PTX_STRICT=${K3D_PTX_STRICT:-1}
export K3D_FORCE_PTX_FUSE=${K3D_FORCE_PTX_FUSE:-1}

if [[ $# -gt 0 && "$1" == "run" ]]; then
  shift 1
  exec "$@"
else
  exec "$@"
fi
