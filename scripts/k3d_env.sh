#!/usr/bin/env bash
# Helper for containerized execution on Debian
# Usage:
#   scripts/k3d_env.sh bootstrap     # create/install conda env k3dml with deps
#   scripts/k3d_env.sh run <cmd...>  # run command inside env with PYTHONPATH=.
set -euo pipefail
ENV_NAME=${K3D_CONDA_ENV:-k3dml}

if [[ "${1:-}" == "bootstrap" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda env list | grep -q "^${ENV_NAME}\b" || conda create -y -n "$ENV_NAME" python=3.10
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip
    conda run -n "$ENV_NAME" python -m pip install \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    conda run -n "$ENV_NAME" python -m pip install open_clip_torch pillow av soundfile \
      laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Conda env $ENV_NAME ready"
  else
    echo "[ERR] Conda not found. Install Miniconda or use a venv (see docs/ENV_POLICY.md)." >&2
    exit 1
  fi
elif [[ "${1:-}" == "bootstrap-gpu" ]]; then
  # Create GPU-enabled env using NVIDIA channel (CUDA 12.x)
  if command -v conda >/dev/null 2>&1; then
    conda env list | grep -q "^${ENV_NAME}\b" || conda create -y -n "$ENV_NAME" python=3.10
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip
    # Install PyTorch with CUDA via conda (nvidia channel)
    conda run -n "$ENV_NAME" conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
    # Remaining deps via pip
    conda run -n "$ENV_NAME" python -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Conda GPU env $ENV_NAME ready"
  else
    echo "[ERR] Conda not found. Install Miniconda or use Docker." >&2
    exit 1
  fi
elif [[ "${1:-}" == "run" ]]; then
  shift || true
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] Conda not found. Use bootstrap first or see docs/ENV_POLICY.md." >&2
    exit 1
  fi
  exec conda run -n "$ENV_NAME" env PYTHONPATH=. "$@"
else
  echo "Usage: $0 bootstrap|run <cmd...>" >&2
  exit 2
fi
