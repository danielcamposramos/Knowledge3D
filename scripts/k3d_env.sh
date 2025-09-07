#!/usr/bin/env bash
# Helper for containerized execution on Debian
# Usage:
#   scripts/k3d_env.sh bootstrap        # create/install env (conda preferred, venv fallback)
#   scripts/k3d_env.sh bootstrap-gpu    # GPU-enabled env (conda preferred, venv fallback)
#   scripts/k3d_env.sh run <cmd...>     # run command inside env with PYTHONPATH=.
set -euo pipefail
ENV_NAME=${K3D_CONDA_ENV:-k3dml}
VENV_DIR=${K3D_VENV_DIR:-.venv_k3dml}

have_conda() { command -v conda >/dev/null 2>&1; }
ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
  fi
}

if [[ "${1:-}" == "bootstrap" ]]; then
  if have_conda; then
    conda env list | grep -q "^${ENV_NAME}\b" || conda create -y -n "$ENV_NAME" python=3.10
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip
    conda run -n "$ENV_NAME" python -m pip install \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    conda run -n "$ENV_NAME" python -m pip install open_clip_torch pillow av soundfile \
      laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Conda env $ENV_NAME ready"
  else
    echo "[INFO] Conda not found; bootstrapping Python venv at $VENV_DIR (CPU)."
    ensure_venv
    "$VENV_DIR/bin/python" -m pip install \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    "$VENV_DIR/bin/python" -m pip install open_clip_torch pillow av soundfile \
      laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Venv $VENV_DIR ready (CPU)"
  fi
elif [[ "${1:-}" == "bootstrap-gpu" ]]; then
  # Create GPU-enabled env using NVIDIA channel (CUDA 12.x)
  if have_conda; then
    conda env list | grep -q "^${ENV_NAME}\b" || conda create -y -n "$ENV_NAME" python=3.10
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip
    # Install PyTorch with CUDA via conda (nvidia channel)
    conda run -n "$ENV_NAME" conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
    # Remaining deps via pip
    conda run -n "$ENV_NAME" python -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Conda GPU env $ENV_NAME ready"
  else
    echo "[INFO] Conda not found; bootstrapping Python venv at $VENV_DIR (GPU wheels)."
    ensure_venv
    # Install PyTorch with CUDA wheels (cu121) from pytorch index
    "$VENV_DIR/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio
    "$VENV_DIR/bin/python" -m pip install open_clip_torch pillow av soundfile laion_clap umap-learn scikit-learn numpy pandas pygltflib
    echo "[OK] Venv $VENV_DIR ready (GPU)"
  fi
elif [[ "${1:-}" == "run" ]]; then
  shift || true
  if have_conda; then
    exec conda run -n "$ENV_NAME" env PYTHONPATH=. "$@"
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      echo "[ERR] No conda and venv $VENV_DIR not found. Run: scripts/k3d_env.sh bootstrap or bootstrap-gpu" >&2
      exit 1
    fi
    # Prepend venv bin to PATH so 'python' resolves into the venv
    exec env PATH="$VENV_DIR/bin:$PATH" PYTHONPATH=. "$@"
  fi
else
  echo "Usage: $0 bootstrap|run <cmd...>" >&2
  exit 2
fi
