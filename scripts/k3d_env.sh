#!/usr/bin/env bash
# Helper for containerized execution on Debian
# Usage:
#   scripts/k3d_env.sh bootstrap        # create/install env (conda preferred, venv fallback)
#   scripts/k3d_env.sh bootstrap-gpu    # GPU-enabled env (conda preferred, venv fallback)
#   scripts/k3d_env.sh bootstrap-rapids # New GPU env with FAISS-GPU + RAPIDS cuML (recommended)
#   scripts/k3d_env.sh run <cmd...>     # run command inside env with PYTHONPATH=., GPU-only flags
#   scripts/k3d_env.sh shell            # open an interactive bash with `conda activate <env>` (GPU-only)
#   scripts/k3d_env.sh activate         # print the one-liner to activate this env in your current shell
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
elif [[ "${1:-}" == "bootstrap-rapids" ]]; then
  # Create a fresh GPU env with FAISS-GPU + RAPIDS cuML + PyTorch CUDA
  if have_conda; then
    conda env remove -y -n "$ENV_NAME" >/dev/null 2>&1 || true
    conda create -y -n "$ENV_NAME" python=3.10
    # Core GPU libs
    conda run -n "$ENV_NAME" conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
    # FAISS GPU + sklearn + pyarrow
    conda run -n "$ENV_NAME" conda install -y -c conda-forge faiss-gpu scikit-learn pyarrow
    # RAPIDS cuML (CUDA 12.x). Channel resolution can be slow; pinned via rapidsai.
    conda run -n "$ENV_NAME" conda install -y -c rapidsai -c conda-forge -c nvidia cuml
    # Pip deps
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip
    conda run -n "$ENV_NAME" python -m pip install \
      sentence-transformers open_clip_torch pillow av soundfile laion_clap umap-learn numpy pandas pygltflib
    echo "[OK] Conda GPU+RAPIDS env $ENV_NAME ready"
  else
    echo "[ERR] Conda not found. Install Miniconda or use Docker." >&2
    exit 1
  fi
elif [[ "${1:-}" == "run" ]]; then
  shift || true
  if have_conda; then
    # Enforce GPU-only behavior by default
    exec conda run -n "$ENV_NAME" env K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu K3D_STRICT_GPU=1 PYTHONPATH=. "$@"
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      echo "[ERR] No conda and venv $VENV_DIR not found. Run: scripts/k3d_env.sh bootstrap or bootstrap-gpu" >&2
      exit 1
    fi
    # Prepend venv bin to PATH so 'python' resolves into the venv
    # If venv fallback is used, still set strict GPU flags; tools will error if GPU libs missing
    exec env PATH="$VENV_DIR/bin:$PATH" K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu K3D_STRICT_GPU=1 PYTHONPATH=. "$@"
  fi
elif [[ "${1:-}" == "shell" ]]; then
  # Open an interactive shell using `conda activate <env>` with GPU-only flags
  if ! have_conda; then
    echo "[ERR] Conda not found. Install Miniconda/Miniforge or use 'run' with venv." >&2
    exit 1
  fi
  exec bash -i -c 'eval "$(conda shell.bash hook)" && conda activate "'$ENV_NAME'" && export K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu K3D_STRICT_GPU=1 PYTHONPATH=. && echo "[OK] Activated '$ENV_NAME' (GPU-only)." && bash -i'
elif [[ "${1:-}" == "activate" ]]; then
  # Print the one-liner to activate in the current shell session
  cat <<EOF
# Paste this into your shell to activate (GPU-only):
eval "\$(conda shell.bash hook)"; conda activate "$ENV_NAME"; export K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu K3D_STRICT_GPU=1 PYTHONPATH=.
EOF
  exit 0
else
  echo "Usage: $0 bootstrap|run <cmd...>" >&2
  exit 2
fi
