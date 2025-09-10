#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a dedicated FAISS-GPU environment to avoid conflicts with the main k3dml env.
# - Python 3.10
# - Prefer conda-forge FAISS-GPU
# - If solver conflicts or ABI issues arise, try pip cu12 wheels
# - Optionally install CUDA 12.4 libs from NVIDIA channel to satisfy libcublasLt symbols

ENV_NAME=${K3D_FAISS_ENV:-k3dfaiss}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERR] Conda not found. Please install Miniconda/Conda first." >&2
  exit 1
fi

echo "[FAISS] Creating env: $ENV_NAME (python=3.10)"
conda env list | grep -q "^${ENV_NAME}\b" || conda create -y -n "$ENV_NAME" python=3.10

echo "[FAISS] Attempting conda-forge faiss-gpu"
set +e
conda run -n "$ENV_NAME" conda install -y -c conda-forge faiss-gpu
code=$?
set -e
if [[ $code -ne 0 ]]; then
  echo "[FAISS] conda-forge install failed; trying pip cu12 wheel"
  conda run -n "$ENV_NAME" python -m pip install --upgrade pip
  conda run -n "$ENV_NAME" python -m pip install 'faiss-gpu-cu12'
fi

echo "[FAISS] Validating GPU availability..."
set +e
conda run -n "$ENV_NAME" python - <<'PY'
import sys
try:
    import faiss
    print('faiss', getattr(faiss, '__version__', '?'))
    try:
        g = faiss.get_num_gpus()
        print('faiss.get_num_gpus()', g)
        if g <= 0:
            raise SystemExit(2)
    except Exception as e:
        print('faiss.get_num_gpus error', e)
        raise SystemExit(2)
except Exception as e:
    print('faiss import error', e)
    raise SystemExit(1)
PY
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "[FAISS] GPU not detected by faiss. Installing CUDA 12.4 libs (nvidia channel) to satisfy cublasLt..."
  conda run -n "$ENV_NAME" conda install -y -c nvidia cuda-version=12.4 cuda-cudart=12.4 cuda-libraries=12.4 cuda-cublas=12.4 || true
  echo "[FAISS] Re-validating faiss GPU..."
  conda run -n "$ENV_NAME" python - <<'PY'
import sys
import faiss
print('faiss', getattr(faiss, '__version__','?'))
print('faiss.get_num_gpus()', getattr(faiss,'get_num_gpus',lambda:-1)())
PY
fi

echo "[OK] Env '$ENV_NAME' ready. Use with: K3D_CONDA_ENV=$ENV_NAME scripts/k3d_env.sh run <cmd>"

