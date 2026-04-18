#!/usr/bin/env bash
set -euo pipefail
IMG_NAME=${IMG_NAME:-k3d-gpu:latest}
ROOT=$(pwd)
LOCAL_DIR=${K3D_LOCAL_DIR:-/K3D/Knowledge3D.local}
mkdir -p "$LOCAL_DIR/conda_pkgs" "$LOCAL_DIR/datasets" "$LOCAL_DIR/logs" "$LOCAL_DIR/models" "$LOCAL_DIR/mr"

docker run --rm -it --gpus all \
  -e K3D_ACCEL=${K3D_ACCEL:-gpu} \
  -e K3D_FAISS_DEVICE=${K3D_FAISS_DEVICE:-gpu} \
  -v "$ROOT":"/workspace" -w /workspace \
  -v "$LOCAL_DIR/conda_pkgs":"/opt/conda/pkgs" \
  -v "$LOCAL_DIR":"/k3dlocal" \
  "$IMG_NAME" bash -lc 'python -m pip install -e . && echo "Container ready. Repo installed in editable mode." && bash'

