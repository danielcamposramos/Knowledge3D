#!/usr/bin/env bash
# Helper script to run codec benchmarks inside the CUDA-enabled conda env.
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run CUDA benchmarks" >&2
  exit 1
fi

ENV_NAME="k3d-cranium"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env ${ENV_NAME} not found. Create it with: conda env create -f envs/k3d-cranium.yml" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" env | grep -E "CUDA|LD_LIBRARY_PATH" || true
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n "${ENV_NAME}" python scripts/benchmark_ternary_audio.py --gpu
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n "${ENV_NAME}" python scripts/benchmark_ternary_video.py
