#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=.

echo "[1/4] RLWHF from GLB: queries=500"
scripts/k3d_env.sh run python -m knowledge3d.tools.rlwhf_from_glb \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_glb.jsonl \
  --queries 500

echo "[2/4] RLWHF from open prompts (Anthropic n=1000, compose)"
scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_rl_open \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  --n 1000 --dataset anthropic --mode compose

echo "[3/4] Merge unified RLWHF dataset"
scripts/k3d_env.sh run python -m knowledge3d.tools.merge_jsonl \
  --out docs/reports/training/rlwhf_dataset_unified.jsonl --dedup query \
  docs/reports/training/rlwhf_dataset_glb.jsonl \
  docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  docs/reports/training/rlwhf_dataset.jsonl || true

echo "[4/4] Train RLWHF policy (distilgpt2, 10 epochs)"
scripts/k3d_env.sh run python -m knowledge3d.tools.train_rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset_unified.jsonl \
  --out /K3D/Knowledge3D.local/models/rlwhf_policy \
  --model distilgpt2 --epochs 10 --batch 4 --max_len 384 --lr 5e-5

echo "Done. RLWHF unified dataset + policy ready."

