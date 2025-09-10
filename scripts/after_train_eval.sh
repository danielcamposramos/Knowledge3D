#!/usr/bin/env bash
set -euo pipefail
# Wait for a model train_info.json, then run eval on N samples
# Usage: scripts/after_train_eval.sh <model_dir> <dataset_jsonl> <eval_out_json> <limit>

MODEL_DIR=${1:?"model_dir required"}
DATASET=${2:?"dataset path required"}
OUT=${3:?"eval out path required"}
LIMIT=${4:-500}

INFO="$MODEL_DIR/train_info.json"
echo "[after_train_eval] waiting for $INFO ..."
while [[ ! -f "$INFO" ]]; do sleep 5; done

# Optional quiet wait: file updated? use mtime check for 10s
MT=$(stat -c %Y "$INFO")
sleep 10
if [[ $(stat -c %Y "$INFO") != "$MT" ]]; then
  echo "[after_train_eval] train_info updated; waiting extra 10s"
  sleep 10
fi

echo "[after_train_eval] running eval limit=$LIMIT -> $OUT"
scripts/k3d_env.sh run python -m knowledge3d.tools.eval_rlwhf_policy \
  --dataset "$DATASET" --model "$MODEL_DIR" --out "$OUT" --limit "$LIMIT"
echo "[after_train_eval] done"

