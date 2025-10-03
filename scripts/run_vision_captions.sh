#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/vision_captions_${STAMP}.log"

# Activate conda
if [ -f "/home/daniel/miniforge/etc/profile.d/conda.sh" ]; then
  . "/home/daniel/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
OLLAMA_URL=${OLLAMA_URL:-"http://192.168.0.4:11434"}

echo "[VISION] alternating 20-image batches for qwen2.5vl and llama3.2-vision" | tee -a "$LOG"

TARGET=${VISION_CAPTION_TARGET:-200}
BATCHES=${VISION_CAPTION_BATCHES:-12}
QWEN_MODEL=${VISION_QWEN_MODEL:-qwen2.5vl:latest}
LLAMA_MODEL=${VISION_LLAMA_MODEL:-llama3.2-vision:latest}
QWEN_OUT=../Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl
LLAMA_OUT=../Knowledge3D.local/datasets/image_captions_llama32vision.jsonl

ensure_count() {
  local file=$1
  if [ ! -f "$file" ]; then
    echo 0
  else
    wc -l < "$file"
  fi
}

run_batch() {
  local model=$1
  local out=$2
  local timeout=$3
  local label=$4
  python -m knowledge3d.tools.gen_image_captions_ollama \
    --ollama "$OLLAMA_URL" --model "$model" \
    --images-root viewer/public/house/materialized_objects/docs --limit 20 \
    --timeout "$timeout" --cycle 20 \
    --out "$out" | tee -a "$LOG"
  local count=$(ensure_count "$out")
  echo "[VISION] $label count=$count" | tee -a "$LOG"
}

for ((i=1; i<=BATCHES; ++i)); do
  q_count=$(ensure_count "$QWEN_OUT")
  if [ "$q_count" -lt "$TARGET" ]; then
    echo "[VISION] batch $i/$BATCHES — qwen2.5vl" | tee -a "$LOG"
    run_batch "$QWEN_MODEL" "$QWEN_OUT" 900 "qwen"
  else
    echo "[VISION] qwen target reached ($q_count >= $TARGET), skipping" | tee -a "$LOG"
  fi

  l_count=$(ensure_count "$LLAMA_OUT")
  if [ "$l_count" -lt "$TARGET" ]; then
    echo "[VISION] batch $i/$BATCHES — llama3.2-vision" | tee -a "$LOG"
    run_batch "$LLAMA_MODEL" "$LLAMA_OUT" 2400 "llama"
  else
    echo "[VISION] llama target reached ($l_count >= $TARGET), skipping" | tee -a "$LOG"
  fi

  q_count=$(ensure_count "$QWEN_OUT")
  l_count=$(ensure_count "$LLAMA_OUT")
  if [ "$q_count" -ge "$TARGET" ] && [ "$l_count" -ge "$TARGET" ]; then
    break
  fi
done

Q_SZ=$(ensure_count "$QWEN_OUT")
L_SZ=$(ensure_count "$LLAMA_OUT")
echo "[VISION] final counts: qwen=$Q_SZ llama=$L_SZ" | tee -a "$LOG"
echo "[VISION] complete" | tee -a "$LOG"
