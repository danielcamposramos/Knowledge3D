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

echo "[VISION] warm + small batch (limit=20) with qwen2.5vl and llama3.2-vision" | tee -a "$LOG"
python -m knowledge3d.tools.gen_image_captions_ollama \
  --ollama "$OLLAMA_URL" --model qwen2.5vl:latest \
  --images-root viewer/public/house/materialized_objects/docs --limit 20 \
  --timeout 600 --cycle 10 \
  --out ../Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl | tee -a "$LOG"

python -m knowledge3d.tools.gen_image_captions_ollama \
  --ollama "$OLLAMA_URL" --model llama3.2-vision:latest \
  --images-root viewer/public/house/materialized_objects/docs --limit 20 \
  --timeout 600 --cycle 10 \
  --out ../Knowledge3D.local/datasets/image_captions_llama32vision.jsonl | tee -a "$LOG"

Q_SZ=$(stat -c%s "../Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl" 2>/dev/null || echo 0)
L_SZ=$(stat -c%s "../Knowledge3D.local/datasets/image_captions_llama32vision.jsonl" 2>/dev/null || echo 0)

echo "[VISION] sizes: qwen=$Q_SZ llama=$L_SZ" | tee -a "$LOG"

if [ "${Q_SZ:-0}" -gt 0 ]; then
  echo "[VISION] expand qwen2.5vl to 200" | tee -a "$LOG"
  python -m knowledge3d.tools.gen_image_captions_ollama \
    --ollama "$OLLAMA_URL" --model qwen2.5vl:latest \
    --images-root viewer/public/house/materialized_objects/docs --limit 200 \
    --timeout 600 --cycle 10 \
    --out ../Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl | tee -a "$LOG"
fi

if [ "${L_SZ:-0}" -gt 0 ]; then
  echo "[VISION] expand llama3.2-vision to 200" | tee -a "$LOG"
  python -m knowledge3d.tools.gen_image_captions_ollama \
    --ollama "$OLLAMA_URL" --model llama3.2-vision:latest \
    --images-root viewer/public/house/materialized_objects/docs --limit 200 \
    --timeout 600 --cycle 10 \
    --out ../Knowledge3D.local/datasets/image_captions_llama32vision.jsonl | tee -a "$LOG"
fi

echo "[VISION] complete" | tee -a "$LOG"
