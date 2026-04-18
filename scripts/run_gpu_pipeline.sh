#!/usr/bin/env bash
set -euo pipefail

# Single-GPU sequential pipeline. Runs each step one after another.
# Uses your local conda env (k3d-cranium) so CUDA + PTX stack are consistent.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/gpu_pipeline_${STAMP}.log"

echo "[GPU-PIPELINE] starting @ $STAMP" | tee -a "$LOG"

# Activate conda env
if [ -f "/home/daniel/miniforge/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  . "/home/daniel/miniforge/etc/profile.d/conda.sh"
  conda activate k3d-cranium >/dev/null 2>&1 || true
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export K3D_PTX_STRICT=1
export K3D_FORCE_PTX_FUSE=1
export K3D_RPN_BEAM=1
export K3D_RPN_BEAM_WIDTH=5

OLLAMA_URL=${OLLAMA_URL:-"http://192.168.0.4:11434"}

step () { echo; echo "[GPU-PIPELINE] $1" | tee -a "$LOG"; }

# 0) Ensure external docs are fully ingested (previews + OCR)
step "Ingest PDFs/JSON with previews + OCR"
# DEPRECATED: phase25.ingest_pdf_corpus was removed. Use inject_pdf_to_galaxy instead.
python -m knowledge3d.tools.training_pipelines.inject_pdf_to_galaxy \
  --galaxy-path /K3D/Knowledge3D.local/galaxy/working/galaxy.glb \
  --roots "/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias,/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries" \
  --recursive --limit 0 | tee -a "$LOG"

# 1) Generation expansions — text (run models one at a time)
TOPICS="physics,biology,engineering,ethics,ai,systems,mathematics,economics,history,art"

step "Generate text with exaone3.5"
python -m knowledge3d.tools.gen_text_ollama \
  --ollama "$OLLAMA_URL" --models exaone3.5:latest --topics "$TOPICS" --per-topic 60 \
  --out /K3D/Knowledge3D.local/datasets/text_exaone3p5_v1.txt | tee -a "$LOG"

step "Generate text with granite3.3:8b"
python -m knowledge3d.tools.gen_text_ollama \
  --ollama "$OLLAMA_URL" --models granite3.3:8b --topics "$TOPICS" --per-topic 60 \
  --out /K3D/Knowledge3D.local/datasets/text_granite3p3_8b_v1.txt | tee -a "$LOG"

step "Generate text with gemma3:12b"
python -m knowledge3d.tools.gen_text_ollama \
  --ollama "$OLLAMA_URL" --models gemma3:12b --topics "$TOPICS" --per-topic 60 \
  --out /K3D/Knowledge3D.local/datasets/text_gemma3_12b_v1.txt | tee -a "$LOG"

step "Generate text with gemma3n"
python -m knowledge3d.tools.gen_text_ollama \
  --ollama "$OLLAMA_URL" --models gemma3n --topics "$TOPICS" --per-topic 60 \
  --out /K3D/Knowledge3D.local/datasets/text_gemma3n_v1.txt | tee -a "$LOG"

# 2) Convert generated text into K3D GLBs
mkdir -p "$ROOT_DIR/viewer/public/text"
step "Convert exaone3.5 text → GLB"
python -m k3dgen --text /K3D/Knowledge3D.local/datasets/text_exaone3p5_v1.txt \
  --gltf viewer/public/text/text_exaone3p5_v1.glb --k 10 | tee -a "$LOG"
step "Convert granite3.3:8b text → GLB"
python -m k3dgen --text /K3D/Knowledge3D.local/datasets/text_granite3p3_8b_v1.txt \
  --gltf viewer/public/text/text_granite3p3_8b_v1.glb --k 10 | tee -a "$LOG"
step "Convert gemma3:12b text → GLB"
python -m k3dgen --text /K3D/Knowledge3D.local/datasets/text_gemma3_12b_v1.txt \
  --gltf viewer/public/text/text_gemma3_12b_v1.glb --k 10 | tee -a "$LOG"
step "Convert gemma3n text → GLB"
python -m k3dgen --text /K3D/Knowledge3D.local/datasets/text_gemma3n_v1.txt \
  --gltf viewer/public/text/text_gemma3n_v1.glb --k 10 | tee -a "$LOG"

# 3) Image captions (one model at a time)
step "Image captions with qwen2.5vl"
python -m knowledge3d.tools.gen_image_captions_ollama \
  --ollama "$OLLAMA_URL" --model qwen2.5vl:7b-q8_0 \
  --images-root viewer/public/house/materialized_objects/docs --limit 200 \
  --out /K3D/Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl | tee -a "$LOG"

step "Image captions with llama3.2-vision"
python -m knowledge3d.tools.gen_image_captions_ollama \
  --ollama "$OLLAMA_URL" --model llama3.2-vision \
  --images-root viewer/public/house/materialized_objects/docs --limit 200 \
  --out /K3D/Knowledge3D.local/datasets/image_captions_llama32vision.jsonl | tee -a "$LOG"

# 4) Embedding comparison table (small encoders; still run sequentially)
step "Embedding comparison — qwen3-embedding, embeddinggemma, snowflake-arctic-embed2"
python -m knowledge3d.tools.compare_ollama_embeddings \
  --ollama "$OLLAMA_URL" \
  --models qwen3-embedding:4b,embeddinggemma,snowflake-arctic-embed2 \
  --prompts "energia; conhecimento; sistemas; probability theory; computer vision" \
  --out docs/reports/status/embedding_comparison_${STAMP}.md | tee -a "$LOG"

# 5-7) Training stages — DEPRECATED phase25 modules consolidated into train_rlwhf_policy
step "RLWHF Policy Training (replaces consistency/shapes/long_run trainers)"
python -m knowledge3d.tools.training_pipelines.train_rlwhf_policy \
  --epochs 100 --limit 5000 --lr 5e-4 | tee -a "$LOG"

# Old (deprecated) commands - phase25 modules were removed:
# python -m knowledge3d.tools.phase25.consistency_trainer --epochs 50 --limit 5000 --lr 1e-3
# python -m knowledge3d.tools.phase25.shapes_trainer --epochs 100 --limit 5000
# python -m knowledge3d.tools.phase25.long_run --epochs 50 --limit 300 --eval-every 5

# 8) RLWHF refresh + policy train (optional but included for completeness)
step "RLWHF refresh from GLB (500) + open prompts (1000) + unify"
python -m knowledge3d.tools.rlwhf_from_glb \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_glb.jsonl \
  --queries 500 | tee -a "$LOG"
python -m knowledge3d.tools.ingest_rl_open \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  --n 1000 --dataset anthropic --mode compose | tee -a "$LOG"
python -m knowledge3d.tools.merge_jsonl \
  --out docs/reports/training/rlwhf_dataset_unified.jsonl --dedup query \
  docs/reports/training/rlwhf_dataset_glb.jsonl \
  docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  docs/reports/training/rlwhf_dataset.jsonl | tee -a "$LOG"

step "RLWHF policy training (distilgpt2, 10 epochs)"
python -m knowledge3d.tools.train_rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset_unified.jsonl \
  --out /K3D/Knowledge3D.local/models/rlwhf_policy \
  --model distilgpt2 --epochs 10 --batch 4 --max_len 384 --lr 5e-5 | tee -a "$LOG"

step "Done"
echo "[GPU-PIPELINE] complete" | tee -a "$LOG"

