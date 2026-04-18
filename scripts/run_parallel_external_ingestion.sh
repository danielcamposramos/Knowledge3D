#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/mnt/anaconda3/bin/conda}"
if [ ! -x "$CONDA_BIN" ]; then
  echo "[parallel] ERROR: conda not found at $CONDA_BIN" >&2
  exit 1
fi

# Build external payloads in parallel, then ingest in a single persistent world.
# Usage:
#   bash scripts/run_parallel_external_ingestion.sh [storage_root] [output_root]
# Optional env:
#   INCLUDE_BENCHMARK_AUGMENTATION=1

STORAGE_ROOT="${1:-/K3D/Knowledge3D.local}"
OUTPUT_ROOT="${2:-/K3D/Knowledge3D.local/datasets/external_payloads}"
PAYLOAD_DIR="${OUTPUT_ROOT%/}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PAYLOAD_DIR"

echo "[parallel] payload_dir=$PAYLOAD_DIR"

# Build three modality payloads concurrently.
"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/prepare_external_multicurriculum_payload.py \
  --modality lexicon \
  --output "$PAYLOAD_DIR/lexicon_payload.jsonl" &
PID_LEX=$!

"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/prepare_external_multicurriculum_payload.py \
  --modality audio \
  --output "$PAYLOAD_DIR/audio_payload.jsonl" \
  --max-audio 5000 &
PID_AUD=$!

"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/prepare_external_multicurriculum_payload.py \
  --modality geometry3d \
  --output "$PAYLOAD_DIR/geometry3d_payload.jsonl" \
  --use-fallback-templates \
  --max-geometry 2000 &
PID_GEO=$!

wait "$PID_LEX" "$PID_AUD" "$PID_GEO"

echo "[parallel] payload build complete"

# Single-world ingestion apply.
"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/fundamental_ingest_payloads.py \
  --storage-root "$STORAGE_ROOT" \
  --payload "$PAYLOAD_DIR/lexicon_payload.jsonl" "$PAYLOAD_DIR/audio_payload.jsonl" "$PAYLOAD_DIR/geometry3d_payload.jsonl" \
  --report "$PAYLOAD_DIR/ingestion_report.json"

echo "[parallel] done report=$PAYLOAD_DIR/ingestion_report.json"

if [ "${INCLUDE_BENCHMARK_AUGMENTATION:-0}" = "1" ]; then
  echo "[parallel] running benchmark augmentation + single-world ingest"
  OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}" \
  OLLAMA_STRIDE="${OLLAMA_STRIDE:-50}" \
  MAX_OLLAMA_CALLS="${MAX_OLLAMA_CALLS:-200}" \
  MAX_ARC_TASKS="${MAX_ARC_TASKS:-400}" \
  MAX_MATH_PROBLEMS="${MAX_MATH_PROBLEMS:-2000}" \
  MAX_LHE_QUESTIONS="${MAX_LHE_QUESTIONS:-2500}" \
  MAX_MMLU_QUESTIONS="${MAX_MMLU_QUESTIONS:-2000}" \
  MAX_WORD_ENTRIES="${MAX_WORD_ENTRIES:-50000}" \
  bash scripts/fundamental_construct_knowledge.sh "$STORAGE_ROOT" "$OUTPUT_ROOT"
fi
