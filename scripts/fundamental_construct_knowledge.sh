#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/mnt/anaconda3/bin/conda}"
if [ ! -x "$CONDA_BIN" ]; then
  echo "[augment-ingest] ERROR: conda not found at $CONDA_BIN" >&2
  exit 1
fi

# Fundamental Knowledge Construction Pipeline
#
# PURPOSE:
#   Foundational bootstrap/expansion of Knowledgeverse from benchmark corpora.
#   This is not hot-path inference. It is a construction pipeline.
#
# USAGE:
#   bash scripts/fundamental_construct_knowledge.sh [storage_root] [output_root]
# Optional env vars:
#   OLLAMA_MODEL=llama3.2
#   OLLAMA_STRIDE=50
#   MAX_OLLAMA_CALLS=200
#   SKIP_OLLAMA_ENRICHMENT=1  (emergency diagnostics only)
#   MAX_ARC_TASKS=400
#   MAX_MATH_PROBLEMS=2000
#   MAX_LHE_QUESTIONS=2500
#   MAX_MMLU_QUESTIONS=2000
#   MAX_WORD_ENTRIES=50000

STORAGE_ROOT="${1:-../Knowledge3D.local}"
OUTPUT_ROOT="${2:-../Knowledge3D.local/datasets/external_payloads}"
RUN_DIR="${OUTPUT_ROOT%/}/benchmark_aug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

PAYLOAD="$RUN_DIR/benchmark_augmentation_payload.jsonl"
REPORT="$RUN_DIR/benchmark_augmentation_report.json"
INGEST_REPORT="$RUN_DIR/benchmark_augmentation_ingest_report.json"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
OLLAMA_STRIDE="${OLLAMA_STRIDE:-50}"
MAX_OLLAMA_CALLS="${MAX_OLLAMA_CALLS:-200}"
SKIP_OLLAMA_ENRICHMENT="${SKIP_OLLAMA_ENRICHMENT:-0}"
MAX_ARC_TASKS="${MAX_ARC_TASKS:-400}"
MAX_MATH_PROBLEMS="${MAX_MATH_PROBLEMS:-2000}"
MAX_LHE_QUESTIONS="${MAX_LHE_QUESTIONS:-2500}"
MAX_MMLU_QUESTIONS="${MAX_MMLU_QUESTIONS:-2000}"
MAX_WORD_ENTRIES="${MAX_WORD_ENTRIES:-50000}"

OLLAMA_ARGS=(--ollama-model "$OLLAMA_MODEL" --ollama-stride "$OLLAMA_STRIDE" --max-ollama-calls "$MAX_OLLAMA_CALLS")
if [ "$SKIP_OLLAMA_ENRICHMENT" = "1" ]; then
  echo "[augment-ingest] WARNING: SKIP_OLLAMA_ENRICHMENT=1 (emergency mode, architecture override)" >&2
  OLLAMA_ARGS+=(--skip-ollama-enrichment)
fi

echo "[augment-ingest] run_dir=$RUN_DIR"
"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/fundamental_augment_benchmarks.py \
  --dataset-root "$STORAGE_ROOT/datasets" \
  --output "$PAYLOAD" \
  --report "$REPORT" \
  --max-arc-tasks "$MAX_ARC_TASKS" \
  --max-math-problems "$MAX_MATH_PROBLEMS" \
  --max-lhe-questions "$MAX_LHE_QUESTIONS" \
  --max-mmlu-questions "$MAX_MMLU_QUESTIONS" \
  --max-word-entries "$MAX_WORD_ENTRIES" \
  "${OLLAMA_ARGS[@]}"

echo "[augment-ingest] generated payload"
"${CONDA_BIN}" run -n k3d-cranium env PYTHONPATH=. python scripts/fundamental_ingest_payloads.py \
  --storage-root "$STORAGE_ROOT" \
  --payload "$PAYLOAD" \
  --report "$INGEST_REPORT"

echo "[augment-ingest] done"
echo "[augment-ingest] payload=$PAYLOAD"
echo "[augment-ingest] report=$REPORT"
echo "[augment-ingest] ingest_report=$INGEST_REPORT"
