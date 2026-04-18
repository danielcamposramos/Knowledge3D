#!/usr/bin/env bash
# Overnight PDF Ingestion - Simple Launcher
#
# Processes all 1,952 PDFs from database using existing infrastructure
# Expected time: 8-12 hours
#
# Usage:
#   tmux new -s k3d_pdf_ingestion
#   bash scripts/run_overnight_pdf_ingestion.sh
#   # Detach: Ctrl+b then d
#   # Reattach: tmux attach -t k3d_pdf_ingestion
#   # Check status: tail -f /tmp/k3d_overnight_pdf_ingestion.log

set -euo pipefail

PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
DATABASE_ROOT="/mnt/arquivos/0 ChatGPTs/DataBase"
OUTPUT_DIR="/K3D/Knowledge3D.local/fundamental_augmentation"
CACHE_DIR="/K3D/Knowledge3D.local/pdf_cache"
LOG_FILE="/tmp/k3d_overnight_pdf_ingestion.log"

# Output files
# Keep stable names by default so interrupted runs can resume from staging.
# Override with K3D_INGEST_RUN_ID if you want separate concurrent run artifacts.
RUN_ID="${K3D_INGEST_RUN_ID:-overnight}"
PAYLOAD_OUTPUT="$OUTPUT_DIR/full_pdf_payloads_${RUN_ID}.jsonl"
REPORT_OUTPUT="$OUTPUT_DIR/full_pdf_report_${RUN_ID}.json"

echo "=========================================" | tee -a "$LOG_FILE"
echo "K3D Overnight PDF Ingestion" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Database: $DATABASE_ROOT" | tee -a "$LOG_FILE"
echo "PDFs to process: ~1,952" | tee -a "$LOG_FILE"
echo "Classifier: deepseek-r1:14b" | tee -a "$LOG_FILE"
echo "Augmenter: qwen2.5:14b" | tee -a "$LOG_FILE"
echo "Output: $PAYLOAD_OUTPUT" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
# Start fresh log for each run (old runs can be archived externally).
: > "$LOG_FILE"

# Run ingestion
cd "$(dirname "$0")/.."
echo "Starting ingestion..." | tee -a "$LOG_FILE"

set +e
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf-dir "$DATABASE_ROOT" \
  --pattern "**/*.pdf" \
  --limit-pdfs 2000 \
  --max-pages-per-pdf 0 \
  --classifier-model "deepseek-r1:14b" \
  --augmenter-model "qwen2.5:14b" \
  --ollama-timeout 180.0 \
  --cache-dir "$CACHE_DIR" \
  --payload-output "$PAYLOAD_OUTPUT" \
  --report-output "$REPORT_OUTPUT" \
  2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Ingestion COMPLETE" | tee -a "$LOG_FILE"
else
    echo "Ingestion FAILED (exit code $EXIT_CODE)" | tee -a "$LOG_FILE"
fi
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Output: $PAYLOAD_OUTPUT" | tee -a "$LOG_FILE"
echo "Report: $REPORT_OUTPUT" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

if [ -f "$PAYLOAD_OUTPUT" ]; then
    ENTRIES=$(wc -l < "$PAYLOAD_OUTPUT")
    echo "Total entries: $ENTRIES" | tee -a "$LOG_FILE"
fi

echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Next step - Ingest to Galaxy:" | tee -a "$LOG_FILE"
    echo "PYTHONPATH=. $PYTHON_BIN scripts/fundamental_ingest_payloads.py \\" | tee -a "$LOG_FILE"
    echo "  --payload $PAYLOAD_OUTPUT \\" | tee -a "$LOG_FILE"
    echo "  --storage-root /K3D/Knowledge3D.local \\" | tee -a "$LOG_FILE"
    echo "  --report /K3D/Knowledge3D.local/results/overnight_pdf_ingestion_report.json" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
