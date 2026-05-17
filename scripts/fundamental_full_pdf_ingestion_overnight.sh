#!/usr/bin/env bash
# Fundamental Full PDF Ingestion - Overnight Run
#
# Purpose: Process ALL PDFs from database (1,952 files, 42GB)
# Strategy: Batch processing with checkpointing, progress tracking, tmux persistence
# Expected Time: 8-12 hours (depends on Ollama throughput)
#
# Usage:
#   tmux new -s k3d_pdf_ingestion
#   bash scripts/fundamental_full_pdf_ingestion_overnight.sh
#   # Detach: Ctrl+b then d
#   # Reattach: tmux attach -t k3d_pdf_ingestion

set -euo pipefail

# Configuration
PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATABASE_ROOT="/mnt/arquivos/0 ChatGPTs/DataBase"
OUTPUT_ROOT="/K3D/Knowledge3D.local/fundamental_augmentation"
CACHE_DIR="/K3D/Knowledge3D.local/pdf_cache"
LOG_DIR="/K3D/Knowledge3D.local/logs/overnight_ingestion_$(date +%Y%m%d_%H%M%S)"

# Models
CLASSIFIER_MODEL="deepseek-r1:14b"
AUGMENTER_MODEL="qwen2.5:14b"
OLLAMA_TIMEOUT="180.0"

# Batch settings
BATCH_SIZE=50  # PDFs per batch
MAX_PAGES_PER_PDF=0  # 0 = all pages
PARALLEL_BATCHES=1  # Sequential for stability

# Create directories
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$CACHE_DIR"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "K3D Fundamental Full PDF Ingestion"
echo "========================================="
echo "Database: $DATABASE_ROOT"
echo "Total PDFs: $(find "$DATABASE_ROOT" -name "*.pdf" -type f 2>/dev/null | wc -l)"
echo "Database size: $(du -sh "$DATABASE_ROOT" | cut -f1)"
echo "Output: $OUTPUT_ROOT"
echo "Cache: $CACHE_DIR"
echo "Logs: $LOG_DIR"
echo "Classifier: $CLASSIFIER_MODEL"
echo "Augmenter: $AUGMENTER_MODEL"
echo "Batch size: $BATCH_SIZE PDFs"
echo "========================================="
echo ""

# Function: Process single batch
process_batch() {
    local batch_num=$1
    local batch_start=$2
    local batch_end=$3

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num: Processing PDFs $batch_start to $batch_end"

    local batch_output="$OUTPUT_ROOT/pdf_payloads_batch_${batch_num}.jsonl"
    local batch_report="$OUTPUT_ROOT/pdf_report_batch_${batch_num}.json"
    local batch_log="$LOG_DIR/batch_${batch_num}.log"

    # Run ingestion for this batch
    cd "$PROJECT_ROOT"
    set +e
    PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
        --pdf-dir "$DATABASE_ROOT" \
        --pattern "**/*.pdf" \
        --limit-pdfs "$batch_end" \
        --skip-first "$batch_start" \
        --max-pages-per-pdf "$MAX_PAGES_PER_PDF" \
        --classifier-model "$CLASSIFIER_MODEL" \
        --augmenter-model "$AUGMENTER_MODEL" \
        --ollama-timeout "$OLLAMA_TIMEOUT" \
        --cache-dir "$CACHE_DIR" \
        --payload-output "$batch_output" \
        --report-output "$batch_report" \
        2>&1 | tee "$batch_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num: SUCCESS"
        echo "  Output: $batch_output ($(wc -l < "$batch_output" 2>/dev/null || echo 0) entries)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch $batch_num: FAILED (exit code $exit_code)"
        echo "  Check log: $batch_log"
    fi

    return $exit_code
}

# Main ingestion loop
TOTAL_PDFS=$(find "$DATABASE_ROOT" -name "*.pdf" -type f 2>/dev/null | wc -l)
TOTAL_BATCHES=$(( ($TOTAL_PDFS + $BATCH_SIZE - 1) / $BATCH_SIZE ))

echo "Total batches: $TOTAL_BATCHES"
echo "Starting ingestion at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)
SUCCESSFUL_BATCHES=0
FAILED_BATCHES=0

for ((batch=1; batch<=$TOTAL_BATCHES; batch++)); do
    batch_start=$(( ($batch - 1) * $BATCH_SIZE ))
    batch_end=$(( $batch * $BATCH_SIZE ))

    if [ $batch_end -gt $TOTAL_PDFS ]; then
        batch_end=$TOTAL_PDFS
    fi

    if process_batch "$batch" "$batch_start" "$batch_end"; then
        SUCCESSFUL_BATCHES=$((SUCCESSFUL_BATCHES + 1))
    else
        FAILED_BATCHES=$((FAILED_BATCHES + 1))

        # Optional: Stop on first failure (comment out to continue)
        # echo "Stopping due to batch failure"
        # break
    fi

    # Progress update
    COMPLETED_PDFS=$batch_end
    PROGRESS_PCT=$(( 100 * $COMPLETED_PDFS / $TOTAL_PDFS ))
    ELAPSED=$(( $(date +%s) - $START_TIME ))
    ELAPSED_MIN=$(( $ELAPSED / 60 ))

    if [ $COMPLETED_PDFS -gt 0 ]; then
        AVG_TIME_PER_PDF=$(( $ELAPSED / $COMPLETED_PDFS ))
        REMAINING_PDFS=$(( $TOTAL_PDFS - $COMPLETED_PDFS ))
        ETA_SEC=$(( $AVG_TIME_PER_PDF * $REMAINING_PDFS ))
        ETA_MIN=$(( $ETA_SEC / 60 ))
        ETA_HR=$(( $ETA_MIN / 60 ))

        echo ""
        echo "========================================="
        echo "Progress: $COMPLETED_PDFS / $TOTAL_PDFS PDFs ($PROGRESS_PCT%)"
        echo "Batches: $SUCCESSFUL_BATCHES successful, $FAILED_BATCHES failed"
        echo "Elapsed: ${ELAPSED_MIN} minutes"
        echo "ETA: ${ETA_HR}h ${ETA_MIN}m remaining"
        echo "========================================="
        echo ""
    fi

    # Sleep between batches to avoid overwhelming Ollama
    if [ $batch -lt $TOTAL_BATCHES ]; then
        echo "Cooling down for 10 seconds..."
        sleep 10
    fi
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( $END_TIME - $START_TIME ))
TOTAL_ELAPSED_MIN=$(( $TOTAL_ELAPSED / 60 ))
TOTAL_ELAPSED_HR=$(( $TOTAL_ELAPSED_MIN / 60 ))

echo ""
echo "========================================="
echo "Ingestion Complete!"
echo "========================================="
echo "Total time: ${TOTAL_ELAPSED_HR}h ${TOTAL_ELAPSED_MIN}m"
echo "Successful batches: $SUCCESSFUL_BATCHES / $TOTAL_BATCHES"
echo "Failed batches: $FAILED_BATCHES"
echo ""

# Merge all batch payloads
echo "Merging batch payloads..."
MERGED_OUTPUT="$OUTPUT_ROOT/full_pdf_payloads_overnight.jsonl"
cat "$OUTPUT_ROOT"/pdf_payloads_batch_*.jsonl > "$MERGED_OUTPUT" 2>/dev/null || true
TOTAL_ENTRIES=$(wc -l < "$MERGED_OUTPUT" 2>/dev/null || echo 0)

echo "Merged output: $MERGED_OUTPUT"
echo "Total entries: $TOTAL_ENTRIES"
echo ""

# Generate summary report
SUMMARY_REPORT="$OUTPUT_ROOT/overnight_ingestion_summary.json"
cat > "$SUMMARY_REPORT" <<EOF
{
  "execution_date": "$(date -Iseconds)",
  "total_time_seconds": $TOTAL_ELAPSED,
  "total_time_hours": $(awk "BEGIN {printf \"%.2f\", $TOTAL_ELAPSED / 3600}"),
  "total_pdfs_attempted": $TOTAL_PDFS,
  "total_batches": $TOTAL_BATCHES,
  "successful_batches": $SUCCESSFUL_BATCHES,
  "failed_batches": $FAILED_BATCHES,
  "total_entries_generated": $TOTAL_ENTRIES,
  "classifier_model": "$CLASSIFIER_MODEL",
  "augmenter_model": "$AUGMENTER_MODEL",
  "cache_directory": "$CACHE_DIR",
  "merged_payload": "$MERGED_OUTPUT",
  "log_directory": "$LOG_DIR"
}
EOF

echo "Summary report: $SUMMARY_REPORT"
cat "$SUMMARY_REPORT"

echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Review logs: ls -lh $LOG_DIR/"
echo "2. Check cache: ls -lh $CACHE_DIR/"
echo "3. Validate payload: head -5 $MERGED_OUTPUT"
echo "4. Ingest to Galaxy:"
echo "   PYTHONPATH=. $PYTHON_BIN scripts/fundamental_ingest_payloads.py \\"
echo "     --payload $MERGED_OUTPUT \\"
echo "     --storage-root /K3D/Knowledge3D.local \\"
echo "     --report /K3D/Knowledge3D.local/results/overnight_pdf_ingestion_report.json"
echo ""
echo "Tmux session: tmux attach -t k3d_pdf_ingestion"
echo "========================================="
