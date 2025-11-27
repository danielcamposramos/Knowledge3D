#!/bin/bash
# GPU monitoring script for K3D training runs
# Usage: Run in separate tmux before starting training
#   tmux new-session -d -s gpu_monitor "scripts/monitor_gpu.sh RUN_ID"

set -euo pipefail

RUN_ID="${1:-unknown}"
INTERVAL=5  # seconds between samples
LOG_DIR="/K3D/Knowledge3D.local/metrics/gpu"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/gpu_metrics_run_${RUN_ID}_${TIMESTAMP}.csv"

# CSV header
echo "timestamp,gpu_util_%,mem_used_mb,mem_total_mb,mem_util_%,temp_c,power_w,power_limit_w" > "$LOG_FILE"

echo "[GPU MONITOR] Started monitoring for run $RUN_ID"
echo "[GPU MONITOR] Logging to: $LOG_FILE"
echo "[GPU MONITOR] Sampling every ${INTERVAL}s"

while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits >> "$LOG_FILE"
    sleep "$INTERVAL"
done
