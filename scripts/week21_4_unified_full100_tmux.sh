#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-k3d_week21_4_full100}"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-/K3D/Knowledge3D.local}"
OUTPUT_DIR="${OUTPUT_DIR:-/K3D/Knowledge3D.local/results/week21_4_unified_full100}"

# Keep monitor startup first so we do not miss the initial GPU usage spike.
MONITOR_CMD="${MONITOR_CMD:-watch -n 1 \"nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader\"}"
RUN_CMD="${RUN_CMD:-python3 scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip \
  --output-dir ${OUTPUT_DIR} \
  --storage-root ${STORAGE_ROOT}}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux attach-session -t "${SESSION_NAME}"
  exit 0
fi

# Window 0: monitor first.
tmux new-session -d -s "${SESSION_NAME}" -n "gpu_monitor"
tmux send-keys -t "${SESSION_NAME}:gpu_monitor" "cd \"${PROJECT_DIR}\"" C-m
tmux send-keys -t "${SESSION_NAME}:gpu_monitor" "${MONITOR_CMD}" C-m

# Window 1: benchmark run starts after a short delay so monitor captures startup.
tmux new-window -t "${SESSION_NAME}" -n "benchmark_run"
tmux send-keys -t "${SESSION_NAME}:benchmark_run" "cd \"${PROJECT_DIR}\"" C-m
tmux send-keys -t "${SESSION_NAME}:benchmark_run" "sleep 2; ${RUN_CMD}" C-m

tmux select-window -t "${SESSION_NAME}:gpu_monitor"
tmux attach-session -t "${SESSION_NAME}"

