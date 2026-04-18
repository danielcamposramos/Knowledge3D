#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="k3d_week14_benchmarks"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux attach-session -t "${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n "gpu_monitor"
tmux send-keys -t "${SESSION_NAME}:gpu_monitor" "watch -n 1 nvidia-smi" C-m

tmux new-window -t "${SESSION_NAME}" -n "arc_agi"
tmux send-keys -t "${SESSION_NAME}:arc_agi" "cd \"${PROJECT_DIR}\"" C-m

tmux new-window -t "${SESSION_NAME}" -n "math"
tmux send-keys -t "${SESSION_NAME}:math" "cd \"${PROJECT_DIR}\"" C-m

tmux new-window -t "${SESSION_NAME}" -n "lhe"
tmux send-keys -t "${SESSION_NAME}:lhe" "cd \"${PROJECT_DIR}\"" C-m

tmux new-window -t "${SESSION_NAME}" -n "all_benchmarks"
tmux send-keys -t "${SESSION_NAME}:all_benchmarks" "cd \"${PROJECT_DIR}\"" C-m

tmux select-window -t "${SESSION_NAME}:gpu_monitor"
tmux attach-session -t "${SESSION_NAME}"
