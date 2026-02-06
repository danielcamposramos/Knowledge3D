#!/usr/bin/env bash
set -euo pipefail

SESSION="k3d_week13"
ROOT="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n gpu_monitor

# Window 0: GPU monitor
tmux send-keys -t "$SESSION":0 "watch -n 1 nvidia-smi" C-m

# Window 1: LLM hardening
tmux new-window -t "$SESSION":1 -n llm_enhance
tmux send-keys -t "$SESSION":1 "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION":1 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":1 "echo 'Week13 Day1-2: Local LLM hardening commands here'" C-m

# Window 2: Stargate hardening
tmux new-window -t "$SESSION":2 -n stargate
tmux send-keys -t "$SESSION":2 "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION":2 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":2 "echo 'Week13 Day3-5: Stargate crystallization commands here'" C-m

# Window 3: Tests
tmux new-window -t "$SESSION":3 -n tests
tmux send-keys -t "$SESSION":3 "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION":3 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":3 "echo 'Run focused pytest suites here'" C-m

# Window 4: Phase 1B rerun
tmux new-window -t "$SESSION":4 -n phase1b_rerun
tmux send-keys -t "$SESSION":4 "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION":4 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":4 "echo 'Run execute_knowledge_prep_phase1b.py here'" C-m

tmux select-window -t "$SESSION":0
tmux attach-session -t "$SESSION"
