#!/usr/bin/env bash
set -euo pipefail
# Kill running knowledge3d live servers and free default ports.

PIDS=$(ps -ef | awk '/python -m knowledge3d\.bridge\.live_server/{print $2}') || true
if [[ -n "${PIDS:-}" ]]; then
  echo "[CLEANUP] Killing live_server PIDs: $PIDS"
  echo "$PIDS" | xargs -r kill || true
fi

for port in 8765 8787 8788 8789; do
  pid=$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $6}' | sed 's/.*pid=\([0-9]*\).*/\1/' | head -n1)
  if [[ -n "$pid" ]]; then
    echo "[CLEANUP] Killing pid $pid on port $port"
    kill "$pid" || true
  fi
done

echo "[CLEANUP] Done."

