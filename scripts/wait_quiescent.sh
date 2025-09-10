#!/usr/bin/env bash
set -euo pipefail
# Wait until a file stops changing (size or mtime) for a quiet period.
# Usage: scripts/wait_quiescent.sh <path> [quiet_sec]

FILE=${1:?"file path required"}
QUIET=${2:-30}

if [[ ! -e "$FILE" ]]; then
  echo "[wait_quiescent] $FILE does not exist yet; waiting for creation..."
  while [[ ! -e "$FILE" ]]; do sleep 1; done
fi

echo "[wait_quiescent] monitoring $FILE (quiet=${QUIET}s)"
last_size=-1
last_mtime=-1
last_change=$(date +%s)

while true; do
  size=$(stat -c %s "$FILE" 2>/dev/null || echo 0)
  mtime=$(stat -c %Y "$FILE" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [[ "$size" != "$last_size" || "$mtime" != "$last_mtime" ]]; then
    last_change=$now
    last_size=$size
    last_mtime=$mtime
  fi
  delta=$(( now - last_change ))
  if (( delta >= QUIET )); then
    echo "[wait_quiescent] QUIET for ${QUIET}s (size=${size})"
    exit 0
  fi
  sleep 1
done

