#!/usr/bin/env bash
# Queue next Wikipedia embedding shard then rebuild Galaxy when done.
# Usage: scripts/queue_wiki_embed_build.sh [N]
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${K3D_DATA_DIR:-"$ROOT_DIR/../Knowledge3D.local/datasets"}
N=${1:-500000}
LOG="$DATA_DIR/wikipedia_embed.log"
PID="$DATA_DIR/wikipedia_embed.pid"

if [[ -f "$PID" ]]; then
  old=$(cat "$PID" || true)
  if [[ -n "$old" ]] && ps -p "$old" >/dev/null 2>&1; then
    echo "[info] An embedding job is already running (PID=$old)." >&2
    exit 0
  fi
fi

cmd_embed="$ROOT_DIR/scripts/wiki_galaxy_pipeline.sh embed-next $N"
cmd_build="$ROOT_DIR/scripts/wiki_galaxy_pipeline.sh build $N"

echo "[queue] $cmd_embed && $cmd_build" | tee -a "$LOG"
(
  set -e
  date '+[%F %T] START embed-next'
  $cmd_embed
  date '+[%F %T] DONE embed-next'
  date '+[%F %T] START build'
  $cmd_build
  date '+[%F %T] DONE build'
) >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "[ok] Queued. PID=$(cat "$PID") log=$LOG"

