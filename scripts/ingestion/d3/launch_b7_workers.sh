#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

WORKERS="${1:-4}"
STOP_AFTER="${2:-3600}"
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

PIDS=()
echo "Launching ${WORKERS} B7 workers with stop-after=${STOP_AFTER}s"

for INDEX in $(seq 1 "$WORKERS"); do
  WORKER_ID="${HOSTNAME:-$(hostname)}-$$-${INDEX}-$(date +%s)"
  nohup python3 scripts/ingestion/d3/differentiate_b7_residual.py \
    --worker \
    --worker-id "$WORKER_ID" \
    --cluster-dir scripts/ingestion/staging/D3_dedup/differentiate_b7/clusters \
    --out-root scripts/ingestion/staging/D3_dedup/differentiate_b7 \
    --row-concurrency 4 \
    --stop-after "$STOP_AFTER" \
    "${EXTRA_ARGS[@]}" \
    >/dev/null 2>&1 &
  PIDS+=("$!")
  echo "worker_id=${WORKER_ID} pid=${PIDS[-1]}"
done

while true; do
  python3 scripts/ingestion/d3/differentiate_b7_residual.py --status --out-root scripts/ingestion/staging/D3_dedup/differentiate_b7 || true
  ALIVE=0
  for PID in "${PIDS[@]}"; do
    if kill -0 "$PID" 2>/dev/null; then
      ALIVE=1
      break
    fi
  done
  if [[ "$ALIVE" -eq 0 ]]; then
    break
  fi
  sleep 30
done
