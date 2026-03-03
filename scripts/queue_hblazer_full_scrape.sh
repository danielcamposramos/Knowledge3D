#!/usr/bin/env bash
# Queue full hblazer scrape after topical scrape completes.
# Runs sequentially (single worker) to keep request profile conservative.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"

TOPICAL_STATE="$PROJECT_ROOT/../Knowledge3D.local/scrapes/hblazer_substack/state.json"
FULL_OUTPUT="$PROJECT_ROOT/../Knowledge3D.local/scrapes/hblazer_substack_full"
LOG_FILE="/tmp/k3d_hblazer_full_queue.log"

MIN_DELAY="${MIN_DELAY:-60}"
MAX_DELAY="${MAX_DELAY:-137}"
BACKOFF_MIN="${BACKOFF_MIN:-900}"
BACKOFF_MAX="${BACKOFF_MAX:-3600}"
POLL_SEC="${POLL_SEC:-120}"

echo "=== queue_hblazer_full_scrape ===" | tee -a "$LOG_FILE"
echo "start: $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "topical_state: $TOPICAL_STATE" | tee -a "$LOG_FILE"
echo "full_output: $FULL_OUTPUT" | tee -a "$LOG_FILE"
echo "delay_window: ${MIN_DELAY}-${MAX_DELAY}s" | tee -a "$LOG_FILE"
echo "backoff_window: ${BACKOFF_MIN}-${BACKOFF_MAX}s" | tee -a "$LOG_FILE"

pending_count() {
  TOPICAL_STATE_PATH="$TOPICAL_STATE" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
state_path = Path(os.environ["TOPICAL_STATE_PATH"])
if not state_path.exists():
    print(-1)
    raise SystemExit(0)
try:
    obj = json.loads(state_path.read_text(encoding="utf-8", errors="ignore"))
    pending = obj.get("pending_urls", [])
    print(len(pending) if isinstance(pending, list) else -1)
except Exception:
    print(-1)
PY
}

while true; do
  p="$(pending_count)"
  if [[ "$p" == "0" ]]; then
    echo "[queue] topical scrape completed. launching full scrape." | tee -a "$LOG_FILE"
    break
  fi
  echo "[queue] waiting. topical pending=$p next_check=${POLL_SEC}s" | tee -a "$LOG_FILE"
  sleep "$POLL_SEC"
done

cd "$PROJECT_ROOT"
mkdir -p "$FULL_OUTPUT"

echo "[launch] $(date -Iseconds)" | tee -a "$LOG_FILE"
set +e
PYTHONPATH=. "$PYTHON_BIN" scripts/scrape_hblazer_substack.py \
  --all-posts \
  --output-dir "$FULL_OUTPUT" \
  --min-delay-sec "$MIN_DELAY" \
  --max-delay-sec "$MAX_DELAY" \
  --backoff-min-sec "$BACKOFF_MIN" \
  --backoff-max-sec "$BACKOFF_MAX" \
  2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "[exit] code=$EXIT_CODE end=$(date -Iseconds)" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
