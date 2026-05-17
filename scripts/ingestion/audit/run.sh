#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STORAGE_ROOT="/K3D/Knowledge3D.local/galaxies"
OUTPUT_DIR="scripts/ingestion/staging/D1_audit"
REPORT_PATH="TEMP/CODEX_D1_AUDIT_REPORT_04.18.2026.md"

mkdir -p "$OUTPUT_DIR"

python3 scripts/ingestion/audit/galaxy_audit.py \
  --mode scan \
  --storage-root "$STORAGE_ROOT" \
  --output-dir "$OUTPUT_DIR"

python3 scripts/ingestion/audit/galaxy_audit.py \
  --mode report \
  --storage-root "$STORAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --report-path "$REPORT_PATH"
