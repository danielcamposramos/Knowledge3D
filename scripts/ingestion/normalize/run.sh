#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stage_dir="$repo_root/scripts/ingestion/staging/D2_normalize"
normalized_dir="$stage_dir/normalized"
reaudit_dir="$stage_dir/re_audit"

cd "$repo_root"

python3 scripts/ingestion/normalize/galaxy_normalize.py --mode normalize

python3 scripts/ingestion/audit/galaxy_audit.py \
  --storage-root "$normalized_dir" \
  --output-dir "$reaudit_dir" \
  --report-path "$reaudit_dir/RE_AUDIT_REPORT.md" \
  --mode scan

python3 scripts/ingestion/audit/galaxy_audit.py \
  --storage-root "$normalized_dir" \
  --output-dir "$reaudit_dir" \
  --report-path "$reaudit_dir/RE_AUDIT_REPORT.md" \
  --mode report

python3 scripts/ingestion/normalize/galaxy_normalize.py --mode report
