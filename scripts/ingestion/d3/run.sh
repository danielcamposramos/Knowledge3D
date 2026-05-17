#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$repo_root"

if [[ "${1:-}" == "--recover-only" ]]; then
  shift
  python3 scripts/ingestion/d3/recover_b6.py "$@"
else
  python3 scripts/ingestion/d3/galaxy_d3.py "$@"
fi
