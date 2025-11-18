#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
exec "$ROOT_DIR/scripts/k3d_env.sh" run python -m knowledge3d.cranium.tests.benchmark_trm_ternary_speedup
