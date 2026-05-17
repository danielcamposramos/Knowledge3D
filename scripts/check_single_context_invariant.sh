#!/usr/bin/env bash
# Enforces K3D single-CUDA-context invariant:
# loader.py is the ONLY file that may call cuCtxCreate / cuDevicePrimaryCtxRetain / cuInit.
# See TEMP/CLAUDE_SINGLE_CONTEXT_LIVING_AI_SPEC_04.18.2026.md

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VIOLATIONS=$(grep -rn -E 'cuCtxCreate|cuDevicePrimaryCtxRetain|cuInit\(' \
    --include='*.py' --include='*.cu' --include='*.cpp' --include='*.c' \
    --exclude-dir=.git --exclude-dir=node_modules --exclude-dir='*.egg-info' \
    . 2>/dev/null | \
    grep -v 'knowledge3d/cranium/sovereign/loader.py' | \
    grep -v '\.md:' | \
    grep -v '# ALLOW: ' || true)

if [ -n "$VIOLATIONS" ]; then
    echo "ERROR: Single-context invariant violated."
    echo "Only knowledge3d/cranium/sovereign/loader.py may create/retain CUDA contexts."
    echo ""
    echo "Violations:"
    echo "$VIOLATIONS"
    echo ""
    echo "Fix by migrating to the shared-context pattern — see"
    echo "  knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py:101-122"
    echo "Reference spec: TEMP/CLAUDE_SINGLE_CONTEXT_LIVING_AI_SPEC_04.18.2026.md"
    exit 1
fi

echo "Single-context invariant: CLEAN"
