#!/usr/bin/env bash
# ------------------------------------------------------------------
# K3D sovereignty preflight — enforces the 2026-04-18 Absolute
# Sovereignty Purge. Blocks any commit that reintroduces a banned
# bulk-library import on the hot path.
#
# Hot path:  knowledge3d/cranium/**
#            knowledge3d/knowledgeverse/**
#            knowledge3d/bridge/**           (Tablet live-loop — Phase 7.5)
#
# Exempt subtrees (per TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md §4.1):
#   knowledge3d/cranium/tests/**    — test harness may use numpy for fixtures
#   knowledge3d/cranium/ocr/**      — OCR subsystem is an ingestion adapter
#
# (Phase 7.6 — 2026-04-18) Bridge carve-out lifted:
#   knowledge3d/bridge/live_server.py was purged of all 17 numpy+torch
#   sites per TEMP/CLAUDE_PHASE_7_6_LIVE_SERVER_PURGE_SPEC_04.18.2026.md
#   and is now fully sovereign. headless_tablet.py + replay_builder.py
#   remain sovereign.
#
# Banned names:  numpy  cupy  scipy  sympy  torch
#
# Invoke directly:
#   bash scripts/sovereignty_preflight.sh
#
# Wired as .git/hooks/pre-commit — see tail of this file for the
# one-liner hook that delegates here.
# ------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

BANNED_RE='^[[:space:]]*(import|from)[[:space:]]+(numpy|cupy|scipy|sympy|torch)([[:space:]]|$|\.)'

# --- Collect candidate files ---------------------------------------
# In pre-commit mode: only staged .py files.
# In manual mode:    full hot-path tree.
STAGED="$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)"

if [[ -n "$STAGED" ]]; then
  MODE="staged"
  FILES="$(printf '%s\n' "$STAGED" | \
    grep -E '^knowledge3d/(cranium|knowledgeverse|bridge)/.*\.py$' | \
    grep -Ev '^knowledge3d/cranium/(tests|ocr)/' || true)"
else
  MODE="full-tree"
  FILES="$(find knowledge3d/cranium knowledge3d/knowledgeverse knowledge3d/bridge \
    -type f -name '*.py' \
    -not -path 'knowledge3d/cranium/tests/*' \
    -not -path 'knowledge3d/cranium/ocr/*' 2>/dev/null || true)"
fi

if [[ -z "$FILES" ]]; then
  echo "[preflight] $MODE: no hot-path Python files to check — OK"
  exit 0
fi

# --- Scan -----------------------------------------------------------
HITS="$(printf '%s\n' "$FILES" | xargs -r grep -HnE "$BANNED_RE" 2>/dev/null || true)"

if [[ -z "$HITS" ]]; then
  FILE_COUNT=$(printf '%s\n' "$FILES" | wc -l)
  echo "[preflight] $MODE: $FILE_COUNT hot-path files clean — sovereignty intact"
  exit 0
fi

# --- Report + block -------------------------------------------------
cat <<EOF >&2

========================================================================
  K3D SOVEREIGNTY VIOLATION — commit BLOCKED
========================================================================

The Absolute Sovereignty Purge (2026-04-18) bans the following names on
the hot path:  numpy  cupy  scipy  sympy  torch

The following import(s) violate that rule:

EOF

printf '%s\n' "$HITS" >&2

cat <<EOF >&2

Fix options (per TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md §5.2):

  (a) Replace with a PTX kernel call through sovereign/loader.py.
  (b) Move the file to knowledge3d/cranium/tests/ if it is a test harness.
  (c) raise NotImplementedError with a spec pointer if the sovereign
      successor is designed but not yet wired.

Do NOT wrap in try/except ImportError. No fallbacks. Ever.
========================================================================
EOF

exit 1
