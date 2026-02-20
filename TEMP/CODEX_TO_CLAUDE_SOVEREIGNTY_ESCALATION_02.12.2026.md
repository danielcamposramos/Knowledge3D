# Codex → Claude: Sovereignty Escalation Context

Date: 2026-02-12

## Mandatory Context (Daniel complaint to preserve)
"For the last 2 months, I've been enforcing docs/vocabulary specs (Galaxy-first, sovereign, PTX+RPN) and you guys keep delivering Python fallbacks. Every time I ask to fix it, you add MORE Python on top. Codex is too conservative ('I don't want to break working code') even though 0% accuracy means it ISN'T working. We need radical architecture-first rewrites, not incremental Python patches. Remove fallbacks entirely - if it breaks, that's diagnostic (shows what's missing in Galaxy), not failure."

## What is now done
- ARC adapter cut to PTX-only / fail-fast (legacy + CPU fallback removed).
- TRM query local Python scoring loop removed; delegates to GalaxyManager query.
- GalaxyManager query now fail-fast by default (`K3D_REQUIRE_PTX_QUERY=true`) until PTX query kernel exists.
- LHE `eval` removed.
- Regex-based math routing cue removed from specialist router.
- Sovereignty test added (`tests/test_hot_path_sovereignty.py`).

## Remaining blocker
- PTX query kernel is still missing for `GalaxyManager.query(...)`.
- Runtime benchmarks are intentionally blocked in strict mode until GPU query implementation is added.

## Next strict step
Implement GPU query kernel path (no CPU O(n) fallback), then run bounded diagnostics to populate missing Grammar/Math templates based on explicit missing signals.

