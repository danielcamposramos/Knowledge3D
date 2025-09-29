Title: Phase 23 — RPN Trace, ARC/HLE Teacher Scoring, Math Bench Auto‑Discovery, and Wiki Sweep

Summary
- Add optional RPN trace blocks in fused‑head math paths (program/infix/direct) gated by `K3D_RPN_TRACE=1`.
- Harden Phase 23 ARC/HLE tester; add `--teacher` lightweight feedback scoring and support unlimited `--limit 0`.
- Extend math bench to auto‑discover locally cached HF repos and run multiple suites (`--auto`, `--repos`, `--limit`).
- Add a Wikipedia sweep evaluator to sanity‑check non‑math routing and summaries.
- Fix fused‑head fallback path so memory/tablet lookup and neural fallback always return an answer (no `None`).

Changes
- knowledge3d/cranium/fused_head.py
  - Add RPN trace output blocks (trace + register map for programs).
  - Fix predict() control flow so memory → language → summary → neural fallback paths always execute.
  - Summarize inline text for prompts like `Summarize: <text>` when no corpus entry exists.
- knowledge3d/skills/infix_to_rpn.py
  - Add `program_to_rpn_with_trace()` to return compiled tokens and register map.
- knowledge3d/tools/phase23/arc_hle_tester.py
  - Add `--teacher` option; guard against empty predictions; support unlimited `--limit 0`.
- knowledge3d/tools/phase25/math_bench_evaluator.py
  - Add `--auto`, `--list`, and `--repos` flags; discover repos from the local HF cache.
- knowledge3d/tools/wiki_sweep_evaluator.py
  - New tool to run large “Summarize” sweeps over a local AI‑topics corpus; reports non‑math routing quality.

How to Run
- ARC/HLE (unlimited + teacher scoring):
  `conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.phase23.arc_hle_tester --limit 0 --teacher`
- Math benches (auto‑discover cached repos):
  `conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.phase25.math_bench_evaluator --auto --limit 50`
  List discovered repos: add `--list`.
- Wikipedia sweep (unlimited):
  `conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.wiki_sweep_evaluator --max-lines 0 --summarize`
- Optional RPN trace in answers:
  `export K3D_RPN_TRACE=1 K3D_RPN_ROUND_MODE=half_even K3D_RATIONAL_OUTPUT=1`

Notes
- No CPU fallbacks added; fused head remains GPU‑first and tablet/memory‑first in line with Phase A/B design.
- Logs and large artifacts remain uncommitted per memory policy.

