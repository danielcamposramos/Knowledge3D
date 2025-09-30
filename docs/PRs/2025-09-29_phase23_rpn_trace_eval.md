Title: Phase 23 — RPN Trace, ARC/HLE Teacher Scoring, Math Bench Auto‑Discovery, and Wiki Sweep

Summary
- Add optional RPN trace blocks in fused‑head math paths (program/infix/direct) gated by `K3D_RPN_TRACE=1`.
- Harden Phase 23 ARC/HLE tester; add `--teacher` lightweight feedback scoring and support unlimited `--limit 0`.
- Extend math bench to auto‑discover locally cached HF repos and run multiple suites (`--auto`, `--repos`, `--limit`).
- Add a Wikipedia sweep evaluator to sanity‑check non‑math routing and summaries.
- Fix fused‑head fallback path so memory/tablet lookup and neural fallback always return an answer (no `None`).

New (2025‑09‑29 PM)
- In‑core RPN Policy Head: tiny GRU that generates RPN tokens; evaluation remains PTX‑only. Checkpoint: `viewer/public/house/house_rpn_policy.pt`. Gate with `K3D_ENABLE_RPN_POLICY=1`.
- RPN Policy Trainer: `knowledge3d/tools/phase25/rpn_policy_trainer.py` (teacher‑forced tokens + quick in‑process eval). Trained 5 epochs on 5k sequences (corpus now 7k lines).
- Omni Bench Evaluator: `knowledge3d/tools/omni_bench_evaluator.py` scans the local HF cache and evaluates many datasets (nonempty/strict/soft metrics). Writes `docs/benchmarks/omni_bench_report.json`.
- NVRTC/Driver stabilization: lazy init of PTX subsystems + inter‑process compile locks; disabled cuDNN in fused head for stability on this driver.
- ARC JSON guard: avoid routing ARC/HLE JSON through RPN program/infix; require explicit assignment for program path.

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

One‑process evaluation (avoids segfaults on short‑lived CLI):
- Interactive shell (no bashrc changes required):
  - `source /home/daniel/miniforge/bin/activate k3d-cranium`
  - `cd /K3D/Knowledge3D && export PYTHONPATH=. K3D_EVAL_MINIMAL=1 K3D_DISABLE_TEXT_MODALITY=1 K3D_ENABLE_RPN_POLICY=1`
  - `python -m knowledge3d.tools.run_evals_serial`
  - Reports: `docs/benchmarks/math_bench_report.json`, `docs/benchmarks/omni_bench_report.json`

Notes
- No CPU fallbacks added; fused head remains GPU‑first and tablet/memory‑first in line with Phase A/B design.
- Logs and large artifacts remain uncommitted per memory policy.
