# PR: F64 PTX RPN, Parser + Programs, RPN‑only Math in Fused Head

## Summary
- Switch PTX RPN engine to double precision by default (F64), F32 as fallback.
- Macro‑based single kernel path (no code duplication) with mapped intrinsics.
- Add convenience ops: floor, ceil, mod, round, round_he (half‑even), clamp, fact, gcd, lcm, nCr, nPr.
- Add per‑instance scalar registers with `store`/`load` (16 regs) to support simple RPN “programs”.
- Parser (Shunting‑Yard) extended with LaTeX normalization (\\frac, \\sqrt, \\log_b, \\binom, \\lfloor/\\rfloor, \\lceil/\\rceil), Unicode floor/ceil glyphs, absolute value bars, factorial postfix, aliases (C/P→nCr/nPr).
- Fused head executes Program→RPN→PTX and Infix→RPN→PTX before any learned numeric path (which is off by default). Rational formatting available for exact fractions.
- Added focused evaluator for RPN‑only math (expressions and programs).

## Files Changed
- `knowledge3d/cranium/phase10/modular_rpn_engine.py`
  - F64 default kernel via `-DUSE_DOUBLE` with macro intrinsics.
  - Added ops: floor, ceil, mod, round, round_he, clamp, fact, gcd, lcm, nCr, nPr, store, load.
  - Scalar division fix; per‑instance `regs[16]` added; host dtype/ctypes synced with precision.
- `knowledge3d/skills/infix_to_rpn.py`
  - Shunting‑Yard infix→RPN; LaTeX normalization; glyphs; abs bars; factorial postfix.
  - Program compiler `program_to_rpn` for simple assignments and final expression.
  - Rounding mode mapping via `K3D_RPN_ROUND_MODE` (half_up|half_even).
- `knowledge3d/cranium/fused_head.py`
  - RPN‑first math path: Program→RPN→PTX, then Infix→RPN→PTX; learned head off by default.
  - Rational formatter (`K3D_RATIONAL_OUTPUT=1`) when result is an exact small fraction.
- `knowledge3d/tools/phase25/rpn_focus_evaluator.py`
  - Focused tests for pure RPN math (expressions + programs). Report saved to `docs/benchmarks/rpn_focus_report.json`.

## Results
- AIME (pre‑existing fused‑only run): 30/30 correct (report unchanged).
- RPN‑only Focused Evaluation (F64, learned head off):
  - Expressions: 8/10 (0.80)
  - Programs: 2/2 (1.00)
  - Report: `docs/benchmarks/rpn_focus_report.json`

## Env Flags
- `K3D_RPN_USE_DOUBLE` (default 1): set `0` for F32.
- `K3D_RPN_ROUND_MODE` (default `half_up`; supports `half_even`).
- `K3D_RATIONAL_OUTPUT` (default `1`): set `0` to disable fraction display.
- `K3D_ENABLE_MATH_HEAD` (default `0`): enable to allow learned numeric head.

## Next Work
- Extend parser with more aliases (e.g., ⌊⌋/⌈⌉ nesting, multi‑expression joins like “sum of … and …”).
- Add more program patterns (chained definitions, vector ops) and broaden evaluation bank.
- Optional rational arithmetic mode at compute time (exact rationals) for targeted tasks.

