# Phase 4 Validation: MATH + AMC-AIME (Book Artifacts + Condition Gate)

**Date:** December 18, 2025  
**Implementer:** Codex (GPT‑5.2)

## Run Configuration

Common flags:
- `--use-trm-navigator`
- `--disable-retrieval`
- `--thinking-budget 8`
- `--shadow-readonly`
- `--load-all-galaxies`
- `--enable-book-galaxies`
- `--book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v2`
- `--book-max-books 16`
- `--book-top-k 5`
- `--max-problems 200 --shuffle --shuffle-seed 123`

Logs:
- MATH: `/tmp/math_phase4_v2_artifacts_200_seed123.log`
- AMC-AIME: `/tmp/amc_aime_phase4_v2_artifacts_200_seed123.log`

Baseline reference:
- `TEMP/CODEX_MULTI_BENCHMARK_BASELINE_12.18.2025.md`

## Results

### MATH (200 problems)

- Accuracy: **5/200 = 2.50%**
- Failure categories:
  - `no_rule_match`: **17** (baseline 24, **-7**)
  - `wrong_computation`: **120** (baseline 114, **+6**)
  - `multi_step_needed`: **12** (baseline 12, **±0**)
  - `word_problem`: **13** (baseline 12, **+1**)
  - `algebra_needed`: **33** (baseline 33, **±0**)

### AMC-AIME (200 problems)

- Accuracy: **1/200 = 0.50%**
- Failure categories:
  - `no_rule_match`: **43** (baseline 44, **-1**)
  - `wrong_computation`: **75** (baseline 71, **+4**)
  - `multi_step_needed`: **26** (baseline 27, **-1**)
  - `word_problem`: **29** (baseline 30, **-1**)
  - `algebra_needed`: **26** (baseline 27, **-1**)

## Interpretation

1. **Condition-gated artifact selection is implemented and unit-tested**, but **benchmark-level accuracy did not improve yet** on MATH/AMC-AIME with this configuration.
2. We see a **small reduction in `no_rule_match`** (especially MATH: 24→17), suggesting extra candidate coverage (TTC families + book lookups) is being exercised.
3. **`wrong_computation` increased slightly** on both datasets, which indicates the current candidate pool is still dominated by generic/incorrect compositions (i.e., we are attempting “something” more often, but not the right thing).

## Likely Root Causes (Why Phase 3 Didn’t Move Accuracy Yet)

- **Books are ingested from `pdftotext`**, while **MATH/AMC prompts are LaTeX-heavy**; lexical overlap is weak, so book hits often do not retrieve the intended theorem blocks.
- **Articulated artifacts in `books_v2` are “semantic blocks only” (501 total)**, but many blocks are *examples/exercises* with incidental equations. Without stronger filtering + binding, these are risky to apply.
- Current integration uses book candidates primarily inside **test-time compute (TTC)**; the **primary template selection path** is still driven by the generic TRM/heuristic templates, so the “apply only when conditions hold” effect is limited.

## Next Actions (Recommended)

1. **Tighten artifact ranking/filtering**:
   - Prefer `theorem/lemma/proposition/corollary/definition` over `example/exercise`.
   - Require at least one *evaluated* condition to match (not just “unknown/ignored”).
   - Require executable RPN shape (stack-validity), not just token-validity.
2. **Improve book query normalization for LaTeX prompts**:
   - Normalize `\\cos`, `\\sin`, `\\sqrt`, `\\frac`, braces, and math punctuation into tokens that match `pdftotext` output.
3. **Move book artifacts earlier in the solve path**:
   - Attempt top artifact-derived candidates *before* generic template application (or use them to gate template selection).
4. **Add instrumentation**:
   - Log book hit counts (`hits/template_hits/artifact_hits`) and “artifact used” attribution so we can measure whether book artifacts are actually being used in benchmark runs.

