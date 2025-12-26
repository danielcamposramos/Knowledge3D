# Phase 5 Validation: Math Galaxy Symlink Registry + books_v3 Re-Ingestion

**Date:** December 19, 2025  
**Implementer:** Codex (GPT‑5.2)

## Goal

Validate the **Math Galaxy symlink/variant registry** (“one meaning, many surface forms”) by:
1) Re-ingesting the same 8 PDFs into `books_v3` so indices can include **canonical Math Galaxy keys** (e.g. `\\cos`) in addition to plain `pdftotext` tokens (e.g. `cos`).
2) Re-running the standard MATH + AMC-AIME benchmark configuration against `books_v3`.

This work intentionally avoids any ad-hoc “LaTeX normalization” in the solver hot path; variant bridging is performed via **Galaxy-owned** mapping (`variants_for`).

## What Changed (Phase 5)

- `knowledge3d/training/arc_agi/math_symbol_galaxy.py`
  - Added bidirectional variant registry (`variant → canonical`, `canonical → variants`) and `variants_for()`.
- `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`
  - Removed query-string “latex loosening”; retrieval now expands variants via Math Galaxy registry.
- `knowledge3d/training/math_benchmarks/book_galaxy_ingestion.py`
  - During ingestion indexing, also indexes canonical Math Galaxy symbols (e.g. `\\cos`) when a token resolves to a math meaning.

## books_v3 Re-Ingestion

Output root:
- `/K3D/Knowledge3D.local/galaxies/books_v3/`

Ingested (same book_ids as `books_v2`):
- `la_done_right`, `advanced_calculus`, `dmoi3`, `transition_v104`
- `areavol`, `numbersets`, `physquantities`, `mathgems`

Counts (matched the prior v2 totals):
- `la_done_right`: pages=353 templates=924 artifacts=43
- `advanced_calculus`: pages=593 templates=5305 artifacts=163
- `dmoi3`: pages=413 templates=1661 artifacts=60
- `transition_v104`: pages=291 templates=2120 artifacts=235
- `areavol`: pages=2 templates=3 artifacts=0
- `numbersets`: pages=2 templates=9 artifacts=0
- `physquantities`: pages=2 templates=32 artifacts=0
- `mathgems`: pages=2 templates=0 artifacts=0

**Index sanity check (new in v3):** `token_index.json` now contains canonical Math Galaxy keys like `\\cos`, `\\sin`, `\\sqrt`, `\\pi` alongside plain forms like `cos`, `sin`.

## Benchmark Runs (books_v3)

Common flags (same as Phase 4, except book root):
- `--use-trm-navigator`
- `--disable-retrieval`
- `--thinking-budget 8`
- `--shadow-readonly`
- `--load-all-galaxies`
- `--enable-book-galaxies`
- `--book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v3`
- `--book-max-books 16`
- `--book-top-k 5`
- `--max-problems 200 --shuffle --shuffle-seed 123`

Logs:
- MATH: `/tmp/math_phase5_books_v3_200_seed123.log`
- AMC-AIME: `/tmp/amc_aime_phase5_books_v3_200_seed123.log`

### MATH (200 problems)

- Accuracy: **3/200 = 1.50%** (Phase 4 `books_v2`: 5/200 = 2.50%)
- Failure categories:
  - `no_rule_match`: **11** (Phase 4 `books_v2`: 17)
  - `wrong_computation`: **125** (Phase 4 `books_v2`: 120)
  - `multi_step_needed`: **13** (Phase 4 `books_v2`: 12)
  - `word_problem`: **13** (Phase 4 `books_v2`: 13)
  - `algebra_needed`: **35** (Phase 4 `books_v2`: 33)

### AMC-AIME (200 problems)

- Accuracy: **1/200 = 0.50%** (unchanged vs Phase 4 `books_v2`)
- Failure categories:
  - `no_rule_match`: **38** (Phase 4 `books_v2`: 43)
  - `wrong_computation`: **76** (Phase 4 `books_v2`: 75)
  - `multi_step_needed`: **27** (Phase 4 `books_v2`: 26)
  - `word_problem`: **31** (Phase 4 `books_v2`: 29)
  - `algebra_needed`: **27** (Phase 4 `books_v2`: 26)

## Interpretation

1. **Lexical mismatch is architecturally resolved at the data-structure level**:
   - `books_v3` indices include canonical Math Galaxy symbol keys (e.g. `\\cos`) and the reader expands variants via the registry.
2. **Coverage improved** (lower `no_rule_match` on both datasets), which suggests book-driven candidate generation is being exercised more often.
3. **Accuracy did not improve yet** under this benchmark configuration:
   - The extra coverage currently manifests mostly as additional `wrong_computation`, not additional correct solves.

## Likely Next Work (Phase 6: “Accuracy Refinement”)

This is now clearly **no longer a lexical plumbing issue**; it’s a **candidate quality + selection** issue:
- Add instrumentation in `TRMGalaxyReader` to report:
  - book page hits, template hits, artifact hits per problem,
  - how many book-derived RPN candidates are generated and actually executed,
  - which (if any) artifact/template was responsible for a correct solve.
- Tighten artifact candidate ranking:
  - prioritize theorem/definition over example/exercise,
  - require at least one condition to be positively matched (not “unknown/ignored”) for high-risk artifacts,
  - prefer candidates that bind variables from `symbol_bindings` roles (leg/hypotenuse, etc.).

