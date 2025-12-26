# Phase 6A: Comprehensive Book Ingestion (books_v4) + Benchmark Validation

**Date:** December 19, 2025  
**Implementer:** Codex (GPT‑5.2)

## Goal

Expand Book Galaxy coverage beyond the initial 8/38 PDFs by ingesting the Phase 6A priority set and validating MATH + AMC-AIME again, while keeping the solver hot-path sovereign (no ad-hoc LaTeX normalization; variant bridging happens via Math Galaxy symlink registry).

## Output

New comprehensive root (13 books total):
- `/K3D/Knowledge3D.local/galaxies/books_v4/`

Per-book ingestion logs:
- `/tmp/phase6a_books_v4_<book_id>.log`

## What Was Ingested

### Baseline 8 (per Phase 6 plan)

| book_id | source PDF |
|---|---|
| `la_done_right` | `Linear.Algebra.Done.Right.pdf` |
| `advanced_calculus` | `advcalc.pdf` |
| `dmoi3` | `dmoi3-tablet.pdf` |
| `transition_v104` | `Transition_v104.pdf` |
| `areavol` | `BasicMath/AreaVol.pdf` |
| `numbersets` | `BasicMath/NumberSets.pdf` |
| `physquantities` | `BasicMath/PhysQuantities.pdf` |
| `mathgems` | `BasicMath/MathGems.pdf` |

### Phase 6A additions (5)

| book_id | source PDF |
|---|---|
| `multivariable_calc` | `Multivariable Calculus 7th Edition By James Stewart.pdf` |
| `advanced_calc_1_2` | `ADVANCED CALCULUS I and II.pdf` |
| `advanced_calc_alt` | `Advanced_Calculus.pdf` |
| `numerical_analysis` | `Handbook Of Numerical Analysis - Special Volume - Foundations Of Computational Mathematics.pdf` |
| `shortestshortcut` | `BasicMath/ShortestShortcut.pdf` |

## Size / Counts Summary

Totals across `books_v4`:
- Books: **13**
- Pages: **3367**
- Templates (`templates.jsonl`): **22986**
- Artifacts (`artifacts.jsonl`): **928**

Selected per-book counts:
- `multivariable_calc`: pages=609 templates=1781 artifacts=68
- `advanced_calc_1_2`: pages=309 templates=3759 artifacts=175
- `advanced_calc_alt`: pages=593 templates=5305 artifacts=163
- `numerical_analysis`: pages=473 templates=3540 artifacts=9
- `shortestshortcut`: pages=9 templates=93 artifacts=0

Index sanity checks:
- The new books include canonical Math Galaxy tokens in `token_index.json` such as `\\cos`, `\\sin`, `\\sqrt`, `\\pi` (alongside plain forms like `cos`, `sin`), confirming canonical symbol indexing is active.

Note:
- `ADVANCED CALCULUS I and II.pdf` appears to be content-identical (or extremely close) to `advcalc.pdf` based on identical page/template/artifact counts; this may be worth deduplicating later to reduce “duplicate evidence” effects.

## Benchmark Runs (books_v4)

Common flags:
- `--use-trm-navigator --disable-retrieval --thinking-budget 8 --shadow-readonly --load-all-galaxies`
- `--enable-book-galaxies --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4`
- `--book-max-books 32 --book-top-k 5`
- `--max-problems 200 --shuffle --shuffle-seed 123`

Logs:
- MATH: `/tmp/math_phase6a_books_v4_200_seed123.log`
- AMC-AIME: `/tmp/amc_aime_phase6a_books_v4_200_seed123.log`

### MATH (200)

- Accuracy: **4/200 = 2.00%**
- Failure categories: `{'no_rule_match': 13, 'wrong_computation': 123, 'multi_step_needed': 13, 'word_problem': 13, 'algebra_needed': 34, 'unknown': 0}`

### AMC-AIME (200)

- Accuracy: **1/200 = 0.50%**
- Failure categories: `{'no_rule_match': 38, 'wrong_computation': 75, 'multi_step_needed': 27, 'word_problem': 32, 'algebra_needed': 27, 'unknown': 0}`

## Interpretation

1. **Coverage is materially higher** (13 books, 928 artifacts, 22,986 templates), and canonical symbol indexing is present for the new books.
2. **Accuracy remains bottlenecked by candidate quality / selection**, not lexical mismatch plumbing:
   - MATH improved vs Phase 5 (`books_v3`: 1.5% → `books_v4`: 2.0%), but still far below target.
   - AMC-AIME unchanged at 0.5%.
3. **Next likely leverage** is Phase 6B (more books) *plus* ranking/selection refinement so that additional matches reduce `wrong_computation` rather than increasing it.

