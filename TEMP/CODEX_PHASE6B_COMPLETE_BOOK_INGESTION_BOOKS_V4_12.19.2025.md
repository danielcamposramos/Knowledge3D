# Phase 6B: Complete Book Galaxy Ingestion (books_v4) + Validation

**Date:** December 19, 2025  
**Implementer:** Codex (GPT‑5.2)  
**Directive:** `TEMP/CODEX_DIRECTIVE_COMPLETE_BOOK_INGESTION_12.19.2025.md`

## Objective

Complete “coverage-first” ingestion by adding the remaining 10 core math PDFs (skipping Financial Math) into the existing `books_v4` Book Galaxy root, then re-run MATH + AMC-AIME benchmarks under the standard evaluation flags.

## Output Root

- `/K3D/Knowledge3D.local/galaxies/books_v4/`

## Phase 6B Books Ingested (10)

| book_id | domain | source PDF |
|---|---|---|
| `wrede_calculus` | `calculus` | `Advanced-Calculus-Robert-Wrede.pdf` |
| `hildebrand` | `advanced_mathematics` | `hildebrand.pdf` |
| `math_2f05` | `university_course` | `MATH 2F05.pdf` |
| `math_for_programmers` | `applied_mathematics` | `Manning.Math.for.Programmers.2020.11.pdf` |
| `algebraic_topology` | `topology` | `Undergraduate Texts in Mathematics - Basic concepts of algebraic topology - Croom.pdf` |
| `rpn_intermediate` | `rpn_methods` | `3.3. Reverse Polish - Intermediate.pdf` |
| `rpn_method` | `rpn_methods` | `ReversePolishNotatonMethod.pdf` |
| `orland_math_prog` | `applied_mathematics` | `Orland_MfP_MEAP_V02_ch1.pdf` |
| `adv_math_programming` | `optimization` | `advmathprog.pdf` |
| `stavely_python` | `programming` | `Stavely_python_ebook.pdf` |

Per-book ingestion logs:
- `/tmp/phase6b_books_v4_<book_id>.log`

## Final books_v4 Totals (Phase 6A + 6B)

Computed by summing all `metadata.json` under `books_v4`:
- Books: **23**
- Pages: **5990**
- Templates: **38309**
- Artifacts: **1329**

Notes:
- Some books legitimately produce no templates (e.g. poster-style or narrative sources). When `template_count=0`, `templates.jsonl` / `template_index.json` may be absent (observed for `mathgems`, `rpn_intermediate`), which is expected behavior in the current ingester.
- Canonical Math Galaxy token indexing is active across the newly ingested set (e.g. `\\cos`, `\\sin`, `\\sqrt`, `\\pi` appear in `token_index.json` alongside plain forms where applicable).

## Benchmark Validation (books_v4 complete)

Common flags:
- `--use-trm-navigator --disable-retrieval --thinking-budget 8 --shadow-readonly --load-all-galaxies`
- `--enable-book-galaxies --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4`
- `--book-max-books 64 --book-top-k 5`
- `--max-problems 200 --shuffle --shuffle-seed 123`

Logs:
- MATH: `/tmp/math_phase6b_books_v4_complete_200_seed123.log`
- AMC-AIME: `/tmp/amc_aime_phase6b_books_v4_complete_200_seed123.log`

### MATH (200)

- Accuracy: **4/200 = 2.00%**
- Failure categories: `{'no_rule_match': 15, 'wrong_computation': 122, 'multi_step_needed': 12, 'word_problem': 13, 'algebra_needed': 34, 'unknown': 0}`

### AMC-AIME (200)

- Accuracy: **1/200 = 0.50%**
- Failure categories: `{'no_rule_match': 34, 'wrong_computation': 79, 'multi_step_needed': 27, 'word_problem': 33, 'algebra_needed': 26, 'unknown': 0}`

## Outcome vs Phase 6B Success Criteria

Directive thresholds:
- MATH ≥ 3.0% (6/200) → **NOT met** (2.0%)
- AMC-AIME ≥ 1.5% (3/200) → **NOT met** (0.5%)

However, coverage is now substantially higher (23 books / 1329 artifacts), so remaining work is primarily **quality + selection** (Phase 7), not data availability.

