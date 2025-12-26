# Phase 7A Diagnostics: Wrong Computation Bottleneck (Books v4)

**Date:** 2025-12-19  
**Implementer:** Codex (GPT-5.2)  
**Scope:** Phase 7A (diagnostics + instrumentation), focused on **wrong_computation** dominance after Phase 6 coverage expansion.

## Executive Summary

Phase 6 proved **coverage is no longer the bottleneck** (23 books, 1,329 artifacts, 38k templates), but accuracy remains stuck because the solver selects **plausible-but-wrong** candidates at very high rates.

Phase 7A adds **source attribution + TTC usage telemetry** to answer a key question:

> Are book-derived candidates being generated and selected, or are generic TTC candidates dominating the final answer?

Result: book-derived candidates are **often available**, but are **rarely selected**.

## Benchmarks (books_v4)

### Baseline (Phase 6B, coverage complete; selection unrefined)
- **MATH:** 4/200 = 2.00%, wrong_computation=122, no_rule_match=15  
- **AMC-AIME:** 1/200 = 0.50%, wrong_computation=79, no_rule_match=34  

### Phase 7A (with instrumentation + artifact/template provenance)
Logs:
- `/tmp/math_phase7_ranked6_books_v4_200_seed123.log`
- `/tmp/amc_aime_phase7_ranked6_books_v4_200_seed123.log`

Results:
- **MATH:** 5/200 = 2.50%, wrong_computation=117, no_rule_match=19  
- **AMC-AIME:** 1/200 = 0.50%, wrong_computation=77, no_rule_match=35  

Interpretation:
- Marginal reductions in wrong_computation (MATH -5, AMC -2), but **selection remains the bottleneck**.

## Key Diagnostic: TTC Source Attribution

Phase 7A introduces a 3-way attribution for test-time compute selections:
- `non_book`: candidate came from generic TTC families / heuristics.
- `book_heuristic`: candidate came from `_generate_book_galaxy_candidates()` but not from ingested artifact/template programs.
- `book`: candidate came from **ingested artifacts/templates** (sourced seed set).

### MATH (200 problems)
From `/tmp/math_phase7_ranked6_books_v4_200_seed123.log`:
- `TTC best_source: {'non_book': 170, 'book': 3, 'book_heuristic': 3}`
- `TTC usage: {'ttc_calls': 176, 'with_book_seed': 116, 'with_book_sourced_seed': 83}`

Key takeaway:
- **83/176 TTC calls had book-sourced seeds available**, but only **3/176** actually used a `book` candidate.

### AMC-AIME (200 problems)
From `/tmp/amc_aime_phase7_ranked6_books_v4_200_seed123.log`:
- `TTC best_source: {'non_book': 153, 'book_heuristic': 5}`
- `TTC usage: {'ttc_calls': 158, 'with_book_seed': 103, 'with_book_sourced_seed': 78}`

Key takeaway:
- **78/158 TTC calls had book-sourced seeds available**, but **0/158** used a `book` candidate.

## Diagnosis

### 1) “Books are present but not winning”
Book galaxies are producing page/template/artifact hits consistently, and the book pipeline frequently produces **sourced** candidates, but those candidates rarely win TTC selection.

This implies one (or more) of:
- Book-sourced programs evaluate to **implausible** numeric results (filtered by plausibility).
- Book-sourced programs evaluate, but score lower than generic TTC candidates.
- Instantiation/binding (numbers → variables) is still too weak to make book programs competitive.

### 2) Wrong computation dominance is primarily selection-quality, not retrieval
We already improved retrieval (variant registry; more books ingested; fewer no_rule_match in some runs), but wrong_computation remains the largest category.

## Phase 7A Code Changes (Instrumentation + Provenance)

### TRM
- `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`
  - `_generate_book_galaxy_candidates()` now returns `seed_sources` (artifact/template provenance).
  - `_test_time_compute()` now records `best_source` as `book|book_heuristic|non_book`.

### Runner
- `scripts/run_sovereign_math_benchmarks.py`
  - Aggregates and prints `TTC best_source` and `TTC usage` summaries per dataset.

## Recommended Phase 7B Next Steps (Quality Refinement)

1. **Make book seeds competitive** (core issue)
   - Improve binding for book templates and artifacts beyond “first N numbers”.
   - Add role/intent constraints (radius/height/legs/hypotenuse; area vs volume; circle vs sphere).

2. **Separate “book heuristic” from “book sourced” in scoring**
   - `_generate_book_galaxy_candidates()` currently mixes:
     - true book-sourced programs (artifacts/templates)
     - generic heuristics (det/gcd/binomial/etc)
   - Continue pushing the solver toward **artifact/template** programs when applicable.

3. **Add function-op candidate families for trig/roots/log**
   - Example: `Compute \cos 60^\circ` remains `no_rule_match` because TTC does not generate `cos()` candidates and/or lacks degree→radian handling.

4. **Targeted failure sampling**
   - Extract 20 wrong_computation cases and annotate:
     - whether `with_book_sourced_seed` was true
     - top 3 artifact/template candidates (and their numeric results)
     - why they lost (implausible vs scored lower)

## Repro Commands

MATH:
```bash
K3D_LOCAL_DIR=/K3D/Knowledge3D.local PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator --disable-retrieval --datasets math \
  --max-problems 200 --shuffle --shuffle-seed 123 --thinking-budget 8 \
  --shadow-readonly --load-all-galaxies \
  --enable-book-galaxies --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4 \
  --book-max-books 64 --book-top-k 5 --verbose \
  2>&1 | tee /tmp/math_phase7_ranked6_books_v4_200_seed123.log
```

AMC-AIME:
```bash
K3D_LOCAL_DIR=/K3D/Knowledge3D.local PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator --disable-retrieval --datasets amc_aime \
  --max-problems 200 --shuffle --shuffle-seed 123 --thinking-budget 8 \
  --shadow-readonly --load-all-galaxies \
  --enable-book-galaxies --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4 \
  --book-max-books 64 --book-top-k 5 --verbose \
  2>&1 | tee /tmp/amc_aime_phase7_ranked6_books_v4_200_seed123.log
```

