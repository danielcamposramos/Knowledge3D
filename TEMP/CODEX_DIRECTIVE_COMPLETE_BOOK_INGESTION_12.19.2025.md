# CODEX DIRECTIVE: Complete Book Galaxy Ingestion

**Date:** December 19, 2025
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** HIGH
**Context:** Phase 6B - Complete comprehensive book coverage before quality refinement

---

## Objective

Ingest ALL remaining math PDFs from the Advanced Maths library into `/K3D/Knowledge3D.local/galaxies/books_v4/` to achieve **comprehensive mathematical knowledge coverage**.

**Rationale:** Coverage must be complete before quality refinement. We need the full "math grammar with several representations united by meaning" before we can optimize candidate selection.

---

## Current State (Phase 6A Complete)

**Ingested:** 13 books in books_v4
- Baseline 8: la_done_right, advanced_calculus, dmoi3, transition_v104, areavol, numbersets, physquantities, mathgems
- Phase 6A (+5): multivariable_calc, advanced_calc_1_2, advanced_calc_alt, numerical_analysis, shortestshortcut

**Statistics:**
- Pages: 3,367
- Templates: 22,986
- Artifacts: 928

---

## Books to Ingest (Phase 6B: Remaining Core Math)

### Priority 1: Core Math Textbooks (5 books)

1. **Advanced-Calculus-Robert-Wrede.pdf**
   - book_id: `wrede_calculus`
   - domain: `calculus`
   - Notes: Advanced calculus topics, rigorous analysis

2. **hildebrand.pdf**
   - book_id: `hildebrand`
   - domain: `advanced_mathematics`
   - Notes: Classic advanced math reference

3. **MATH 2F05.pdf**
   - book_id: `math_2f05`
   - domain: `university_course`
   - Notes: University-level math course material

4. **Manning.Math.for.Programmers.2020.11.pdf**
   - book_id: `math_for_programmers`
   - domain: `applied_mathematics`
   - Notes: Practical math for programming/problem-solving

5. **Undergraduate Texts in Mathematics - Basic concepts of algebraic topology - Croom.pdf**
   - book_id: `algebraic_topology`
   - domain: `topology`
   - Notes: Advanced topology (may have limited MATH dataset overlap, but comprehensive coverage goal)

### Priority 2: RPN/Procedural Pattern Books (3 books)

6. **3.3. Reverse Polish - Intermediate.pdf**
   - book_id: `rpn_intermediate`
   - domain: `rpn_methods`
   - Notes: RPN patterns, procedural thinking - may help with template generation

7. **ReversePolishNotatonMethod.pdf**
   - book_id: `rpn_method`
   - domain: `rpn_methods`
   - Notes: RPN methodology - procedural pattern mining

8. **Orland_MfP_MEAP_V02_ch1.pdf**
   - book_id: `orland_math_prog`
   - domain: `applied_mathematics`
   - Notes: Math for Programmers excerpt (may overlap with #4, but include for completeness)

### Priority 3: Additional References (2 books)

9. **advmathprog.pdf**
   - book_id: `adv_math_programming`
   - domain: `optimization`
   - Notes: Mathematical programming, optimization, linear programming

10. **Stavely_python_ebook.pdf**
    - book_id: `stavely_python`
    - domain: `programming`
    - Notes: Python programming (lower relevance, but may have computational pattern examples)

### SKIP: Financial Math (14 books)
**Rationale:** Not relevant for MATH/AMC-AIME datasets. Reserve for future GSM8K enhancement.

**Files to skip:**
- All PDFs in `Financial Math/` subdirectory
- Exception: If any contain general mathematical methods (unlikely), flag for review

---

## Ingestion Specification

### Command Template

```bash
# For each book above, run:
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  -c "
from pathlib import Path
from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester

pdf_path = '<FULL_PATH_TO_PDF>'
book_id = '<BOOK_ID_FROM_TABLE_ABOVE>'
title = '<DESCRIPTIVE_TITLE>'
domain = '<DOMAIN_FROM_TABLE_ABOVE>'
output_dir = Path('/K3D/Knowledge3D.local/galaxies/books_v4')

ingester = BookGalaxyIngester(local_dir=output_dir)

# Convert PDF to JSON pages first
json_path = ingester.pdf_to_json_pages(
    pdf_path=pdf_path,
    title=title
)

# Ingest JSON pages into Book Galaxy
result_dir = ingester.ingest_json_pages(
    json_path=json_path,
    title=title,
    book_id=book_id,
    domain=domain
)

print(f'Ingested {book_id}: {result_dir}')
"
```

### Validation Per Book

After each book ingestion, verify:
1. **Directory created:** `/K3D/Knowledge3D.local/galaxies/books_v4/<book_id>/`
2. **Files present:**
   - `pages.jsonl` (page-level content)
   - `templates.jsonl` (extracted formulas)
   - `artifacts.jsonl` (structured knowledge - theorems/definitions)
   - `token_index.json` (includes canonical Math Galaxy symbols)
   - `template_index.json`
   - `artifact_index.json`
3. **Canonical symbols indexed:** Check `token_index.json` contains `\cos`, `\sin`, `\sqrt`, `\pi` etc.
4. **No crashes or errors** in ingestion log

### Output Format

For each book, log:
```
Ingested <book_id>: pages=<N> templates=<N> artifacts=<N>
```

At the end, report cumulative totals:
```
books_v4 TOTALS:
- Books: <N>
- Pages: <N>
- Templates: <N>
- Artifacts: <N>
```

---

## Expected Outcome (Phase 6B Complete)

### Cumulative Statistics (Estimated)

**Current (Phase 6A):** 13 books, 3,367 pages, 22,986 templates, 928 artifacts

**After Phase 6B (+10 books):** ~23 books total

**Estimated totals:**
- Books: **23** (13 + 10)
- Pages: **~5,000-6,000** (depends on PDF sizes)
- Templates: **~35,000-40,000** (assuming similar density)
- Artifacts: **~1,400-1,600** (assuming similar theorem/definition density)

### Coverage Validation

After Phase 6B completion, verify comprehensive coverage:

**Calculus:**
- ✅ Multivariable Calculus (Stewart)
- ✅ Advanced Calculus (3 versions: advcalc, advanced_calc_alt, wrede_calculus)
- ✅ Advanced Calculus I & II
- ✅ Numerical Analysis

**Algebra/Discrete Math:**
- ✅ Linear Algebra Done Right
- ✅ Discrete Math (dmoi3)
- ✅ Transitions to Advanced Math
- ✅ Algebraic Topology

**Applied/Computational:**
- ✅ Math for Programmers (2 sources)
- ✅ Numerical Analysis
- ✅ Advanced Math Programming
- ✅ Python programming (computational patterns)

**Problem-Solving Methods:**
- ✅ ShortestShortcut (competition tricks)
- ✅ MathGems
- ✅ RPN methods (2 books)

**BasicMath:**
- ✅ AreaVol, NumberSets, PhysQuantities

**Missing domains (acceptable):**
- ❌ Financial Math (intentionally skipped - not relevant for MATH/AMC-AIME)
- ❌ Geometry (may need dedicated geometry textbook in future - flag if MATH has high geometry failure rate)
- ❌ Number Theory (covered partially in dmoi3, but may need dedicated book)
- ❌ Combinatorics (covered in dmoi3)

---

## Post-Ingestion Validation Benchmark

After Phase 6B ingestion is complete, run FULL benchmark suite:

### MATH Dataset (200 problems, seed 123)
```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --benchmark MATH \
  --max-problems 200 \
  --shuffle --shuffle-seed 123 \
  --use-trm-navigator \
  --disable-retrieval \
  --thinking-budget 8 \
  --shadow-readonly \
  --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4 \
  --book-max-books 32 \
  --book-top-k 5 \
  > /tmp/math_phase6b_books_v4_complete_200_seed123.log 2>&1
```

### AMC-AIME Dataset (200 problems, seed 123)
```bash
# Same as above but --benchmark AMC-AIME
# Log to: /tmp/amc_aime_phase6b_books_v4_complete_200_seed123.log
```

### Omni-MATH Dataset (200 problems, seed 123) - OPTIONAL
```bash
# Same as above but --benchmark Omni-MATH
# Log to: /tmp/omni_math_phase6b_books_v4_complete_200_seed123.log
```

---

## Success Criteria (Phase 6B)

### Ingestion Success
- ✅ All 10 remaining core math books ingested without errors
- ✅ Canonical Math Galaxy symbols indexed in all new books
- ✅ Total artifacts > 1,400
- ✅ Total templates > 35,000

### Coverage Success (Qualitative)
- ✅ Calculus comprehensively covered (6 books)
- ✅ Algebra/discrete math covered (4 books)
- ✅ Applied/computational math covered (4 books)
- ✅ Problem-solving methods covered (4 books)

### Benchmark Success (Threshold)
**Target:** Show improvement over Phase 6A

**Phase 6A baseline:**
- MATH: 2.0% (4/200)
- AMC-AIME: 0.5% (1/200)

**Phase 6B threshold (acceptable):**
- MATH: **≥ 3.0%** (6/200) - 50% relative improvement
- AMC-AIME: **≥ 1.5%** (3/200) - 3× improvement

**Phase 6B target (desired):**
- MATH: **≥ 5.0%** (10/200) - 2.5× improvement
- AMC-AIME: **≥ 3.0%** (6/200) - 6× improvement

### Quality Metrics
- `no_rule_match`: Should continue decreasing (currently 13 on MATH, 38 on AMC)
- `wrong_computation`: May increase (acceptable - more coverage = more candidates = more wrong matches initially)

**Note:** Quality refinement (Phase 7) will address wrong_computation AFTER coverage is complete.

---

## Timeline Estimate

**Per-book ingestion:** ~5-10 minutes (depends on PDF size)
**Total ingestion time:** ~50-100 minutes (10 books)
**Benchmark runs:** ~40 minutes (MATH + AMC-AIME, 200 each)
**Analysis + reporting:** ~20 minutes

**Total Phase 6B timeline:** ~2-3 hours

---

## Deliverables

1. **Ingestion logs:** `/tmp/phase6b_<book_id>.log` for each book
2. **Book galaxies:** `/K3D/Knowledge3D.local/galaxies/books_v4/<book_id>/` directories
3. **Cumulative statistics:** Final counts (books, pages, templates, artifacts)
4. **Benchmark logs:**
   - `/tmp/math_phase6b_books_v4_complete_200_seed123.log`
   - `/tmp/amc_aime_phase6b_books_v4_complete_200_seed123.log`
5. **Completion report:** `TEMP/CODEX_PHASE6B_COMPLETE_BOOK_INGESTION_12.19.2025.md`

---

## Special Instructions

### Deduplication Notes
- `advanced_calc_1_2` and `advanced_calculus` appear to be duplicates (identical page/template/artifact counts)
- **Action:** Flag this in the completion report, but keep both for now (deduplication is Phase 7+ work)

### Error Handling
- If a PDF fails to parse (corrupted, scanned images without OCR, etc.):
  - Log the error clearly
  - Skip that book
  - Continue with remaining books
  - Report failed books in completion report

### Index Validation
- For each new book, sample-check `token_index.json`:
  - Verify canonical Math Galaxy symbols present (e.g., `\cos`, `\pi`)
  - Verify plain forms also present (e.g., `cos`, `pi`)
  - If canonical symbols missing, investigate ingestion issue

---

## Next Phase Preview (Phase 7: Quality Refinement)

After Phase 6B completion, we will have:
- ✅ Comprehensive math knowledge coverage (~23 books)
- ✅ Math Galaxy symlink architecture validated
- ✅ Sovereign inference pipeline (no external normalization)

**Then we tackle quality:**
- Phase 7A: Diagnostic analysis (sample wrong_computation failures)
- Phase 7B: Candidate ranking improvements (artifact scoring, condition matching)
- Phase 7C: Multi-step chaining (theorem composition)

**Goal:** Use comprehensive coverage (Phase 6B) to improve candidate selection precision (Phase 7) → target MATH 10-15%, AMC 8-12%.

---

## Authorization

**Proceed immediately with Phase 6B ingestion.**

User directive: "make the prompt for full ingestion - only then we can refine and achieve the quality (with coverage first)"

**Coverage first, quality second.**

---

**Architect:** Claude (Architecture Partner)
**Date:** December 19, 2025
**Status:** READY FOR EXECUTION (Codex proceed immediately)
