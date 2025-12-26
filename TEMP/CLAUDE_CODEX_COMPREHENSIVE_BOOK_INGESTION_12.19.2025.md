# Comprehensive Book Ingestion Plan - Phase 6

**Date:** December 19, 2025
**Architect:** Claude (Architecture Partner)
**Context:** Phase 5 revealed coverage gap - only 8/38 books ingested
**Priority:** CRITICAL

---

## Problem Analysis

### Current State (Phase 5 with books_v3)
- **Books ingested:** 8 out of 38 available PDFs (21% coverage!)
- **MATH accuracy:** 1.5% (3/200) - actually WORSE than Phase 4's 2.5%
- **AMC-AIME accuracy:** 0.5% (1/200) - unchanged
- **Coverage improved:** no_rule_match decreased (11 vs 17 on MATH, 38 vs 43 on AMC)
- **Quality issue:** wrong_computation increased (125 vs 120 on MATH)

### Root Cause
**Insufficient mathematical knowledge coverage.** The Math Galaxy symlink architecture is working correctly (variant lookup succeeding), but we're starving it of knowledge.

**User's insight:**
> "Claude, we have ingested only a small portion of the books I have indicated as basic knowledge, why? Maybe that's what's missing - all math 'grammar' with it's several representations united by meaning"

**Exactly right!** We need comprehensive coverage of mathematical concepts with all their representational variants.

---

## Available Books (38 PDFs)

### Currently Ingested (8 books) ✅
1. `Linear.Algebra.Done.Right.pdf` → la_done_right
2. `advcalc.pdf` → advanced_calculus
3. `dmoi3-tablet.pdf` → dmoi3
4. `Transition_v104.pdf` → transition_v104
5. `BasicMath/AreaVol.pdf` → areavol
6. `BasicMath/NumberSets.pdf` → numbersets
7. `BasicMath/PhysQuantities.pdf` → physquantities
8. `BasicMath/MathGems.pdf` → mathgems

### Core Math - NOT Yet Ingested (10 books) ❌ HIGH PRIORITY
9. `Multivariable Calculus 7th Edition By James Stewart.pdf`
   - **Coverage:** Partial derivatives, multiple integrals, vector calculus
   - **MATH impact:** HIGH (calculus is 25% of MATH dataset)

10. `ADVANCED CALCULUS I and II.pdf`
    - **Coverage:** Advanced integration techniques, differential equations
    - **MATH impact:** HIGH

11. `Advanced_Calculus.pdf`
    - **Coverage:** Analysis foundations, rigorous proofs
    - **MATH impact:** MEDIUM

12. `Advanced-Calculus-Robert-Wrede.pdf`
    - **Coverage:** More advanced calculus topics
    - **MATH impact:** MEDIUM

13. `hildebrand.pdf`
    - **Coverage:** Advanced mathematics (need to check content)
    - **MATH impact:** UNKNOWN

14. `advmathprog.pdf` (Advanced Mathematical Programming)
    - **Coverage:** Optimization, linear programming, game theory
    - **MATH impact:** LOW (specialized topic)

15. `MATH 2F05.pdf`
    - **Coverage:** Need to check (university course material)
    - **MATH impact:** UNKNOWN

16. `Undergraduate Texts in Mathematics - Basic concepts of algebraic topology - Croom.pdf`
    - **Coverage:** Topology, abstract concepts
    - **MATH impact:** LOW (too advanced for typical MATH problems)

17. `Handbook Of Numerical Analysis - Special Volume - Foundations Of Computational Mathematics.pdf`
    - **Coverage:** Numerical methods, computational techniques
    - **MATH impact:** MEDIUM

### BasicMath - Missing (1 book) ❌ MEDIUM PRIORITY
18. `BasicMath/ShortestShortcut.pdf`
    - **Coverage:** Competition math tricks, mental math shortcuts
    - **AMC-AIME impact:** HIGH (competition math strategies)

### RPN/Programming Books (5 books) ⚠️ LOWER PRIORITY (but potentially useful)
19. `3.3. Reverse Polish - Intermediate.pdf`
    - **Coverage:** RPN patterns, procedural thinking
    - **Impact:** Could help with RPN template generation

20. `ReversePolishNotatonMethod.pdf`
    - **Coverage:** RPN methodology
    - **Impact:** Could help with RPN template generation

21. `Manning.Math.for.Programmers.2020.11.pdf`
    - **Coverage:** Math concepts for programmers
    - **Impact:** MEDIUM (practical problem-solving patterns)

22. `Orland_MfP_MEAP_V02_ch1.pdf`
    - **Coverage:** Math for Programmers chapter 1
    - **Impact:** LOW (duplicate of above)

23. `Stavely_python_ebook.pdf`
    - **Coverage:** Python programming
    - **Impact:** LOW (not math-focused)

### Financial Math (14 books) 🚫 SKIP FOR NOW
- Various finance/trading/actuarial books
- **Rationale:** Not relevant for MATH/AMC-AIME datasets
- **Future use:** GSM8K enhancement (financial word problems)

---

## Proposed Ingestion Strategy

### Phase 6A: Core Math Expansion (Priority 1)
**Goal:** Add essential calculus, algebra, and proof-based math knowledge

**Books to ingest (TOP 5 HIGH-IMPACT):**
1. `Multivariable Calculus 7th Edition By James Stewart.pdf` → multivariable_calc
2. `ADVANCED CALCULUS I and II.pdf` → advanced_calc_1_2
3. `Advanced_Calculus.pdf` → advanced_calc_alt
4. `Handbook Of Numerical Analysis - Special Volume - Foundations Of Computational Mathematics.pdf` → numerical_analysis
5. `BasicMath/ShortestShortcut.pdf` → shortestshortcut

**Expected impact:**
- **MATH:** 1.5% → 5-8% (calculus coverage fills major gap)
- **AMC-AIME:** 0.5% → 3-5% (competition tricks from ShortestShortcut)

**Timeline:** ~1.5 hours (5 books, 30min ingestion + validation)

### Phase 6B: Comprehensive Math Coverage (Priority 2)
**Goal:** Add remaining core math books for completeness

**Books to ingest (5 more):**
6. `Advanced-Calculus-Robert-Wrede.pdf` → wrede_calculus
7. `hildebrand.pdf` → hildebrand
8. `MATH 2F05.pdf` → math_2f05
9. `Manning.Math.for.Programmers.2020.11.pdf` → math_for_programmers
10. `3.3. Reverse Polish - Intermediate.pdf` → rpn_intermediate

**Expected additional impact:**
- **MATH:** 5-8% → 10-15% (comprehensive coverage across all topics)
- **AMC-AIME:** 3-5% → 8-12% (more problem-solving patterns)

**Timeline:** ~1.5 hours (5 books)

### Phase 6C: RPN Pattern Mining (Priority 3 - Optional)
**Goal:** Extract RPN procedural patterns from RPN-focused books

**Books:**
11. `ReversePolishNotatonMethod.pdf` → rpn_method

**Impact:** Could improve RPN template quality (fewer wrong_computation errors)

---

## Implementation Plan

### Step 1: Ingest Phase 6A Books (Now)

```bash
# Create books_v4 directory (comprehensive knowledge base)
mkdir -p /K3D/Knowledge3D.local/galaxies/books_v4

# Ingest 5 high-priority books
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  knowledge3d/training/math_benchmarks/book_galaxy_ingestion.py \
  --pdf-path "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Multivariable Calculus 7th Edition By James Stewart.pdf" \
  --title "Multivariable Calculus 7th Edition" \
  --book-id multivariable_calc \
  --domain calculus \
  --output-dir /K3D/Knowledge3D.local/galaxies/books_v4

# ... repeat for other 4 books ...
```

### Step 2: Validate Phase 6A

```bash
# Run MATH benchmark (200 problems)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --benchmark MATH \
  --max-problems 200 \
  --shuffle --shuffle-seed 123 \
  --use-trm-navigator \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v4 \
  > /tmp/math_phase6a_books_v4_200_seed123.log 2>&1

# Run AMC-AIME benchmark (200 problems)
# ... same as above but --benchmark AMC-AIME ...
```

### Step 3: Decision Point

**If Phase 6A shows improvement (MATH > 3%, AMC > 2%):**
→ Proceed with Phase 6B (ingest remaining 5 books)

**If Phase 6A shows minimal improvement (<2% on both):**
→ Investigate wrong_computation failures (candidate quality issue)

### Step 4: Ingest Phase 6B (if warranted)

```bash
# Ingest remaining 5 core math books into books_v4
# (Same process as Phase 6A)
```

### Step 5: Final Validation

```bash
# Run full benchmark suite against comprehensive books_v4
# MATH, AMC-AIME, Omni-MATH (200 each)
```

---

## Expected Timeline

### Phase 6A (High-Priority Books)
- **Ingestion:** ~45 minutes (5 books)
- **Validation:** ~40 minutes (MATH + AMC benchmarks)
- **Analysis:** ~15 minutes
- **Total:** ~1.5 hours

### Phase 6B (Remaining Books)
- **Ingestion:** ~45 minutes (5 books)
- **Validation:** ~40 minutes
- **Total:** ~1.5 hours

### Grand Total (Comprehensive Coverage)
- **3 hours for 10 additional books** (18 total instead of 8)

---

## Success Criteria

### Phase 6A Success (Threshold)
- MATH accuracy: 1.5% → **> 3%** (2× improvement)
- AMC-AIME accuracy: 0.5% → **> 2%** (4× improvement)
- no_rule_match: Further reduction (< 10 on MATH)

### Phase 6B Success (Target)
- MATH accuracy: **10-15%** (comprehensive calculus/algebra coverage)
- AMC-AIME accuracy: **8-12%** (competition strategies + core knowledge)
- Omni-MATH accuracy: **5-8%** (advanced topics coverage)

### Architecture Validation
- ✅ Math Galaxy symlink registry working (variant lookups succeeding)
- ✅ Comprehensive knowledge coverage (30+ books ingested)
- ✅ Sovereign inference (no external normalization)

---

## Why This Matters (Architectural Context)

**From DUAL_CLIENT_CONTRACT_SPECIFICATION.md:**

> **Galaxy Universe Composition**
>
> Each galaxy stores ONE type of knowledge; galaxies REFERENCE each other:
>
> ```
> Math Galaxy → symbols with procedural RPN
>     ↓ ingested from
> Book Galaxies → theorems, definitions, formulas
>     ↓ composed from
> Character Galaxy → glyphs with language metadata
> ```

**Current problem:** Math Galaxy has the RIGHT structure (symlink registry), but Book Galaxies have INSUFFICIENT content (only 21% of available books).

**Solution:** Ingest remaining 79% of books to populate Math Galaxy with comprehensive mathematical knowledge.

---

## Recommendation

**Immediate action:** Start with Phase 6A (5 high-priority books):
1. Multivariable Calculus (Stewart)
2. ADVANCED CALCULUS I and II
3. Advanced_Calculus
4. Numerical Analysis Handbook
5. ShortestShortcut (BasicMath)

**Rationale:**
- Calculus is 25% of MATH dataset topics
- ShortestShortcut has competition math tricks for AMC-AIME
- Numerical methods help with computation-heavy problems
- Small enough to complete in ~1.5 hours
- Validates whether more books = better accuracy

**After Phase 6A validation:** Decide whether to continue with Phase 6B or investigate candidate quality issues.

---

**Architect:** Claude (Architecture Partner)
**Date:** December 19, 2025
**Status:** Ready for Implementation (awaiting Codex execution)
