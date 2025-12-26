# Cross-Benchmark Transfer Analysis (Phase 2)

**Date:** December 18, 2025
**Architect:** Claude (Architecture Partner)
**Context:** Multi-benchmark baseline results from Phase 1

---

## Executive Summary

### Critical Finding: Poor Cross-Benchmark Transfer

Our architecture shows **strong performance on GSM8K (45.5%)** but **fails to generalize** to other math benchmarks:

| Benchmark | Accuracy | Transfer from GSM8K |
|-----------|----------|---------------------|
| GSM8K | 45.5% | (baseline) |
| MATH | 3.0% | **❌ 93% drop** |
| Omni-MATH | 0.0% | **❌ 100% drop** |
| AMC-AIME | 0.5% | **❌ 99% drop** |

**Interpretation:** Our generic patterns are NOT as general as claimed. We have **narrative arithmetic** patterns (GSM8K-specific problem types) but lack **symbolic manipulation**, **geometric reasoning**, and **number theory** patterns.

---

## Pattern Transfer Analysis

### What Transfers (Working on Multiple Benchmarks)

**Arithmetic operations:**
- ✅ Basic operations (+, -, *, /)
- ✅ Multi-step arithmetic chains
- ✅ Percent calculations (when narrative)
- ✅ Sum of products (when explicit counts)

**Why these transfer:** They appear in all benchmarks, though GSM8K uses them far more frequently.

### What Does NOT Transfer (GSM8K-Specific)

**Narrative patterns (high GSM8K frequency, rare elsewhere):**
- ❌ "After gifts" state tracking
- ❌ Weekly/daily schedule aggregation
- ❌ "How many more X than Y" relative chains
- ❌ "Twice as many as" narrative multipliers
- ❌ "Remaining equal value" partitioning
- ❌ Multi-entity budget/cost tracking

**Why these don't transfer:** MATH/Omni-MATH/AMC-AIME use **symbolic expressions**, not narrative word problems.

---

## Missing Pattern Families (Coverage Gaps)

### Coverage Comparison

| Category | GSM8K | MATH | Omni-MATH | AMC-AIME |
|----------|-------|------|-----------|----------|
| **no_rule_match** | 0% ✅ | 10.5% | 13.5% | 18.5% |
| **algebra_needed** | 0% | 17% | 3% | 14% |
| **wrong_computation** | 26% | 57% | 51.5% | 38.5% |

**Key insight:** We have 100% coverage on GSM8K but **10-18% coverage gaps** on other benchmarks. Additionally, even when we have pattern matches, **composition quality is worse** (50-57% wrong_computation vs. 26% on GSM8K).

### Missing Pattern Families (by Benchmark)

#### MATH Dataset (3.0% accuracy)

**Missing patterns:**
1. **Trigonometric functions** (cos, sin, tan)
   - Example: "Compute \\cos 60^\\circ"
   - Current: NO_RULE_MATCH
   - Needed: Math Galaxy entries for trig values, unit circle

2. **Number theory** (factors, divisibility, modular arithmetic)
   - Example: "How many positive two-digit integers have an odd number of positive factors?"
   - Current: NO_RULE_MATCH
   - Needed: Factor counting rules, perfect square detection

3. **Linear algebra** (matrices, determinants, inverses)
   - Example: "matrix non-invertible ... list all possible values"
   - Current: NO_RULE_MATCH
   - Needed: Matrix operations in RPN (requires 2D tensor ops)

4. **Algebraic manipulation** (factor, expand, simplify)
   - Example: "Find $a+b$ if piecewise function is continuous"
   - Current: wrong_computation (generates partial RPN)
   - Needed: Equation constraint solving (continuity → equal at boundaries)

5. **Symbolic expressions** (LaTeX parsing, variable binding)
   - Example: "Let f(x) = ax+3 if x>2..."
   - Current: Extracts numbers but loses variable relationships
   - Needed: Symbolic equation system solver

#### Omni-MATH Dataset (0.0% accuracy)

**Missing patterns:**
1. **Graph theory** (paths, cycles, chromatic numbers)
   - Example: "bug on a cube (graph/path counting)"
   - Current: NO_RULE_MATCH
   - Needed: Combinatorial graph patterns

2. **Abstract algebra** (groups, automorphisms, homomorphisms)
   - Example: "finite groups / automorphism sizes"
   - Current: NO_RULE_MATCH
   - Needed: Domain outside current scope (requires theorem prover)

3. **Optimization** (maximize/minimize under constraints)
   - Example: "Find minimum m satisfying conditions..."
   - Current: NO_RULE_MATCH
   - Needed: Constrained optimization patterns

4. **Calculus** (derivatives, integrals, limits)
   - Current: Not observed in sample, but likely in full dataset
   - Needed: Calculus rules in Galaxy (power rule, chain rule, etc.)

#### AMC-AIME Dataset (0.5% accuracy)

**Missing patterns:**
1. **Geometry** (triangles, trapezoids, circles, angles)
   - Example: "trapezoid midpoints area relation"
   - Current: NO_RULE_MATCH
   - Needed: Geometric formulas (area, perimeter, angle relationships)

2. **Modular arithmetic** (congruences, mod operations)
   - Example: "modular congruence 14u ≡ 46 (mod 100)"
   - Current: NO_RULE_MATCH
   - Needed: Modular arithmetic RPN ops (mod, gcd, lcm)

3. **Statistics** (mean, median, mode, percentiles)
   - Example: "stem-and-leaf plot median/mode mean"
   - Current: NO_RULE_MATCH
   - Needed: Statistical aggregation patterns

4. **Combinatorics** (permutations, combinations, counting)
   - Example: "How many ways to arrange..."
   - Current: NO_RULE_MATCH
   - Needed: Factorial, nCr, nPr operations in RPN

5. **Number theory** (primes, divisibility, gcd/lcm)
   - Example: Various number theory problems
   - Current: NO_RULE_MATCH
   - Needed: Number theory primitives

---

## Root Cause Analysis

### Why Did Transfer Fail?

**Hypothesis 1: Narrative vs. Symbolic**
- GSM8K = **narrative arithmetic** (word problems with explicit numbers)
- MATH/Omni/AMC = **symbolic reasoning** (LaTeX expressions, variables, constraints)
- Our patterns extract numbers from narratives, not parse symbolic expressions

**Evidence:**
- GSM8K: "There are 100 students. 80% are girls..." → extracts [100, 80], generates RPN
- MATH: "Let f(x) = ax+3 if x>2..." → extracts [3, 2] but loses variables a, x

**Hypothesis 2: Domain Coverage**
- GSM8K = arithmetic + basic algebra + word problems (narrow domain)
- MATH/Omni/AMC = arithmetic + algebra + geometry + number theory + trig + calculus + ... (broad domain)
- Our 150+ Grammar rules cover GSM8K domains, not others

**Evidence:**
- GSM8K domains: percentages, costs, schedules, ratios, fractions
- MATH domains: trigonometry (0 rules), linear algebra (0 rules), number theory (0 rules)

**Hypothesis 3: Composition Quality**
- Even when patterns match, RPN composition is worse on other benchmarks
- wrong_computation: 26% (GSM8K) vs. 57% (MATH) vs. 51.5% (Omni) vs. 38.5% (AMC)
- Our TTC candidates are tuned for GSM8K problem structures

**Evidence:**
- GSM8K: "twice as many" → generates `base 2 *` (works well)
- MATH: "twice as many" in algebraic context → generates wrong constraint

### Architectural Implications

**What we thought:**
- "Generic building blocks + TRM exploration > hardcoded task-specific rules"
- "Patterns learned in math help visual reasoning, and vice versa"
- "Galaxy-First design ensures cross-domain transfer"

**What the data shows:**
- Our "generic" patterns are actually **GSM8K domain-specific**
- Transfer to other math domains is POOR (3%, 0%, 0.5%)
- Galaxy-First design is correct, but we populated Galaxy with GSM8K patterns only

**The good news:**
- Architecture is sound (Galaxy + TRM + TTC framework works)
- GSM8K validation proves the paradigm (4% → 45.5%)
- We just need to add OTHER domain patterns to Galaxy

**The challenge:**
- Adding MATH/Omni/AMC patterns is MUCH harder than GSM8K
- Symbolic manipulation requires parsing LaTeX, binding variables, solving constraints
- Some domains (abstract algebra, calculus) may require theorem provers

---

## Comparison to Success Criteria

From [CLAUDE_MULTI_BENCHMARK_VALIDATION_12.18.2025.md](CLAUDE_MULTI_BENCHMARK_VALIDATION_12.18.2025.md):

### Minimum Viable Validation ❌
- [x] GSM8K: >40% ✅ (45.5%)
- [ ] MATH: >15% ❌ (3.0% - missed by 12%)
- [ ] Omni-MATH: >5% ❌ (0.0% - missed by 5%)
- [ ] AMC-AIME: >10% ❌ (0.5% - missed by 9.5%)

**Result:** Did NOT meet minimum criteria for sovereignty transition.

### Strong Validation ❌
- [x] GSM8K: >40% ✅
- [ ] MATH: >20% ❌
- [ ] Omni-MATH: >10% ❌
- [ ] AMC-AIME: >15% ❌
- [ ] Pattern transfer: >70% ❌ (actual: <10%)

**Result:** Did NOT meet strong validation criteria.

### Exceptional Validation ❌
- All criteria failed except GSM8K baseline.

---

## Pattern Transfer Metrics

### Domain Coverage Heatmap

| Domain | GSM8K | MATH | Omni | AMC | Transfer Score |
|--------|-------|------|------|-----|----------------|
| **Arithmetic** | ✅✅✅ | ✅✅ | ✅✅ | ✅✅ | **High (80%)** |
| **Basic Algebra** | ✅✅✅ | ✅ | ✅ | ✅ | **Medium (50%)** |
| **Percentages** | ✅✅✅ | ✅ | ❌ | ✅ | **Medium (60%)** |
| **Word Problems** | ✅✅✅ | ❌ | ❌ | ❌ | **Low (5%)** |
| **Symbolic Algebra** | ❌ | ❌❌❌ | ❌❌ | ❌❌ | **None (0%)** |
| **Geometry** | ❌ | ❌❌ | ❌ | ❌❌❌ | **None (0%)** |
| **Number Theory** | ❌ | ❌❌ | ❌ | ❌❌ | **None (0%)** |
| **Trigonometry** | ❌ | ❌❌ | ❌ | ❌ | **None (0%)** |
| **Calculus** | ❌ | ❌ | ❌❌ | ❌ | **None (0%)** |
| **Statistics** | ❌ | ❌ | ❌ | ❌❌ | **None (0%)** |

**Overall transfer score: ~20%** (only arithmetic and basic algebra patterns transfer)

---

## Recommended Next Steps

### Option A: Continue Python Iteration (Recommended)

**Goal:** Add generic patterns for MATH/AMC domains, validate transfer before sovereignty.

**Phase 2A: Symbolic Algebra Patterns (Target: MATH 10-15%)**
1. Add LaTeX parsing to extract variables and equations
2. Add constraint-solving candidates (continuity, equal at boundary)
3. Add polynomial manipulation (factor, expand, roots)
4. Estimated impact: 3% → 10-15% on MATH

**Phase 2B: Geometry and Trig (Target: AMC 5-10%)**
1. Add geometric formula patterns (area, perimeter, angles)
2. Add trigonometric values (unit circle, special angles)
3. Add Pythagorean theorem and triangle rules
4. Estimated impact: 0.5% → 5-10% on AMC-AIME

**Phase 2C: Number Theory (Target: AMC 8-12%)**
1. Add modular arithmetic operations (mod, gcd, lcm)
2. Add prime factorization and divisibility rules
3. Add combinatorics (factorial, nCr, nPr)
4. Estimated impact: 0.5% → 8-12% on AMC-AIME

**Timeline:** 2-3 weeks of iteration
**Outcome:** Multi-domain validation BEFORE sovereignty transition

### Option B: Transition to Sovereignty Now (Not Recommended)

**Why not recommended:**
- Current patterns are GSM8K-specific (poor transfer)
- Migrating to Galaxy now would "freeze" a narrow architecture
- We'd need to add new domains later anyway (better to do in Python first)

**When to reconsider:**
- After Option A phases complete
- Once we have validated multi-domain patterns
- When MATH >15%, AMC >10%, Omni >5%

### Option C: Hybrid Approach (Compromise)

**Goal:** Transition GSM8K patterns to sovereignty, continue Python for other domains.

**Phase 1: Sovereignty for GSM8K**
- Extract current 150+ Grammar rules → Galaxy entries
- Wire TRM navigation to Galaxy
- Validate 45.5% accuracy maintained

**Phase 2: Python iteration for MATH/AMC**
- Add symbolic/geometry/number theory patterns in Python
- Test multi-domain transfer
- Migrate to Galaxy once validated

**Advantage:** Demonstrates sovereignty architecture sooner
**Disadvantage:** Maintaining two systems (Python + Galaxy) during transition

---

## Architectural Lessons Learned

### What Worked

1. **Galaxy-First framework** ✅
   - Architecture is sound (Galaxy + TRM + TTC)
   - Just need to populate with more domains

2. **Test-Time Compute** ✅
   - Variable thinking depth works well
   - Plausibility filtering is effective
   - Candidate exploration scales

3. **Data-driven iteration** ✅
   - Diagnostic metadata reveals exact gaps
   - Failure categorization guides fixes
   - Regression tests prevent backsliding

4. **Generic patterns (within GSM8K domain)** ✅
   - Relative chains, percent operations, multi-item costs transfer within GSM8K
   - 102 regression tests passing
   - 45.5% accuracy achieved

### What Needs Improvement

1. **Cross-domain coverage** ❌
   - GSM8K patterns don't transfer to MATH/Omni/AMC
   - Need explicit multi-domain validation DURING development
   - Should test on 2-3 benchmarks simultaneously, not just GSM8K

2. **Symbolic reasoning** ❌
   - Current architecture extracts numbers from narratives
   - Needs LaTeX parser, variable binding, equation solver
   - Fundamental extension required (not just new patterns)

3. **Domain breadth** ❌
   - 150+ rules cover arithmetic + basic algebra only
   - Missing: geometry, trig, number theory, calculus, statistics
   - Each domain needs dedicated pattern family

4. **"Generic" claim validation** ❌
   - We claimed patterns were generic, but they're GSM8K-specific
   - True genericity requires multi-benchmark validation
   - Lesson: Test transfer EARLY, not after optimization

---

## Decision Point

**Question:** Should we proceed to sovereignty transition now, or continue Python iteration for multi-domain patterns?

**Claude's Recommendation: Option A (Continue Python Iteration)**

**Rationale:**
1. Current patterns are GSM8K-specific (poor transfer)
2. Sovereignty transition should happen AFTER multi-domain validation
3. Adding MATH/AMC patterns in Python is faster than in Galaxy (can iterate quickly)
4. Once multi-domain patterns validated, THEN migrate to Galaxy

**Expected timeline:**
- Phase 2A (Symbolic Algebra): 1 week → MATH 10-15%
- Phase 2B (Geometry/Trig): 1 week → AMC 5-10%
- Phase 2C (Number Theory): 1 week → AMC 8-12%
- **Total: 3 weeks** to multi-domain validation
- **Then:** Sovereignty transition with confidence

**Alternative (if time-constrained):**
- Option C (Hybrid): Sovereignty for GSM8K now, Python for other domains
- Demonstrates architecture while continuing iteration

---

## Next Actions

### For User Decision

**Question to user:**
- Proceed with Option A (continue Python iteration for multi-domain)?
- OR Option C (hybrid: sovereignty for GSM8K, Python for others)?
- OR Option B (sovereignty now, accept narrow domain)?

### If Option A Chosen (Recommended)

**Phase 2A: Symbolic Algebra (1 week)**
1. Add LaTeX parsing to TRMGalaxyReader
2. Add equation constraint patterns (continuity, boundary conditions)
3. Add polynomial manipulation seeds
4. Target: MATH 10-15% accuracy
5. Validate with 50-100 MATH problems

**Phase 2B: Geometry/Trig (1 week)**
1. Add geometric formula patterns (area, angles, Pythagorean)
2. Add trigonometric unit circle values
3. Target: AMC 5-10% accuracy
4. Validate with 50-100 AMC problems

**Phase 2C: Number Theory (1 week)**
1. Add modular arithmetic ops (mod, gcd, lcm)
2. Add combinatorics (factorial, nCr, nPr)
3. Add prime/divisibility rules
4. Target: AMC 8-12% accuracy

### If Option C Chosen (Hybrid)

**Phase 1: GSM8K Sovereignty (1 week)**
1. Extract 150+ Grammar rules → Galaxy entries
2. Wire TRM to navigate Galaxy
3. Validate 45.5% maintained on GSM8K

**Phase 2: Multi-domain Python (2-3 weeks)**
1. Same as Option A phases 2A, 2B, 2C
2. Migrate to Galaxy once validated

---

## Files and Logs

**Baseline logs:**
- `/tmp/gsm8k_200_seed123_after_schedule_day_sum.log` - GSM8K 45.5%
- `/tmp/math_baseline_200_seed123.log` - MATH 3.0%
- `/tmp/omni_math_baseline_200_seed123.log` - Omni-MATH 0.0%
- `/tmp/amc_aime_baseline_200_seed123.log` - AMC-AIME 0.5%

**Reports:**
- [CODEX_MULTI_BENCHMARK_BASELINE_12.18.2025.md](CODEX_MULTI_BENCHMARK_BASELINE_12.18.2025.md) - Phase 1 results
- [CLAUDE_MULTI_BENCHMARK_VALIDATION_12.18.2025.md](CLAUDE_MULTI_BENCHMARK_VALIDATION_12.18.2025.md) - Original plan

---

**Architect:** Claude (Architecture Partner)
**Status:** Phase 2 analysis complete, awaiting user decision
**Recommendation:** Option A (continue Python iteration for multi-domain validation)
