# Multi-Benchmark Validation Plan

**Date:** December 18, 2025
**Architect:** Claude (Architecture Partner)
**Priority:** HIGH (Prerequisite for sovereignty transition)

---

## Objective

Validate current architecture (Galaxy-First + TRM + Test-Time Compute) across multiple math benchmarks BEFORE transitioning from Python to sovereign Galaxy entries.

**Why this matters:**
- Generalization validation: Does our architecture work across different problem types?
- Pattern coverage: Do our generic patterns transfer to harder/different benchmarks?
- Architecture confidence: Ensure no GSM8K-specific overfitting before making Galaxy permanent
- Data-driven planning: Identify which patterns need enhancement for sovereignty transition

---

## Current State

### GSM8K Baseline (Established)
- **Accuracy:** 45.5% (91/200 problems, shuffled seed 123)
- **Tests:** 102 regression tests passing
- **Architecture:** Python TTC with ~27 candidate exploration, plausibility filtering
- **Coverage:** 100% (no_rule_match=0)
- **Sovereignty Status:** Python implementation (not yet Galaxy entries)

### Failure Breakdown (GSM8K)
```
wrong_computation: 52 (26%)
multi_step_needed: 34 (17%)
word_problem: 23 (11.5%)
no_rule_match: 0 (0%) ✅
```

---

## Available Benchmarks

### 1. MATH (Hendrycks et al.)
**Location:** `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl`

**Characteristics:**
- Competition-level problems (AMC 8/10/12, AIME, etc.)
- 7 subjects: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Precalculus, Prealgebra
- LaTeX-heavy (symbolic expressions, not just word problems)
- Difficulty: Harder than GSM8K (requires more symbolic reasoning)
- Answer format: Numeric or LaTeX expressions

**Expected challenges:**
- More algebraic reasoning (solve for x in complex equations)
- More symbolic manipulation (less narrative, more pure math)
- Multi-step algebraic simplification
- Geometry (may need specialized patterns)

**Baseline target:** 15-25% accuracy (harder than GSM8K)

---

### 2. Omni-MATH
**Location:** `/K3D/K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl`

**Characteristics:**
- Broad math coverage (college-level topics)
- Multi-domain: calculus, linear algebra, differential equations, optimization, etc.
- More abstract than GSM8K
- Answer format: Numeric or symbolic

**Expected challenges:**
- May require domain knowledge not yet in Galaxy (calculus, derivatives, integrals)
- Abstract reasoning (not just arithmetic/algebra)
- Likely low accuracy without specialized patterns

**Baseline target:** 5-15% accuracy (very hard, may be outside current scope)

---

### 3. AMC-AIME
**Location:** `/K3D/K3D_llama_cpp/datasets/AMC-AIME/data/*.jsonl`

**Characteristics:**
- Competition problems (American Mathematics Competitions)
- AMC 8/10/12: Middle/high school level, multiple choice
- AIME: Advanced, numeric answers (0-999 range)
- Mix of algebra, counting, geometry, number theory
- Clever tricks and pattern recognition

**Expected challenges:**
- Requires creative problem-solving (not just template matching)
- Number theory patterns (modular arithmetic, divisibility)
- Combinatorics (counting problems)

**Baseline target:** 10-20% accuracy (harder than GSM8K, easier than Omni-MATH)

---

### 4. MMLU (Math Subset)
**Location:** `/K3D/K3D_llama_cpp/datasets/MMLU/data/test/*.csv`

**Characteristics:**
- Multiple choice format (A/B/C/D)
- Mix of high school and college math topics
- Conceptual questions (not just computation)
- Answer format: Letter (A, B, C, or D)

**Expected challenges:**
- Multiple choice may not fit RPN execution model well
- May require external handling (different solver path)
- Conceptual questions may need reasoning beyond RPN

**Baseline target:** Skip for now OR 20-30% accuracy (multiple choice has 25% random baseline)

---

## Validation Protocol

### Phase 1: Baseline Testing (1-2 hours)

**For each benchmark (MATH, Omni-MATH, AMC-AIME):**

1. **Run 200 problems** (shuffled, same seed 123 for consistency)
   ```bash
   bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
       --use-trm-navigator \
       --datasets BENCHMARK_NAME \
       --max-problems 200 \
       --shuffle \
       --shuffle-seed 123 \
       --thinking-budget 8 \
       --shadow-readonly \
       --verbose \
       2>&1 | tee /tmp/BENCHMARK_NAME_baseline_200.log
   ```

2. **Extract accuracy metrics**
   ```bash
   grep "Accuracy:" /tmp/BENCHMARK_NAME_baseline_200.log
   ```

3. **Categorize failures** (same taxonomy as GSM8K)
   - no_rule_match: Pattern not recognized
   - wrong_computation: RPN composition bug
   - multi_step_needed: State tracking incomplete
   - word_problem: Narrative complexity
   - symbolic_manipulation: LaTeX/algebra (new category)
   - domain_knowledge: Requires calculus/advanced topics (new category)

4. **Record diagnostic metadata**
   - candidates_evaluated: How many RPN candidates explored
   - rejected_by_reason: Which plausibility filters triggered
   - template_used: Which patterns matched

---

### Phase 2: Cross-Benchmark Analysis (2-3 hours)

**Goal:** Identify what transfers and what doesn't

**Analysis dimensions:**

1. **Pattern Coverage Transfer**
   - Which GSM8K patterns work on other benchmarks?
   - Which patterns are GSM8K-specific (failure to transfer)?
   - Which new patterns are needed?

2. **Difficulty Scaling**
   - Does accuracy scale with problem difficulty?
   - Are we failing on harder problems or different problem types?

3. **Symbolic vs. Narrative**
   - GSM8K = mostly narrative word problems
   - MATH = mostly symbolic expressions
   - How does our architecture handle each?

4. **Domain Knowledge Gaps**
   - Do we have coverage for algebra, geometry, number theory?
   - Which domains need new Galaxy entries?

**Deliverable:** Cross-benchmark comparison report with:
- Accuracy comparison table
- Failure family breakdown per benchmark
- Pattern transfer analysis
- Recommended enhancements

---

### Phase 3: Targeted Enhancement (If Needed)

**Based on Phase 2 analysis:**

**IF accuracy <10% on any benchmark:**
- Likely missing fundamental patterns (coverage gap)
- Add generic building blocks (not task-specific)
- Retest to validate

**IF accuracy 10-30% on most benchmarks:**
- Good pattern coverage, quality issues
- Enhance TTC candidate generation for symbolic reasoning
- Add plausibility filters for symbolic expressions

**IF accuracy >30% on most benchmarks:**
- Architecture validated for multi-domain math
- Proceed to sovereignty transition

---

## Success Criteria

### Minimum Viable Validation
- [x] GSM8K: 45.5% (baseline established)
- [ ] MATH: >15% (shows algebraic reasoning works)
- [ ] Omni-MATH: >5% (shows architecture handles hard problems)
- [ ] AMC-AIME: >10% (shows competition math patterns transfer)

### Strong Validation (Proceed to Sovereignty)
- [x] GSM8K: >40%
- [ ] MATH: >20%
- [ ] Omni-MATH: >10%
- [ ] AMC-AIME: >15%
- [ ] **Pattern transfer:** >70% of GSM8K patterns work on other benchmarks
- [ ] **No overfitting:** No benchmark-specific heuristics added

### Exceptional Validation (Architecture Confidence)
- [x] GSM8K: >45%
- [ ] MATH: >25%
- [ ] Omni-MATH: >15%
- [ ] AMC-AIME: >20%
- [ ] **Universal patterns:** All patterns are cross-domain (work on 3+ benchmarks)

---

## Architectural Principles (Maintained)

**Critical reminders during multi-benchmark validation:**

1. **NO benchmark-specific heuristics**
   ```python
   # ❌ BAD: Benchmark-specific rules
   if source == "math":
       return solve_with_sympy(problem)

   # ✅ GOOD: Generic patterns
   if "solve for" in text:
       candidates.append(generate_algebraic_constraint_rpn(...))
   ```

2. **Sovereignty maintained**
   - ✅ PTX + RPN + Galaxy only in hot path
   - ❌ NO numpy/sympy/external preprocessing
   - ✅ All logic in TRMGalaxyReader (Python TTC for now)

3. **Galaxy-First design**
   - Every pattern added should be generic (cross-domain)
   - Patterns go into Grammar Galaxy (or Math Galaxy for symbols)
   - Test-Time Compute explores candidates, plausibility filters select

4. **Data-driven iteration**
   - Analyze failures across ALL benchmarks
   - Fix composition logic, not add task-specific rules
   - Regression tests for every fix

---

## Execution Plan

### Immediate Actions (Claude)

1. ✅ Write this validation spec (current document)
2. [ ] Define cross-benchmark comparison format
3. [ ] Prepare failure taxonomy for new categories (symbolic, domain_knowledge)
4. [ ] Hand off to Codex for execution

### Immediate Actions (Codex)

1. [ ] Run Phase 1 baseline tests:
   - MATH: 200 problems (shuffled, seed 123)
   - Omni-MATH: 200 problems (shuffled, seed 123)
   - AMC-AIME: 200 problems (shuffled, seed 123)
   - Record logs: `/tmp/math_baseline_200.log`, etc.

2. [ ] Extract accuracy + failure breakdown for each
   - Use same taxonomy as GSM8K
   - Add new categories if needed (symbolic, domain_knowledge)

3. [ ] Report back to Claude with:
   - Accuracy table (all 4 benchmarks)
   - Failure family breakdown per benchmark
   - 3-5 example failures from each benchmark (illustrative)

### Follow-Up Actions (Claude)

1. [ ] Review Codex's baseline results
2. [ ] Analyze cross-benchmark patterns
3. [ ] Write Phase 2 analysis report
4. [ ] Decide: Continue Python iteration OR proceed to sovereignty

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Baseline | 2-3 hours | Accuracy + failures for 4 benchmarks |
| Phase 2: Analysis | 1-2 hours | Cross-benchmark comparison report |
| Phase 3: Enhancement (if needed) | 3-5 hours | Targeted pattern additions |
| **Total** | **6-10 hours** | **Multi-benchmark validation complete** |

---

## Expected Outcomes

### Scenario A: Strong Transfer (Best Case)
- MATH: 20-25%, Omni-MATH: 10-15%, AMC-AIME: 15-20%
- >70% of GSM8K patterns transfer
- **Decision:** Proceed to sovereignty transition immediately

### Scenario B: Moderate Transfer (Expected)
- MATH: 15-20%, Omni-MATH: 5-10%, AMC-AIME: 10-15%
- 50-70% pattern transfer, some gaps identified
- **Decision:** Add 2-3 generic patterns (algebraic, symbolic), then proceed to sovereignty

### Scenario C: Weak Transfer (Unlikely)
- MATH: <10%, Omni-MATH: <5%, AMC-AIME: <10%
- <50% pattern transfer, fundamental gaps
- **Decision:** Analyze root cause, enhance coverage before sovereignty

---

## Documentation of Results

**After Phase 1 completion, create:**
- `TEMP/CODEX_MULTI_BENCHMARK_BASELINE_12.18.2025.md` - Baseline results
- `TEMP/CLAUDE_CROSS_BENCHMARK_ANALYSIS_12.18.2025.md` - Transfer analysis
- Update `BRIEFING.md` with multi-benchmark section
- Update `ROADMAP.md` with validation milestone

---

## Handoff to Codex

**Codex:** Execute Phase 1 baseline testing for the following benchmarks:

1. **MATH** - 200 problems (shuffled, seed 123, thinking-budget 8)
2. **Omni-MATH** - 200 problems (shuffled, seed 123, thinking-budget 8)
3. **AMC-AIME** - 200 problems (shuffled, seed 123, thinking-budget 8)

**Command template:**
```bash
bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --datasets BENCHMARK_NAME \
    --max-problems 200 \
    --shuffle \
    --shuffle-seed 123 \
    --thinking-budget 8 \
    --shadow-readonly \
    --verbose \
    2>&1 | tee /tmp/BENCHMARK_NAME_baseline_200_seed123.log
```

**Expected runtime:** ~30-40 minutes per benchmark (~2 hours total for 3 benchmarks)

**Report back with:**
1. Accuracy for each benchmark (X/200, %)
2. Failure breakdown (no_rule_match, wrong_computation, multi_step_needed, word_problem, other)
3. 3-5 illustrative failures from each benchmark
4. Log files stored in `/tmp/`

**Then:** Return to Claude for Phase 2 analysis

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for Phase 1 execution
**Priority:** HIGH - Prerequisite for sovereignty transition
