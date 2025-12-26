# Math Benchmark Real Baseline Report

**Date:** December 15, 2025
**TRM Navigator:** Enabled
**Shadow Copy:** Active (learning)

---

## Baseline Results

| Dataset | Problems | Correct | Accuracy | TRM Hits | Notes |
|---------|----------|---------|----------|----------|-------|
| GSM8K | 50 | 3 | **6.00%** | 26 | Word problems |
| MATH | 50 | 1 | **2.00%** | 22 | Competition math |
| Omni-MATH | 30 | 0 | 0.00% | 13 | Very hard |
| AMC-AIME | 30 | 0 | 0.00% | 20 | Competition |
| MMLU | 30 | 8 | **26.67%** | 0 | Multiple choice |

**Total:** 190 problems, 12 correct (6.3% overall)

---

## Solver Distribution

### GSM8K (50 problems)
- TRM Navigator: 26 (52%)
- Composer: 17 (34%)
- Template: 3 (6%)
- Word: 2 (4%)
- Knowledge: 2 (4%)

### MATH (50 problems)
- TRM Navigator: 22 (44%)
- Word: 12 (24%)
- Composer: 11 (22%)
- Template: 3 (6%)
- Knowledge: 2 (4%)

**Key Insight:** TRM Navigator is hitting 44-52% of problems, but accuracy is low → rules match but produce wrong answers.

---

## Failure Analysis

### Run 1 (GSM8K + MATH): 96 failures
| Category | Count | % |
|----------|-------|---|
| wrong_computation | 50 | 52% |
| word_problem | 16 | 17% |
| multi_step_needed | 15 | 16% |
| algebra_needed | 15 | 16% |

### Run 2 (Omni + AMC + MMLU): 82 failures
| Category | Count | % |
|----------|-------|---|
| wrong_computation | 36 | 44% |
| unknown | 20 | 24% |
| algebra_needed | 12 | 15% |
| word_problem | 10 | 12% |
| multi_step_needed | 4 | 5% |

### Key Findings

1. **wrong_computation (50%+)** - Rules match but RPN produces wrong answer
   - Likely cause: Regex captures wrong numbers
   - Fix: Improve pattern specificity

2. **word_problem (12-17%)** - Natural language extraction fails
   - Needs more GSM8K templates
   - Current: 10 templates, need 25+

3. **multi_step_needed (5-16%)** - Single-step insufficient
   - STORE/RECALL now implemented
   - Need more multi-step grammar rules

4. **algebra_needed (15-16%)** - Equation solving required
   - Quadratic/linear rules added
   - Need better pattern matching for equations

---

## Shadow Copy Learning

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Starting | 1 | 43 |
| Ending | 43 | 68 |
| Growth | +42 | +25 |

**Pattern confidence tracking active.** Discoveries persisted to:
- `/K3D/Knowledge3D.local/checkpoints/math_benchmarks/shadow_copy.json`

---

## Galaxy Status

```
Math Galaxy: 197 → 2740 symbols (after knowledge load)
Grammar Rules: 747 base + 115 math-specific
  - word_rules: 21
  - algebra_rules: 2
  - gsm8k_templates: 10
  - competition_rules: 28
  - calculus_rules: 8
  - symbolic_rules: 15
  - sovereign_rules: 15
```

---

## Next Steps (Priority Order)

### 1. Fix wrong_computation (50% of failures)
- **Issue:** Regex captures wrong numbers from problem text
- **Action:** Audit top-failing rules, improve pattern specificity
- **Target:** Reduce wrong_computation to <30%

### 2. Expand GSM8K templates (17% word_problem failures)
- **Current:** 10 templates
- **Target:** 25+ templates covering common patterns
- **Examples needed:**
  - "X gave Y to Z, how many left?"
  - "Bought N at $X each, total cost?"
  - "Percentage increase/decrease"

### 3. Multi-step algebra chains
- **STORE/RECALL:** Implemented ✓
- **Need:** More chained rules (quadratic → roots, systems)
- **Test:** `x^2 - 5x + 6 = 0` should return 2 or 3

### 4. Competition math patterns
- **AMC/AIME:** 0% accuracy
- **Need:** Specialized competition templates
- **Focus:** Number theory, combinatorics, geometry

---

## Accuracy Targets

| Phase | GSM8K | MATH | MMLU |
|-------|-------|------|------|
| Baseline (now) | 6% | 2% | 27% |
| + Fix wrong_comp | 15% | 8% | 35% |
| + More templates | 25% | 12% | 40% |
| + Multi-step | 35% | 18% | 45% |

---

## Technical Notes

- **Sovereignty:** Maintained (no numpy in hot path)
- **TRM Learning:** Shadow copy active, 68 discoveries
- **Pattern Confidence:** Tracking per-rule success rates
- **Checkpoint:** Auto-saving after each run

---

**Status:** Baseline established. Ready for iterative improvement.
