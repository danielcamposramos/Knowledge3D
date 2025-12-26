# Architectural Milestone: Coverage → Quality Paradigm Shift

**Date:** December 17, 2025
**Architect:** Claude (Architecture Partner)
**Context:** GSM8K test-time compute implementation

---

## What Just Happened: The Paradigm Shift

### Results Summary

| Metric | Before TIER-1 | After TIER-1 | Change |
|--------|---------------|--------------|--------|
| **Accuracy** | 2.50% (5/200) | 3.00% (6/200) | +0.5% |
| **no_rule_match** | ~55% (Phase 5) | **0** | ✅ SOLVED |
| **wrong_computation** | ~50% | 122 (61%) | ⚠️ DOMINANT |
| **multi_step_needed** | ~20% | 44 (22%) | → COMPOSITION |

### Critical Architectural Insight

**We've crossed a fundamental threshold:**

**Before:** Bottleneck = COVERAGE
- Problem: "Can't find any pattern to match"
- Solution: Add more Grammar rules
- Metric: Reduce no_rule_match

**After:** Bottleneck = QUALITY
- Problem: "Finding patterns but computing wrong results"
- Solution: Improve RPN program composition
- Metric: Reduce wrong_computation

**This is not incremental progress - it's a PARADIGM SHIFT.**

---

## Why This Matters Architecturally

### Phase Progression Analysis

| Phase | Focus | Bottleneck | Accuracy | no_rule_match |
|-------|-------|------------|----------|---------------|
| **Phase 1-2** | Basic templates | Coverage | 1.53% | 55% |
| **Phase 3** | Expand patterns | Coverage | 6% (subset) | 47% |
| **Phase 4** | Compositional learning | Coverage | 12% (overfit) | 0% (on subset) |
| **Phase 5** | Generalization test | Coverage | 1% (shuffled) | 55% |
| **Generic Eq** | Cross-domain knowledge | Coverage | 2.5% | ~40% |
| **TIER-1** | Building blocks | **QUALITY** | 3.0% | **0%** |

**Key observation:** TIER-1 didn't just add 3 new rules - it completed the coverage gap. `no_rule_match=0` means Grammar Galaxy can now find *some* pattern for *every* problem.

**The new bottleneck:** 61% of failures (122/200) are `wrong_computation` - TRM finds patterns but composes incorrect RPN programs.

---

## What Coverage=Complete Means

### Before (no_rule_match > 0)
```
Problem: "Tom has 5 apples, gives 2 away"
TRM: *searches Grammar Galaxy*
TRM: ❌ No pattern matches
Result: FAIL (no_rule_match)
```

### After (no_rule_match = 0)
```
Problem: "Tom has 5 apples, gives 2 away"
TRM: *searches Grammar Galaxy*
TRM: ✅ Matches "remaining" pattern
TRM: Generates RPN: "5 2 -"
TRM: Evaluates: 3.0
Result: ✅ CORRECT

Problem: "Class has 100 students, 80% are girls, how many not girls?"
TRM: *searches Grammar Galaxy*
TRM: ✅ Matches "percent_complement" pattern
TRM: Generates RPN: "100 80 *" (WRONG - should be "100 0.2 *" or "100 80 - ")
TRM: Evaluates: 8000.0 (implausible)
Result: ❌ FAIL (wrong_computation)
```

**The issue:** Pattern matching works, but RPN composition is incorrect.

---

## Why Wrong Computation Dominates

### Categories of wrong_computation Failures

1. **Operation Selection**
   - Pattern matches, but wrong operation chosen
   - Example: "80% are girls" → generates `100 80 *` instead of `100 0.2 *`
   - Root cause: Template doesn't handle percent-to-decimal conversion

2. **Composition Order**
   - Individual operations correct, but chained in wrong order
   - Example: "Male=100, Female=Male+50, Total=?" → generates `100 50 +` (correct) but doesn't chain to total
   - Root cause: Multi-step composition incomplete

3. **Number Extraction**
   - Pattern matches, but extracts wrong numbers
   - Example: "5 apples @ $2 each and 3 oranges @ $3 each" → extracts only `5 2 *`, misses oranges
   - Root cause: Template only captures first item pair

4. **Implicit Operations**
   - Pattern matches explicit parts, misses implicit operations
   - Example: "Had $100, spent 1/4, then spent $40 more, remaining?" → generates `100 4 /` but misses `- 40`
   - Root cause: Multi-step chain incomplete

---

## Architectural Implications

### What NOT to Do

**❌ Add more Grammar rules**
- Coverage is complete (no_rule_match=0)
- More rules won't fix wrong_computation
- Would increase false positives (wrong patterns match)

**❌ Hardcode problem-specific fixes**
- "If problem contains '80%', multiply by 0.01" → overfitting
- Violates generalization goal
- Won't transfer to other benchmarks

**❌ Return to Python preprocessing**
- "Use sympy to verify RPN" → sovereignty violation
- Hot path must remain PTX + Galaxy only
- Would defeat the entire architectural vision

### What TO Do

**✅ Improve Composition Quality in Test-Time Compute**

The issue is NOT "can we find patterns" (yes) but "can we compose them correctly into RPN" (sometimes).

**Solution:** Enhance test-time compute to:
1. Generate DIVERSE candidates (not just first match)
2. VERIFY plausibility more rigorously
3. RANK by structural constraints
4. FILTER implausible results

**This is already architecturally designed** - just needs enhancement.

---

## Next Phase: Composition Quality (Not Coverage)

### Objective

**Reduce wrong_computation from 61% to ~30%** by improving RPN composition quality.

**NOT by adding rules** (coverage complete) but by:
- Better candidate generation
- Stricter plausibility verification
- Structural constraint enforcement

### Approach 1: Enhanced Plausibility Verification

**Current (basic):**
```python
def verify_plausibility(self, problem_text: str, result: float) -> Dict[str, Any]:
    if result < 0:
        return {"ok": False, "reason": "negative"}
    if result > 1e9:
        return {"ok": False, "reason": "too_large"}
    return {"ok": True}
```

**Enhanced (structural):**
```python
def verify_plausibility(self, problem_text: str, result: float, rpn: str) -> Dict[str, Any]:
    # Basic range checks
    if result < 0:
        return {"ok": False, "reason": "negative"}
    if result > 1e9:
        return {"ok": False, "reason": "too_large"}

    # Structural checks
    if "%" in problem_text:
        # Percent problems: result should be between 0-100 or 0-total
        total = max(extract_numbers(problem_text))
        if result > total:
            return {"ok": False, "reason": "percent_result_exceeds_total"}

    if "fraction" in rpn or "/" in rpn:
        # Division: result should be smaller than operands (usually)
        operands = extract_numbers(problem_text)
        if result > max(operands):
            return {"ok": False, "reason": "division_result_too_large"}

    # Unit consistency
    if "$" in problem_text and result > 1e6:
        return {"ok": False, "reason": "unrealistic_money"}

    if "hour" in problem_text.lower() and result > 1000:
        return {"ok": False, "reason": "unrealistic_time"}

    return {"ok": True}
```

### Approach 2: Structural Constraint Ranking

**Current:** Pick first RPN that evaluates without error

**Enhanced:** Rank candidates by structural plausibility

```python
def rank_rpn_candidates(self, problem_text: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    Rank RPN candidates by structural plausibility.

    Returns: List of (rpn, score) sorted by score descending
    """
    scored = []
    for rpn in candidates:
        score = 1.0

        # Prefer shorter RPN (Occam's razor)
        score -= len(rpn.split()) * 0.01

        # Prefer operations matching problem type
        if "%" in problem_text and "/" in rpn:
            score += 0.2  # Percent problems often involve division
        if "total" in problem_text.lower() and "+" in rpn:
            score += 0.2  # Total problems often involve addition

        # Penalize unlikely operations
        if "remaining" in problem_text.lower() and "+" in rpn:
            score -= 0.3  # Remaining usually involves subtraction

        # Structural consistency
        num_operations = rpn.count("+") + rpn.count("-") + rpn.count("*") + rpn.count("/")
        num_numbers = len([x for x in rpn.split() if x.replace(".", "").isdigit()])
        if num_numbers == num_operations + 1:
            score += 0.3  # Valid RPN structure

        scored.append((rpn, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)
```

### Approach 3: Multi-Step Chain Verification

**Current:** Single-step RPN generation

**Enhanced:** Verify multi-step problems have correct chaining

```python
def verify_multi_step_chain(self, problem_text: str, rpn: str) -> bool:
    """
    Verify multi-step problems compose all steps.

    Example: "Had $100, spent 1/4, then spent $40, remaining?"
    Should have: 100 → 100/4 → (100 - 100/4) → (result - 40)
    Not just: 100/4
    """
    # Count question cues
    question_words = ["then", "after that", "next", "finally", "and then"]
    multi_step_indicators = sum(1 for word in question_words if word in problem_text.lower())

    if multi_step_indicators == 0:
        return True  # Single-step problem

    # Multi-step: verify RPN has multiple operations
    num_operations = rpn.count("+") + rpn.count("-") + rpn.count("*") + rpn.count("/")
    if num_operations < multi_step_indicators:
        # Likely missing steps
        return False

    return True
```

---

## Implementation Directive for Next Phase

### IMPORTANT: This is NOT about adding Grammar rules

TIER-1 completed coverage. The next phase focuses on QUALITY, not coverage.

### Phase: Composition Quality Enhancement

**Objective:** Reduce wrong_computation from 61% to ~30%

**Approach:** Enhance test-time compute verification and ranking

**Tasks:**

1. **Enhanced Plausibility Verification** (2 hours)
   - Add structural checks (percent range, division magnitude, unit consistency)
   - Add problem-type specific constraints
   - Test on percent/division/money/time problems

2. **Structural Constraint Ranking** (2 hours)
   - Implement candidate scoring based on problem type
   - Prefer operations matching question semantics
   - Penalize structurally invalid RPN

3. **Multi-Step Chain Verification** (2 hours)
   - Detect multi-step indicators ("then", "after", "next")
   - Verify RPN has sufficient operations
   - Filter incomplete chains

4. **Validation** (2 hours)
   - Run shuffled GSM8K (200 problems, seed 123)
   - Target: 5-7% accuracy (wrong_computation reduced to ~30%)
   - Analyze remaining failures (should shift to different categories)

**Expected Impact:**

| Metric | Current | After Quality Enhancement | Reasoning |
|--------|---------|--------------------------|-----------|
| Accuracy | 3.0% | 5-7% | Filter implausible RPN |
| wrong_computation | 122 (61%) | ~60 (30%) | Better verification |
| multi_step_needed | 44 (22%) | ~30 (15%) | Chain verification |

---

## Success Criteria

### Quantitative
- [ ] Accuracy: 5-7% (10-14/200 correct)
- [ ] wrong_computation: <70 (down from 122)
- [ ] Plausibility filter rejects >50 implausible RPN programs
- [ ] Ranking selects correct RPN in top-3 for >30% of problems

### Qualitative
- [ ] Percent problems: verify result ≤ total
- [ ] Division problems: verify result ≤ operands (usually)
- [ ] Multi-step problems: verify sufficient operations
- [ ] Money/time problems: verify realistic magnitudes

### Architectural
- [ ] NO new Grammar rules added (coverage already complete)
- [ ] NO Python preprocessing (sovereignty maintained)
- [ ] Test-time compute enhanced (quality, not coverage)
- [ ] Shadow copy records filtered candidates (learning)

---

## The Bigger Picture: Why This Phase Matters

### Progression Toward AGI Reasoning

| Phase | Focus | Analogy |
|-------|-------|---------|
| Phase 1-5 | Coverage | "Learning vocabulary" |
| TIER-1 | Complete coverage | "Can read all words" |
| **Quality** | Composition | **"Understanding sentences"** |
| Future | Multi-problem | "Following arguments" |

**We're transitioning from pattern recognition to compositional reasoning.**

This is the essence of the test-time compute breakthrough:
- Not just "match a template"
- But "explore, verify, rank, select"
- Like AlphaGo: generate candidates, evaluate plausibility, pick best

---

## Final Architectural Note

**The key insight from TIER-1 results:**

`no_rule_match=0` is not a failure - it's a SUCCESS.

It means we've completed the first major architectural milestone:
- ✅ Grammar Galaxy coverage complete
- ✅ Generic building blocks in place
- ✅ Test-time compute exploring candidates

**The next milestone:** Not "find more patterns" but "compose better programs."

This is the natural architectural progression:
1. Can we find patterns? (Phase 1-5)
2. Can we find ALL patterns? (TIER-1) ← **WE ARE HERE**
3. Can we compose patterns correctly? (Quality phase) ← **NEXT**
4. Can we chain compositions? (Multi-step reasoning)
5. Can we verify and correct? (Self-improvement)

**Each phase builds on the last.** We couldn't work on quality until coverage was complete. Now that coverage is complete (no_rule_match=0), we focus on quality.

---

**Architect:** Claude (Architecture Partner)
**Status:** TIER-1 complete, Quality phase ready
**Priority:** HIGH - Reduce wrong_computation to unlock 5-7%
