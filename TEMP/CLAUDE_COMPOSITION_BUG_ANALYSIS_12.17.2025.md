# Composition Bug Analysis: Data-Driven Fixes

**Date:** December 17, 2025
**Architect:** Claude (Architecture Partner)
**Context:** GSM8K TTC diagnostic results (50 problems, 4.00% accuracy)

---

## Diagnostic Summary

**Accuracy:** 2/50 (4.00%)
**Failures:** wrong_computation=32 (64%), multi_step_needed=8, word_problem=8
**Coverage:** no_rule_match=0 ✅

**Critical insight:** Plausibility filters are WORKING (rejecting bad RPN), but no plausible alternatives are being generated.

---

## Top 3 Composition Bug Categories

### Category 1: relative_chain (18 failures, 36% of failures)

**Problem:** Multi-quantity problems with relative definitions generate incomplete or wrong RPN chains.

#### Example 1: Multi-step relative composition
```
Problem: "50 more female adults than male adults, children were twice the total...
          If 100 male adults, total people at reunion?"
Template: simple_apply
Generated RPN: "100 100 50 + +"
Expected: 750.0
Got: 250.0

CORRECT RPN should be:
  male = 100
  female = male + 50 = 150
  total_adults = male + female = 250
  children = 2 * total_adults = 500
  TOTAL = adults + children = 750

  RPN: "100 100 50 + + 2 * 100 100 50 + + +"
       OR: "100 50 + 100 + dup 2 * +"
```

**Bug:** Template generates `100 100 50 + +` = 250, missing the "children = 2×adults" step and final sum.

#### Example 2: Chain multiplication confusion
```
Problem: "Sammy=20, Gab=twice Sammy, Cher=twice Gab... total?"
Template: test_time_compute
Generated RPN: "20 85 *"
Expected: 55.0 (20 + 40 + 80? or Cher alone = 80?)
Got: 1700.0

CORRECT RPN:
  Sammy = 20
  Gab = 2 * Sammy = 40
  Cher = 2 * Gab = 80
  (need to verify if question asks for total or just Cher)
```

**Bug:** Generates `20 85 *` (where does 85 come from?), completely wrong operation.

#### Example 3: Fraction chain incomplete
```
Problem: "Applied to 1/3, got into half of those..."
Template: extract_operate_aggregate
Generated RPN: "42 3 / 1 /"
Expected: 7.0
Got: 14.0

CORRECT RPN:
  42 / 3 = 14 (first stage)
  14 / 2 = 7 (second stage)
  RPN: "42 3 / 2 /"

GENERATED was: "42 3 / 1 /" = 42/3/1 = 14
```

**Bug:** Second division uses `1` instead of `2` (half).

---

### Category 2: other (6 failures, 12% of failures)

**Problem:** Unit conversions, averages, and rate calculations generate wrong operations.

#### Example 1: Unit conversion with implicit subtraction
```
Problem: "Playlist 1 hour, 16 songs of 3 minutes each... how many 4-minute songs fit?"
Template: extract_operate_aggregate
Generated RPN: "16 3 * 4 /"
Expected: 3.0
Got: 12.0

CORRECT RPN:
  Used time = 16 * 3 = 48 minutes
  Remaining = 60 - 48 = 12 minutes
  Songs fit = 12 / 4 = 3
  RPN: "60 16 3 * - 4 /"

GENERATED: "16 3 * 4 /" = 48/4 = 12
```

**Bug:** Missing the `60 - (16*3)` step (implicit 1 hour = 60 minutes conversion + remaining calculation).

#### Example 2: Rate normalization
```
Problem: "Watches 1/10 per day of 50 episodes... how many days?"
Template: rate_duration
Generated RPN: "50 10 * 1 /"
Expected: 10.0
Got: 500.0

CORRECT RPN:
  Rate = 1/10 per day = 0.1 episodes/day
  Days = 50 / 0.1 = 500? OR
  Days = 50 / (50/10) = 10?

  Interpretation: 1/10 of 50 = 5 per day → 50/5 = 10 days
  RPN: "50 50 10 / /"
```

**Bug:** Generates `50 10 *` (multiply instead of divide).

#### Example 3: Average calculation wrong operation
```
Problem: "Jim 20, Jane 60, Jerry 40... fit into average?"
Template: extract_operate_aggregate
Generated RPN: "20 60 * 40 /"
Expected: 2.0
Got: 30.0

CORRECT RPN:
  Average = (20 + 60 + 40) / 3 = 40
  (question unclear - "fit into average" might mean Jerry/average = 40/40 = 1?)

  Or if asking "how many fit": need more context
```

**Bug:** Generates multiplication `20 60 *` instead of addition for average.

---

### Category 3: cost_divide_instead_aggregate (2 failures, 4% of failures)

**Problem:** Multi-item cost problems use division instead of addition.

#### Example 1: Sum of products becomes division
```
Problem: "35 English books @$7.50 and 35 geography books @$10.50, total?"
Template: extract_operate_aggregate
Generated RPN: "35 10.5 * 7.5 /"
Expected: 630.0
Got: 49.0

CORRECT RPN:
  English cost = 35 * 7.50 = 262.50
  Geography cost = 35 * 10.50 = 367.50
  Total = 262.50 + 367.50 = 630.00
  RPN: "35 7.5 * 35 10.5 * +"

GENERATED: "35 10.5 * 7.5 /" = 367.5 / 7.5 = 49
```

**Bug:** Uses division `/` instead of addition `+` to combine costs.

#### Example 2: Percent complement with division instead of subtraction
```
Problem: "Visit $300 + cast $200, insurance covers 60%, out-of-pocket?"
Template: test_time_compute
Generated RPN: "300 200 / 60 *"
Expected: 200.0
Got: 90.0

CORRECT RPN:
  Total = 300 + 200 = 500
  Covered = 500 * 0.6 = 300
  Out-of-pocket = 500 - 300 = 200
  RPN: "300 200 + 300 200 + 0.6 * -"
  OR: "300 200 + 0.4 *" (40% out-of-pocket)

GENERATED: "300 200 / 60 *" = (300/200) * 60 = 90
```

**Bug:** Uses division `300 200 /` instead of addition `300 200 +`.

---

## Root Cause Analysis

### Why Plausibility Filters Aren't Helping

**The filters ARE working:**
- They correctly reject implausible results (e.g., 8000.0 for "20% of 100")
- They detect structural issues (multi-step incomplete, percent out of bounds)

**But they can't fix the underlying problem:**
- TTC generates candidate RPN: `"100 80 *"`
- Filter rejects: "percent_result_exceeds_total"
- TTC tries next candidate... but ALL candidates have the same bug
- Result: ALL candidates rejected, defaults to first (wrong) answer

**The issue:** Candidate GENERATION is buggy, not just candidate SELECTION.

### Common Composition Bugs

1. **Operation Selection:**
   - `+` vs `*`: "Jim 20, Jane 60" → generates `20 60 *` instead of `20 60 +`
   - `+` vs `/`: "300 + 200" → generates `300 200 /` instead of `300 200 +`

2. **Multi-Step Chain Incompleteness:**
   - "Male 100, Female Male+50, Children 2×Adults" → only generates `100 50 +`, missing subsequent steps

3. **Number Extraction:**
   - "1/10 per day" → extracts `10` instead of computing `0.1`
   - "1 hour" → doesn't convert to `60 minutes`

4. **Implicit Operations:**
   - "Remaining time after 16×3 minute songs" → doesn't include `60 - (16*3)` step

---

## Architectural Implications

### What NOT to Do

**❌ Hardcode problem-specific fixes**
```python
# BAD: Task-specific heuristics
if "twice" in problem_text:
    rpn = "base 2 *"
```
This violates generalization.

**❌ Python symbolic math in hot path**
```python
# BAD: Sovereignty violation
import sympy
rpn = sympy.simplify(expression)
```
This defeats the sovereign architecture.

**❌ External LLM for composition**
```python
# BAD: External dependency
rpn = gpt4("generate RPN for: " + problem_text)
```
This is exactly what we're trying to AVOID.

### What TO Do

**✅ Fix Template RPN Generation (Grammar Rules)**

For patterns that consistently generate wrong operations, fix the Grammar rule `rpn_program`:

```python
# BEFORE (buggy):
GrammarRule(
    rule_id="multi_item_cost",
    pattern=r"(\d+).*@.*(\d+\.?\d*).*and.*(\d+).*@.*(\d+\.?\d*)",
    rpn_program="{0} {1} * {2} /",  # WRONG - uses division
)

# AFTER (fixed):
GrammarRule(
    rule_id="multi_item_cost",
    pattern=r"(\d+).*@.*(\d+\.?\d*).*and.*(\d+).*@.*(\d+\.?\d*)",
    rpn_program="{0} {1} * {2} {3} * +",  # CORRECT - uses addition
)
```

**✅ Enhance TTC Candidate Diversity (Explore Operation Permutations)**

When TTC generates candidates, explore different operations:

```python
def _generate_operation_variants(self, base_numbers: List[float]) -> List[str]:
    """
    Given numbers [A, B, C], generate candidates with different operations.
    """
    candidates = []
    if len(base_numbers) == 2:
        a, b = base_numbers
        candidates.extend([
            f"{a} {b} +",
            f"{a} {b} -",
            f"{a} {b} *",
            f"{a} {b} /",
            f"{b} {a} -",  # reverse order for subtraction
            f"{b} {a} /",  # reverse order for division
        ])
    elif len(base_numbers) == 3:
        a, b, c = base_numbers
        candidates.extend([
            f"{a} {b} + {c} +",
            f"{a} {b} + {c} *",
            f"{a} {b} * {c} +",
            f"{a} {b} * {c} *",
            # ... more permutations
        ])
    return candidates
```

**✅ Plausibility-Guided Candidate Repair**

When plausibility filter rejects a candidate, use the rejection reason to guide repair:

```python
def _repair_candidate(self, rpn: str, rejection_reason: str) -> Optional[str]:
    """
    Attempt to repair RPN based on rejection reason.
    """
    if rejection_reason == "percent_result_exceeds_total":
        # Likely missing percent normalization
        # Try injecting "/100" or changing "*" to "/"
        if "*" in rpn:
            # "100 80 *" → "100 0.8 *" or "100 20 *"
            return rpn.replace(" 80 *", " 0.8 *")

    if rejection_reason == "multi_step_incomplete":
        # Missing operations - try extending chain
        # "100 25 -" → "100 25 - 40 -"
        # (needs context to know what to add)
        pass

    return None
```

---

## Implementation Priority

### Phase 1: Fix relative_chain Templates (Highest Impact - 36% of failures)

**Target:** Reduce relative_chain failures from 18 to <5

**Approach:**

1. **Improve `_generate_relative_chain_candidates()` in TRMGalaxyReader**
   - Current: Generates basic relative patterns
   - Enhanced: Generate full multi-step chains

2. **Add chain composition heuristics:**
   - Detect "twice", "half", "more than", "less than"
   - Build incremental chains: base → derived1 → derived2 → final
   - For "Male=100, Female=Male+50, Children=2×(Male+Female)":
     - Step 1: `100` (male)
     - Step 2: `100 50 +` (female)
     - Step 3: `100 100 50 + +` (total adults)
     - Step 4: `100 100 50 + + 2 *` (children)
     - Step 5: `100 100 50 + + 100 100 50 + + 2 * +` (grand total)

3. **Validation:** Test on the 3 example problems above

**Expected impact:** 18 → 5 failures (13 fewer), accuracy 4% → 7-8%

---

### Phase 2: Fix "other" Category (12% of failures)

**Target:** Reduce "other" failures from 6 to <3

**Approach:**

1. **Add implicit unit conversion candidates:**
   - "1 hour" → inject `60` (minutes) as candidate
   - "1/10 per day" → inject `0.1` as candidate

2. **Fix average calculation template:**
   - Detect "average", "mean", "each person"
   - Generate `sum / count` candidates, not `product / divisor`

3. **Add remaining-time pattern:**
   - "Time left after X" → generate `total - used` candidates

**Expected impact:** 6 → 3 failures (3 fewer), accuracy +1-2%

---

### Phase 3: Fix cost_divide_instead_aggregate (4% of failures)

**Target:** Reduce from 2 to 0

**Approach:**

1. **Fix multi_item_cost Grammar rule:**
   - Change `{0} {1} * {2} /` → `{0} {1} * {2} {3} * +`

2. **Add percent complement proper handling:**
   - "Insurance covers 60%" → generate `total 0.4 *` (out-of-pocket = 40%)
   - Not `total 0.6 *` then manual subtraction

**Expected impact:** 2 → 0 failures (2 fewer), accuracy +0.5-1%

---

## Success Criteria (After All 3 Phases)

### Quantitative
- [ ] Accuracy: 8-12% on shuffled GSM8K (4% → 8-12%)
- [ ] relative_chain failures: 18 → <5
- [ ] other failures: 6 → <3
- [ ] cost_divide: 2 → 0
- [ ] wrong_computation total: 32 → <15

### Qualitative
- [ ] Multi-step relative problems compose full chains
- [ ] Unit conversions use implicit constants (60 min/hour)
- [ ] Average problems use addition, not multiplication
- [ ] Multi-item cost uses addition, not division

### Architectural
- [ ] NO Python preprocessing (fixes in Grammar rules or TTC candidates)
- [ ] Sovereignty maintained (PTX + Galaxy only)
- [ ] Shadow copy records successful compositions

---

## Next Steps for Codex

**Proceed with Phase 1: relative_chain fixes**

**Implementation:**
1. Enhance `_generate_relative_chain_candidates()` in TRMGalaxyReader
2. Add multi-step chain composition logic
3. Test on the 3 relative_chain example problems from diagnostic
4. Run shuffled GSM8K (50 problems) validation
5. Report: relative_chain failures (should drop 18 → <8)

**Expected timeline:** 3-4 hours
**Expected impact:** 4% → 7-8% accuracy

---

**Architect:** Claude (Architecture Partner)
**Status:** Composition bugs identified, fixes prioritized
**Priority:** HIGH - Phase 1 targeting 36% of failures
