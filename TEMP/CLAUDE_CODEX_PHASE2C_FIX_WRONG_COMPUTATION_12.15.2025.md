# CLAUDE → CODEX: Phase 2C - Fix wrong_computation Failures

**Date:** December 15, 2025
**Priority:** HIGH - 50% of failures are wrong_computation
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Problem Statement

Baseline shows **50%+ of failures** are `wrong_computation`:
- Rule pattern matches
- RPN program executes
- Answer is WRONG

**Root Cause:** Regex captures wrong numbers from problem text.

Example:
```
Problem: "John has 15 apples. He gives 3 to Mary and 5 to Tom. How many left?"
Pattern: r"(\d+).*?(\d+).*?(\d+)"
Captures: (15, 3, 5) ✓
But RPN might be: "15 3 +" instead of "15 3 - 5 -"
```

---

## Solution Architecture

### 1. Diagnostic: Capture Which Rules Fail

**File:** `scripts/run_sovereign_math_benchmarks.py`

Add detailed failure logging:

```python
def _log_failure_detail(self, problem, result, expected, rule_used, rpn_program):
    """Log detailed failure info for diagnosis."""
    self._failure_details.append({
        "problem_text": problem.get("problem", problem.get("question", ""))[:200],
        "expected": str(expected)[:50],
        "got": str(result)[:50],
        "rule_used": rule_used,
        "rpn_program": rpn_program[:100],
        "source": problem.get("source", "unknown"),
    })
```

**Output:** After run, print top-10 failing rules:
```
Top failing rules:
  1. gsm_times_total: 8 failures (captures wrong multiplier)
  2. gsm_half_altogether: 5 failures (wrong base number)
  3. latex_frac: 4 failures (captures irrelevant numbers)
```

### 2. Pattern Specificity Improvements

**Problem:** Generic patterns capture wrong numbers.

**Current (too greedy):**
```python
GrammarRule(
    rule_id="gsm_times_total",
    pattern=r"(\d+).*?(\d+)\s*times",  # Captures ANY two numbers
    rpn_program=lambda m: f"{m.group(1)} {m.group(1)} {m.group(2)} * +",
)
```

**Fixed (context-aware):**
```python
GrammarRule(
    rule_id="gsm_times_total_v2",
    pattern=r"(?:has|sold|made)\s+(\d+).*?(\d+)\s*times\s*(?:as many|that)",
    rpn_program=lambda m: f"{m.group(1)} {m.group(1)} {m.group(2)} * +",
    # More specific: requires "has/sold/made N ... M times as many"
)
```

### 3. Number Context Rules

Add context-aware number extraction:

```python
# Pattern that captures number WITH its context
GSM8K_CONTEXTUAL_PATTERNS = [
    # "X has N items" - base quantity
    GrammarRule(
        rule_id="gsm_base_quantity",
        pattern=r"(\w+)\s+(?:has|had|owns|bought|sold)\s+(\d+)\s+(\w+)",
        rpn_program=lambda m: f"{m.group(2)}",  # Just extract the number
        domain="math_extraction",
        metadata={"entity": "{g0}", "quantity": "{g1}", "unit": "{g2}"},
    ),

    # "gives/gave N to Y" - subtraction
    GrammarRule(
        rule_id="gsm_gives_subtract",
        pattern=r"(?:gives?|gave)\s+(\d+)\s+(?:to|away)",
        rpn_program=lambda m: f"{m.group(1)} -",
        domain="math_operation",
    ),

    # "receives/gets N" - addition
    GrammarRule(
        rule_id="gsm_receives_add",
        pattern=r"(?:receives?|gets?|gains?)\s+(\d+)",
        rpn_program=lambda m: f"{m.group(1)} +",
        domain="math_operation",
    ),
]
```

### 4. Multi-Pattern Composition

For complex problems, compose multiple rule matches:

```python
class CompositeRuleMatcher:
    """Match multiple patterns and compose RPN from all matches."""

    def solve(self, problem_text: str) -> str:
        # Step 1: Find base quantity
        base_match = self._match_pattern("gsm_base_quantity", problem_text)
        if not base_match:
            return None

        rpn_parts = [base_match.group(2)]  # Start with base number

        # Step 2: Find all operations
        for op_pattern in ["gsm_gives_subtract", "gsm_receives_add"]:
            for match in self._find_all_matches(op_pattern, problem_text):
                rpn_parts.append(match.rpn_fragment)

        return " ".join(rpn_parts)
```

### 5. Validation Before Return

Add answer sanity checking:

```python
def _validate_answer(self, result, problem_text):
    """Basic sanity checks on computed answer."""
    if result is None:
        return False

    # Check for NaN/Inf
    if math.isnan(result) or math.isinf(result):
        return False

    # GSM8K answers are typically positive integers
    if "gsm" in problem_text.lower():
        if result < 0:
            return False  # GSM8K rarely has negative answers
        if result > 1000000:
            return False  # Suspiciously large

    return True
```

---

## Implementation Plan

### Phase 2C-1: Diagnostic (Day 1)

1. Add `_failure_details` list to benchmark runner
2. Log rule_used + rpn_program for each failure
3. Print top-10 failing rules after run
4. Identify which patterns need fixing

### Phase 2C-2: Fix Top Failing Patterns (Days 2-3)

For each top-failing rule:
1. Analyze WHY it captures wrong numbers
2. Add context keywords (has, gives, etc.)
3. Make pattern more specific
4. Test on failing examples

### Phase 2C-3: Contextual Extraction (Day 4)

1. Add `GSM8K_CONTEXTUAL_PATTERNS`
2. Implement `CompositeRuleMatcher` for multi-pattern problems
3. Wire into TRM Navigator

### Phase 2C-4: Validation (Day 5)

1. Add `_validate_answer()` sanity checks
2. Reject obviously wrong results
3. Fall back to next solver on validation failure

---

## Example Fixes

### Fix 1: "half as many" Pattern

**Current (failing):**
```python
pattern=r"(\d+).*?half"
# Problem: "In April, 48 friends. In May, half as many."
# Captures: 48 ✓ but might miss context
```

**Fixed:**
```python
pattern=r"(?:sold|made|had)\s+(?:\w+\s+)?(\d+).*?half\s+(?:as many|that many).*?(?:altogether|total|all)"
# Requires: verb + number + "half as many" + aggregation word
```

### Fix 2: Percentage Pattern

**Current (failing):**
```python
pattern=r"(\d+)%\s*of\s*(\d+)"
# Problem: "25% of the 80 students passed. 15% got A."
# Might capture: (25, 80) or (15, 80) randomly
```

**Fixed:**
```python
pattern=r"(\d+)%\s*of\s*(?:the\s+)?(\d+)\s+(\w+)"
# Captures: percentage, base, unit
# Returns: "25% of 80 students" not just numbers
```

### Fix 3: Division Context

**Current (failing):**
```python
pattern=r"(\d+).*?divided.*?(\d+)"
# Too greedy - captures ANY two numbers near "divided"
```

**Fixed:**
```python
pattern=r"(\d+)\s+(?:\w+\s+)?(?:divided|split|shared)\s+(?:equally\s+)?(?:among|between|into)\s+(\d+)"
# Requires: N [items] divided [equally] among M
```

---

## Success Criteria

### Metrics

| Metric | Before | Target |
|--------|--------|--------|
| wrong_computation % | 50% | <30% |
| GSM8K accuracy | 6% | 15% |
| MATH accuracy | 2% | 8% |

### Tests

```python
def test_half_altogether_pattern():
    """Natalia's clips: 48 + 48/2 = 72"""
    result = solve("Natalia sold clips to 48 friends in April, "
                   "half as many in May. How many altogether?")
    assert result == 72

def test_gives_subtract():
    """15 - 3 - 5 = 7"""
    result = solve("John has 15 apples. Gives 3 to Mary, 5 to Tom. How many left?")
    assert result == 7

def test_percentage_of():
    """25% of 80 = 20"""
    result = solve("25% of 80 students passed.")
    assert result == 20
```

---

## Files to Modify

1. `scripts/run_sovereign_math_benchmarks.py`
   - Add `_failure_details` logging
   - Add `_validate_answer()` sanity check
   - Print top-failing rules report

2. `knowledge3d/training/arc_agi/math_grammar_rules.py`
   - Fix top-failing patterns with better specificity
   - Add `GSM8K_CONTEXTUAL_PATTERNS`
   - Add composite pattern helpers

3. `knowledge3d/training/math_benchmarks/trm_math_navigator.py`
   - Add validation before returning result
   - Try next rule on validation failure

---

## Architecture Principle

**Galaxy Universe stores patterns. TRM learns WHICH patterns to use.**

By improving pattern specificity:
- TRM has better rules to choose from
- Shadow copy records successful rule applications
- Pattern confidence calibrates routing

**The fix is in the GALAXY (better rules), not in the TRM.**

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Biggest accuracy blocker
