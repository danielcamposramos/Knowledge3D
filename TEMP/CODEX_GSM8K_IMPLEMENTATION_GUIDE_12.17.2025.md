# GSM8K Building Blocks Implementation Guide for Codex

**Date:** December 17, 2025
**Status:** Architectural Spec (Claude → Codex handoff)
**Target:** Add 3 TIER-1 generic operations to unlock 5-8% accuracy

---

## Overview

The diagnostic identified **8 missing generic mathematical operations** blocking GSM8K solving. This doc focuses on **3 TIER-1 operations** (42% of failures) that are:
- **Generic** (not task-specific heuristics)
- **Independently testable**
- **Compositionally clear** (well-defined RPN templates)

---

## TIER 1 Implementation Tasks

### Task 1: Add `percent_complement` Generic Equation

**Location:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/knowledge3d/training/math_benchmarks/math_templates.py`

**What it does:** Given a total and a percentage share, compute the non-percentage portion.

**Example Problem:**
```
Text: "Megan was lead actress in 80% of 100 plays. How many NOT lead?"
Expected: 20.0
```

**RPN Form:**
```
Option A: quantity 1.0 percent_share - *
  100 1.0 0.8 - *  →  100 * (1 - 0.8) = 20

Option B: quantity percent_share * quantity swap -
  100 0.8 * 100 -  →  100 - 80 = 20
```

**Implementation Checklist:**

```python
# In math_templates.py, add to get_all_templates():

GrammarRule(
    rule_id="percent_complement_subtract",
    pattern=r"(\d+(?:\.\d+)?)\s*(?:of\s+)?(?:the\s+)?(?:plays|items|tasks|people|time|work|cases)\s*(?:was|were)?\s*(\d+)%.*?(?:not|wasn't|weren't|failed|unsuccessful)",
    rpn_program="{0} {1} 100 / - *",  # quantity * (1 - percent/100)
    rule_type="arithmetic",
    domain="percent_operations",
)

GrammarRule(
    rule_id="percent_complement_direct",
    pattern=r"(\d+)%\s+(?:of\s+)?(\d+).*?(?:not|remaining|left|non-|un-|failed|unsuccessful)",
    rpn_program="{1} {0} 100 / 1 swap - *",  # quantity * (1 - percent/100)
    rule_type="arithmetic",
    domain="percent_operations",
)
```

**Test Case:**
```python
# Expected test in tests/test_math_*.py
def test_percent_complement():
    runner = SovereignBenchmarkRunner()

    # Test 1: Direct form
    problem = {
        "problem": "Megan was lead actress in 80% of 100 plays. How many NOT lead?",
        "answer": "#### 20",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 20.0, f"Expected 20, got {result}"

    # Test 2: Remaining form
    problem = {
        "problem": "80% of tasks successful. 100 tasks total. How many failed?",
        "answer": "#### 20",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 20.0, f"Expected 20, got {result}"
```

**Validation:**
- Does extracted RPN match expected form?
- Does PTX engine evaluate to correct answer?
- Does pattern trigger on sample problems?

---

### Task 2: Add `multi_item_cost_aggregation` Generic Equation

**Location:** Same file as Task 1 (`math_templates.py`)

**What it does:** Compose multiple item types with distinct unit costs into a single sum RPN.

**Example Problem:**
```
Text: "Order 35 English books @ $10.50 and 35 geography books @ $7.50. Total cost?"
Expected: 630.0
```

**RPN Form:**
```
35 10.50 *        # first_item_cost = 367.50
35 7.50 *         # second_item_cost = 262.50
+                 # total = 630.00
```

**Implementation Checklist:**

```python
# In math_templates.py, add to get_all_templates():

GrammarRule(
    rule_id="multi_item_cost_sum",
    pattern=r"(\d+)\s+(?:books?|items?|packs?|boxes?)\s+(?:at|@|costs?)\s+\$?([\d.]+).*?(?:and|also)\s+(\d+)\s+(?:books?|items?|packs?|boxes?)\s+(?:at|@|costs?)\s+\$?([\d.]+)",
    rpn_program="{0} {1} * {2} {3} * +",  # qty1*cost1 + qty2*cost2
    rule_type="arithmetic",
    domain="aggregation",
)

# For problems with implicit context (e.g., "textbooks cost X, Y books cost Z")
GrammarRule(
    rule_id="multi_item_cost_sum_context",
    pattern=r"(?:ordering?|buying?|purchasing?)\s+(\d+)\s+(\w+).*?costs?\s+\$?([\d.]+).*?and\s+(\d+)\s+(\w+).*?costs?\s+\$?([\d.]+)",
    rpn_program="{0} {2} * {3} {5} * +",  # qty1*cost1 + qty2*cost2
    rule_type="arithmetic",
    domain="aggregation",
)
```

**Test Case:**
```python
def test_multi_item_cost():
    runner = SovereignBenchmarkRunner()

    # Test 1: Explicit "books at X, books at Y"
    problem = {
        "problem": "Ali's class orders 35 English books @ $10.50 and 35 geography books @ $7.50. Total cost?",
        "answer": "#### 630",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 630.0, f"Expected 630, got {result}"

    # Test 2: Implicit costs
    problem = {
        "problem": "Order 10 apples at $2 and 5 oranges at $3. How much?",
        "answer": "#### 35",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 35.0, f"Expected 35, got {result}"
```

**Validation:**
- Does pattern capture all item groups?
- Does RPN compose costs correctly?
- Does evaluation match expected total?

---

### Task 3: Add Relative Quantity Chain Detection to TRM

**Location:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/knowledge3d/training/math_benchmarks/trm_math_navigator.py`

**What it does:** Detect patterns where quantity X is defined in terms of Y, Y in terms of Z; build chained RPN.

**Example Problem:**
```
Text: "There were 50 more female adults than male adults, and children were
      twice the total adults. If 100 male adults, total people at reunion?"
Expected: 750.0
```

**Breakdown:**
- male = 100
- female = male + 50 = 150
- total_adults = 100 + 150 = 250
- children = 2 * 250 = 500
- TOTAL = 250 + 500 = 750

**RPN (chained):**
```
100 (male)
100 50 + (female = male + 50)
100 50 + 100 + (total_adults = male + female)
2 100 50 + 100 + * (children = 2 * total_adults)
100 50 + 100 + 2 100 50 + 100 + * + (TOTAL)
```

**Implementation Checklist:**

```python
# In trm_math_navigator.py, add new routing rule:

class RelativeQuantityChainRule:
    """Detect and compose relative quantity definitions."""

    rule_id = "relative_quantity_chain"

    @staticmethod
    def detect(text: str) -> bool:
        """Check if problem has relative definitions (X is ... Y, Y is ...)"""
        patterns = [
            r"(\w+)\s+(?:is|was|were|had)\s+(\d+)\s+(?:more|less|times?|half)",
            r"(\w+)\s+(?:is|was|were|had)\s+(?:twice|thrice|half)\s+(?:the|as many)",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    @staticmethod
    def compose(text: str, quantities: Dict[str, Any]) -> Optional[str]:
        """Build chained RPN from dependent quantity definitions."""
        # Extract base quantity (e.g., "male = 100")
        base_pattern = r"(?:If\s+)?(\d+)\s+(\w+)"
        m = re.search(base_pattern, text)
        if not m:
            return None

        base_value = float(m.group(1))
        base_name = m.group(2).lower()

        rpn = f"{base_value}"  # Start with base

        # For each derived quantity, append operation
        # This is simplified; real implementation needs more robust parsing

        if "50 more female" in text:
            rpn += f" 50 +"  # female = base + 50

        if "twice" in text and "children" in text.lower():
            # children = 2 * (previous result)
            # Need to be more careful with order of operations
            pass

        return rpn
```

**Better approach: Implement a composition engine**

Instead of hardcoding, create a generic composition engine:

```python
# In trm_math_navigator.py or new file trm_relative_composer.py

def compose_relative_chain(text: str) -> Optional[str]:
    """
    Extract relative quantity definitions and build composite RPN.

    Input: "Male=100, Female=Male+50, Children=2*Total Adults, Total=?"
    Output: RPN expression that defines each quantity from base upward
    """
    # Step 1: Extract all quantity assignments
    assignments = {}
    base_qty = None

    # Pattern: "quantity_name = number" or "quantity_name = other_qty operator number"
    for match in re.finditer(r'(\w+)\s*(?:is|was|were|had|=)\s*([\d.]+|\w+)\s*(plus|minus|times|divided|half)?', text):
        qty_name = match.group(1).lower()
        value_or_ref = match.group(2)
        operator = match.group(3)

        if value_or_ref.isdigit() or (value_or_ref[0].isdigit()):
            # Base assignment: qty = number
            assignments[qty_name] = ('base', float(value_or_ref), None)
            base_qty = qty_name
        else:
            # Derived assignment: qty = other_qty operator number
            assignments[qty_name] = ('derived', value_or_ref, operator)

    # Step 2: Build RPN by topological sort (base first, then dependencies)
    rpn_parts = []
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)

        if name not in assignments:
            return

        asgn_type, value_or_ref, op = assignments[name]

        if asgn_type == 'base':
            rpn_parts.append(str(value_or_ref))
        else:
            # Visit dependency first
            visit(value_or_ref)
            # Then apply operator
            if op and value_or_ref:
                rpn_parts.append(str(value_or_ref))
                if op == 'plus' or op == '+':
                    rpn_parts.append('+')
                # ... handle other operators

    # Visit all quantities
    for qty_name in assignments.keys():
        visit(qty_name)

    return ' '.join(rpn_parts)
```

**Test Case:**
```python
def test_relative_chain():
    runner = SovereignBenchmarkRunner()

    problem = {
        "problem": "There were 50 more female adults than male adults, and children "
                   "were twice the total number of adults. If there were 100 male adults, "
                   "what was the total number of people at the reunion?",
        "answer": "#### 750",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 750.0, f"Expected 750, got {result}"

    # Verify RPN was built correctly
    assert "relative_quantity_chain" in trace.get("rule_used", ""), \
        "Rule should be identified as relative_quantity_chain"
```

**Validation:**
- Does detection trigger on relative definitions?
- Does composition build correct RPN?
- Does RPN evaluate to expected answer?
- Does it NOT trigger on simple problems (false positives)?

---

## Integration Points

### 1. In `math_templates.py`

**Add to `get_all_templates()`:**
```python
def get_all_templates():
    """Return all curated parametric templates."""
    rules = [
        # ... existing templates ...

        # TIER 1 additions
        GrammarRule(rule_id="percent_complement_...", ...),
        GrammarRule(rule_id="multi_item_cost_sum", ...),

        # ... other rules ...
    ]
    return rules
```

### 2. In `trm_math_navigator.py`

**Add to routing logic:**
```python
def solve(self, text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """Solve using TRM navigation."""

    # ... existing attempts ...

    # NEW: Try relative quantity chain (Tier 1, Task 3)
    if RelativeQuantityChainRule.detect(text):
        result, meta = self._try_relative_chain(text)
        if result is not None:
            return result, meta

    # ... fallback attempts ...
```

### 3. In test suite

**Create new test file** or add to existing:
```
tests/test_math_tier1_building_blocks.py
```

With tests for:
- `test_percent_complement_*` (Task 1)
- `test_multi_item_cost_*` (Task 2)
- `test_relative_quantity_chain_*` (Task 3)

---

## Success Criteria

### For Each Task:

**Task 1 (Percent Complement):**
- [ ] Pattern correctly extracts percent and total quantity
- [ ] RPN evaluates to (1 - percent) * quantity
- [ ] At least 2 sample GSM8K problems solve correctly
- [ ] False positive rate < 10% (doesn't trigger on unrelated percents)

**Task 2 (Multi-Item Cost):**
- [ ] Pattern captures all N item groups with distinct costs
- [ ] RPN composes as sum of (qty * cost) pairs
- [ ] At least 2 sample GSM8K problems solve correctly
- [ ] Handles both explicit ("@ $X") and implicit cost markers

**Task 3 (Relative Chain):**
- [ ] Detection correctly identifies relative definitions
- [ ] Composition builds ordered RPN (base quantities first)
- [ ] At least 3 sample GSM8K problems solve correctly
- [ ] Doesn't trigger on simple additive problems

### Overall:
- [ ] Diagnostic re-run on 50 problems reaches **5-8% accuracy** (2-4 correct)
- [ ] No regressions (templates/grammar rules still work)
- [ ] Code is sovereignty-compliant (no numpy/cupy in hot path)

---

## Debugging Tips

### Testing Individual Rules:

```python
# Minimal test harness
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

def test_rule_directly(rule_pattern, test_text, expected_rpn):
    import re
    m = re.search(rule_pattern, test_text, re.IGNORECASE | re.DOTALL)
    if not m:
        print(f"Pattern didn't match: {rule_pattern}")
        return False

    engine = ModularRPNEngine()
    result = engine.evaluate(expected_rpn)
    print(f"RPN '{expected_rpn}' evaluated to {result}")
    return result is not None
```

### Checking Sovereign Compliance:

```bash
# In repository root, check for forbidden imports
grep -r "import numpy\|import cupy\|from numpy\|from cupy" \
    knowledge3d/training/math_benchmarks/math_templates.py \
    knowledge3d/training/math_benchmarks/trm_math_navigator.py
```

Should return ZERO matches in hot path code.

---

## Related Documentation

- Full analysis: `TEMP/CLAUDE_GSM8K_FAILURE_ANALYSIS_12.17.2025.md`
- Galaxy architecture: `BRIEFING.md` (Galaxy Universe section)
- TRM design: `CLAUDE.md` (TRM = Learned Navigation)
- Math core: `docs/vocabulary/MATH_CORE_SPECIFICATION.md`

---

## Handoff Summary for Codex

You have **3 well-scoped implementation tasks**, each with:
1. Clear problem examples
2. Expected RPN format
3. Code locations
4. Test cases
5. Success criteria

**Estimated effort:**
- Task 1: 1-2 hours (straightforward pattern + template rule)
- Task 2: 1-2 hours (pattern extraction + RPN composition)
- Task 3: 2-3 hours (relative chain detection + composition engine)

**Expected impact:**
- Unlock ~42% of current failures (21/50 problems)
- Target: 5-8% accuracy (2-4 correct on next diagnostic run)

All operations are **generic** (reusable across benchmarks) and **independently testable**.

Good luck! 🚀
