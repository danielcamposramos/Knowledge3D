# CODEX DIRECTIVE: Fix relative_chain Composition Bugs

**Date:** December 17, 2025
**Priority:** HIGH
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)

---

## Objective

Reduce `relative_chain` composition failures from **18/50 (36%)** to **<5/50 (10%)**.

**Expected accuracy improvement:** 4% → 7-8%

---

## Problem Analysis

### Current Behavior (BUGGY)

**Example problem:**
```
"There were 50 more female adults than male adults, and children were
twice the total number of adults. If there were 100 male adults, what
was the total number of people at the reunion?"
```

**Current RPN generated:** `"100 100 50 + +"`
**Result:** 250.0
**Expected:** 750.0

**What's missing:**
- Step 3: total_adults = male + female = 250
- Step 4: children = 2 × total_adults = 500
- Step 5: TOTAL = adults + children = 750

---

## Root Cause

`_generate_relative_chain_candidates()` in TRMGalaxyReader generates incomplete chains:
- Detects "50 more" → generates `100 50 +`
- But doesn't continue to "children = twice total adults"
- Result: only partial RPN, missing subsequent steps

---

## Solution Architecture

### Approach: Multi-Step Chain Composition

**Key insight:** Relative chain problems have a DEPENDENCY GRAPH:
```
male (base) = 100
female (derived) = male + 50 = 150
total_adults (derived) = male + female = 250
children (derived) = 2 × total_adults = 500
TOTAL (derived) = adults + children = 750
```

**To generate correct RPN, we need to:**
1. Detect base quantity (100 male)
2. Detect derived quantities (female = male+50)
3. Build chain incrementally (base → derived1 → derived2 → ...)
4. Return complete RPN for final quantity

---

## Implementation Specification

### Location: `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

### Method to Enhance: `_generate_relative_chain_candidates()`

**Current signature:**
```python
def _generate_relative_chain_candidates(
    self,
    problem_text: str,
    understanding: ProblemUnderstanding,
    trace: dict,
    question_type: str,
    max_candidates: int = 9
) -> List[str]:
```

**Enhancement needed:**

```python
def _generate_relative_chain_candidates(
    self,
    problem_text: str,
    understanding: ProblemUnderstanding,
    trace: dict,
    question_type: str,
    max_candidates: int = 9
) -> List[str]:
    """
    Generate RPN candidates for multi-step relative quantity problems.

    Detects patterns like:
    - "X more than Y"
    - "twice the Z"
    - "half of W"

    Builds incremental chains from base to final derived quantity.
    """
    candidates = []

    # Step 1: Extract base quantity (usually has explicit number)
    base_qty, base_value = self._extract_base_quantity(problem_text)
    if base_qty is None:
        return []  # Can't build chain without base

    # Step 2: Extract relative definitions (ordered by dependency)
    # Example: ["female = male + 50", "children = 2 * (male + female)"]
    relative_defs = self._extract_relative_definitions(problem_text, base_qty)

    # Step 3: Build incremental RPN chain
    # Start with base value
    rpn_chain = f"{base_value}"
    intermediate_values = {base_qty: base_value}

    for rel_def in relative_defs:
        # rel_def structure: {"target": "female", "op": "+", "value": 50, "base": "male"}
        target_name = rel_def["target"]
        op = rel_def["op"]
        value = rel_def["value"]
        base_name = rel_def.get("base", base_qty)

        # Append operation to RPN chain
        if op == "+":
            rpn_chain += f" {value} +"
        elif op == "-":
            rpn_chain += f" {value} -"
        elif op == "*":
            rpn_chain += f" {value} *"
        elif op == "/":
            rpn_chain += f" {value} /"

        # For multi-dependency cases (e.g., "children = 2 * (male + female)")
        if "multi_base" in rel_def:
            # Need to compose base quantities first
            # Example: male=100, female=150 → compose to total_adults=250
            rpn_chain += f" {intermediate_values[base_name]} +"

    candidates.append(rpn_chain)

    # Step 4: Generate variants (different operation orders)
    # Example: "twice the total" could be "total 2 *" OR "2 total *"
    for variant in self._generate_chain_variants(rpn_chain):
        candidates.append(variant)

    return candidates[:max_candidates]
```

---

### Helper Methods to Add

#### 1. Extract Base Quantity

```python
def _extract_base_quantity(self, problem_text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract base quantity (the starting point with explicit number).

    Examples:
    - "If there were 100 male adults" → ("male", 100.0)
    - "Sammy has 20 cookies" → ("sammy", 20.0)
    - "Applied to 42 students" → ("students", 42.0)
    """
    import re

    # Pattern: "number + entity"
    patterns = [
        r"(?:If\s+)?(?:there\s+(?:were|are|was)\s+)?(\d+(?:\.\d+)?)\s+(\w+)",
        r"(\w+)\s+(?:has|have|had)\s+(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, problem_text, re.IGNORECASE)
        if match:
            # Check which group is the number
            if match.group(1).replace(".", "").isdigit():
                value = float(match.group(1))
                entity = match.group(2).lower()
            else:
                value = float(match.group(2))
                entity = match.group(1).lower()
            return (entity, value)

    return (None, None)
```

#### 2. Extract Relative Definitions

```python
def _extract_relative_definitions(
    self,
    problem_text: str,
    base_qty: str
) -> List[Dict[str, Any]]:
    """
    Extract relative quantity definitions in dependency order.

    Examples:
    - "50 more female adults than male" → {"target": "female", "op": "+", "value": 50, "base": "male"}
    - "children were twice the total" → {"target": "children", "op": "*", "value": 2, "base": "total"}
    """
    import re
    definitions = []

    # Pattern 1: "X more Y than Z"
    pattern_more = r"(\d+(?:\.\d+)?)\s+more\s+(\w+)\s+than\s+(\w+)"
    for match in re.finditer(pattern_more, problem_text, re.IGNORECASE):
        definitions.append({
            "target": match.group(2).lower(),
            "op": "+",
            "value": float(match.group(1)),
            "base": match.group(3).lower()
        })

    # Pattern 2: "X less Y than Z"
    pattern_less = r"(\d+(?:\.\d+)?)\s+(?:less|fewer)\s+(\w+)\s+than\s+(\w+)"
    for match in re.finditer(pattern_less, problem_text, re.IGNORECASE):
        definitions.append({
            "target": match.group(2).lower(),
            "op": "-",
            "value": float(match.group(1)),
            "base": match.group(3).lower()
        })

    # Pattern 3: "Y were twice the Z"
    pattern_twice = r"(\w+)\s+(?:were|was|is|are)\s+twice\s+(?:the\s+)?(\w+)"
    for match in re.finditer(pattern_twice, problem_text, re.IGNORECASE):
        definitions.append({
            "target": match.group(1).lower(),
            "op": "*",
            "value": 2.0,
            "base": match.group(2).lower()
        })

    # Pattern 4: "Y were N times the Z"
    pattern_times = r"(\w+)\s+(?:were|was|is|are)\s+(\d+(?:\.\d+)?)\s+times\s+(?:the\s+)?(\w+)"
    for match in re.finditer(pattern_times, problem_text, re.IGNORECASE):
        definitions.append({
            "target": match.group(1).lower(),
            "op": "*",
            "value": float(match.group(2)),
            "base": match.group(3).lower()
        })

    # Pattern 5: "Y were half the Z"
    pattern_half = r"(\w+)\s+(?:were|was|is|are)\s+half\s+(?:the\s+)?(\w+)"
    for match in re.finditer(pattern_half, problem_text, re.IGNORECASE):
        definitions.append({
            "target": match.group(1).lower(),
            "op": "/",
            "value": 2.0,
            "base": match.group(2).lower()
        })

    return definitions
```

#### 3. Generate Chain Variants

```python
def _generate_chain_variants(self, base_rpn: str) -> List[str]:
    """
    Generate operation order variants of a chain.

    Example: "100 50 + 2 *" → also try "100 50 + dup 2 *"
    """
    variants = []

    # Variant 1: Use DUP for repeated values
    # "100 100 50 + +" → "100 dup 50 + +"
    if "100 100" in base_rpn:
        variants.append(base_rpn.replace("100 100", "100 dup"))

    # Variant 2: Reverse multiplication order
    # "value 2 *" → "2 value *"
    # (Note: only if semantically equivalent)

    return variants
```

---

## Test Cases

### Test 1: Multi-step relative composition

**File:** `tests/test_math_tier1_building_blocks.py`

```python
def test_relative_chain_multi_step_full_composition(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "There were 50 more female adults than male adults, and children "
                   "were twice the total number of adults. If there were 100 male adults, "
                   "what was the total number of people at the reunion?",
        "answer": "#### 750",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 750.0, f"Expected 750.0, got {result}"
    assert meta.get("template_used") in {"relative_chain", "test_time_compute"}

    # Verify RPN includes all steps
    rpn = meta.get("rpn_program", "")
    num_ops = rpn.count("+") + rpn.count("-") + rpn.count("*") + rpn.count("/")
    assert num_ops >= 3, f"RPN should have >=3 operations for multi-step, got {num_ops}: {rpn}"
```

### Test 2: Chain multiplication

```python
def test_relative_chain_multiplication_series(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "Sammy has 20 cookies. Gab has twice as many as Sammy. "
                   "Cher has twice as many as Gab. How many cookies does Cher have?",
        "answer": "#### 80",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 80.0, f"Expected 80.0, got {result}"
```

### Test 3: Fraction chain

```python
def test_relative_chain_fraction_series(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "42 students applied. 1/3 got accepted. Of those accepted, "
                   "half enrolled. How many enrolled?",
        "answer": "#### 7",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 7.0, f"Expected 7.0, got {result}"
```

---

## Validation

After implementation, run diagnostic again:

```bash
bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --disable-retrieval \
    --shadow-readonly \
    --datasets gsm8k \
    --max-problems 50 \
    --shuffle \
    --shuffle-seed 123 \
    --thinking-budget 8
```

**Success criteria:**
- [ ] relative_chain failures: 18 → <8 (at least 50% reduction)
- [ ] Accuracy: 4% → 7-8% (6-8 correct out of 50)
- [ ] Test cases pass (all 3 above)
- [ ] RPN chains include all steps (verified in meta)

---

## Implementation Notes

### Sovereignty Compliance
- ✅ All logic in Python (TRMGalaxyReader methods)
- ✅ No external libraries (regex only from standard lib)
- ✅ Candidates generated in test-time compute (hot path)
- ❌ Do NOT use sympy, numpy, or external preprocessing

### Architectural Principles
- **Galaxy-First:** Extract patterns from Grammar Galaxy rules (relative_more_than, etc.)
- **Test-Time Compute:** Generate diverse candidates, let plausibility filter select best
- **Shadow Copy:** Record successful chains for future learning

### Edge Cases
- Handle cases where base quantity appears multiple times
- Handle nested dependencies (A → B → C → D)
- Handle ambiguous references ("the total" could mean different totals)

---

## Expected Impact

| Metric | Before | After Phase 1 | Reasoning |
|--------|--------|---------------|-----------|
| Accuracy (50 problems) | 4% (2/50) | 7-8% (6-8/50) | Fix 36% of failures |
| relative_chain failures | 18 | <8 | Multi-step composition |
| wrong_computation total | 32 | ~20 | Chain bugs fixed |

**Next phases:**
- Phase 2: Fix "other" category (unit conversions, averages)
- Phase 3: Fix cost_divide_instead_aggregate (sum of products)

---

## Handoff to Codex

**Codex:** Implement the enhanced `_generate_relative_chain_candidates()` with the 3 helper methods.

**After completion, report:**
1. Test results (3 tests should pass)
2. Shuffled GSM8K diagnostic (50 problems)
3. relative_chain failure count (target: <8)
4. Example RPN chains generated for the 3 test problems

**Then:** Return to Claude for review before Phase 2.

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Targeting 36% of failures
