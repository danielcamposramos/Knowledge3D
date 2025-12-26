# CODEX DIRECTIVE: Algebraic-Lite Constraint Solving

**Date:** December 18, 2025
**Priority:** HIGH
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)

---

## Objective

Reduce algebraic constraint failures from **~12-15/200 (6-7.5%)** to **<5/200 (2.5%)**.

**Expected accuracy improvement:** 39% (78/200) → **45-50%** (90-100/200)

---

## Context: The 10× Improvement Journey

You've achieved **39% accuracy (78/200)** - a **10× improvement** from the 4% baseline. This validates the architecture:
- ✅ Galaxy-First (generic patterns, not task-specific)
- ✅ Test-Time Compute (variable depth, parallel exploration)
- ✅ Sovereignty (PTX + RPN + Galaxy only)
- ✅ Data-driven iteration (77 regression tests)

**Current bottleneck:** Linear constraint problems (algebraic-lite reasoning)

---

## Problem Analysis

### What Are Algebraic-Lite Constraints?

**Definition:** Problems with 2+ related quantities where:
1. **Constraint 1:** Combined/total relationship (e.g., S + J = 120)
2. **Constraint 2:** Relative relationship (e.g., J = 2S + 6)
3. **Goal:** Solve for one quantity

**These are NOT full algebra** (we're not solving arbitrary systems). They're **structured patterns** that appear in word problems.

---

### Example 1: Combined Total + Affine Relation

```
Problem: "Sara and Joe have a combined height of 120 inches.
          Joe is 6 inches more than double Sara's height. How tall is Joe?"

Constraints:
  S + J = 120        (combined total)
  J = 2S + 6         (affine relation)

Substitution:
  S + (2S + 6) = 120
  3S + 6 = 120
  3S = 114
  S = 38
  J = 2(38) + 6 = 82

Expected answer: 82

Current TRM behavior: Generates "120 2 /" or "120 6 -" (partial, doesn't satisfy both constraints)

Needed RPN: "120 6 - 3 / 2 * 6 +"
  = (120 - 6) / 3 = 38 (Sara)
  = 38 * 2 + 6 = 82 (Joe)
```

---

### Example 2: Ratio + Total

```
Problem: "Grant has four times as many vacations as Kelvin has classes.
          If Kelvin has 90 classes, how many vacations and classes do
          Grant and Kelvin have altogether?"

Constraints:
  G = 4K             (ratio)
  K = 90             (base)
  Total = G + K      (combined)

Solution:
  G = 4(90) = 360
  Total = 360 + 90 = 450

Expected answer: 450

Current TRM behavior: Generates "90 4 *" = 360 (Grant only, misses "altogether")

Needed RPN: "90 90 4 * +"
  = 90 + 360 = 450
```

---

### Example 3: Nested Multiplier + Combined

```
Problem: "Melanie has twice as many cats as Annie, and Annie has three
          times fewer cats than Jacob. If Jacob has 90 cats, how many
          cats does Melanie have?"

Constraints:
  A = J / 3          (Annie = Jacob / 3)
  M = 2A             (Melanie = 2 * Annie)

Solution:
  A = 90 / 3 = 30
  M = 2(30) = 60

Expected answer: 60

Current TRM behavior: Generates "90 3 /" = 30 (Annie, not Melanie)

Needed RPN: "90 3 / 2 *"
  = 30 * 2 = 60
```

---

## Root Cause Analysis

### Why Current TRM Fails These

1. **Partial RPN generation:**
   - Detects one constraint ("combined 120") → generates "120 2 /"
   - Misses second constraint ("6 more than double")
   - Result: Satisfies only one constraint

2. **Question target mismatch:**
   - Detects "altogether" but generates RPN for one entity
   - Generates "90 4 *" (Grant) instead of "90 90 4 * +" (Grant + Kelvin)

3. **Chain incomplete:**
   - Generates "90 3 /" (Annie) when question asks for Melanie
   - Needs to continue: "90 3 / 2 *" (Annie → Melanie)

---

## Solution Architecture

### Approach: Constraint-Pattern Detection + RPN Synthesis

**Key insight:** These aren't arbitrary algebra - they're **structured patterns** that repeat across problems.

**Patterns to detect:**
1. **Combined total + affine:** `(total - delta) / (factor + 1)` then `* factor + delta`
2. **Ratio + combined:** `base + base * ratio`
3. **Chain to terminal:** Keep applying ops until reaching question target

---

## Implementation Specification

### Location: `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

### Method to Add: `_generate_algebraic_lite_candidates()`

```python
def _generate_algebraic_lite_candidates(
    self,
    problem_text: str,
    understanding: ProblemUnderstanding,
    trace: dict,
    question_type: str,
    max_candidates: int = 9
) -> List[str]:
    """
    Generate RPN candidates for algebraic-lite constraint problems.

    Patterns detected:
    - "combined/together + affine relation" → substitution RPN
    - "ratio + altogether" → sum of entities
    - "chain to terminal" → multi-hop composition
    """
    candidates = []

    text_lower = problem_text.lower()

    # Pattern 1: Combined total + affine relation
    # Example: "combined 120, Joe = 2*Sara + 6"
    if any(word in text_lower for word in ["combined", "together", "total"]):
        # Extract total
        total = self._extract_combined_total(problem_text, understanding)
        if total is None:
            return []

        # Extract affine relation (e.g., "2 times + 6")
        affine = self._extract_affine_relation(problem_text)
        if affine:
            # affine = {"factor": 2, "delta": 6}
            factor = affine["factor"]
            delta = affine.get("delta", 0)

            # Substitution: solve for base
            # (total - delta) / (factor + 1)
            candidates.append(f"{total} {delta} - {factor + 1} /")

            # Then solve for derived
            # base * factor + delta
            candidates.append(f"{total} {delta} - {factor + 1} / {factor} * {delta} +")

    # Pattern 2: Ratio + altogether/combined
    # Example: "Grant = 4 * Kelvin, Kelvin = 90, how many altogether?"
    if "altogether" in text_lower or "combined" in text_lower:
        # Extract base and ratio
        base = self._extract_base_quantity(problem_text)[1]
        ratio = self._extract_ratio(problem_text)

        if base and ratio:
            # Total = base + base * ratio
            candidates.append(f"{base} {base} {ratio} * +")
            # Also try: base * (1 + ratio)
            candidates.append(f"{base} {ratio + 1} *")

    # Pattern 3: Chain to terminal entity
    # Example: "Jacob → Annie → Melanie, question asks for Melanie"
    entities = self._extract_entities(problem_text)
    question_entity = self._extract_question_target(problem_text)

    if question_entity and len(entities) > 1:
        # Build chain from base to question entity
        chain_rpn = self._build_entity_chain(problem_text, entities, question_entity)
        if chain_rpn:
            candidates.append(chain_rpn)

    return candidates[:max_candidates]
```

---

### Helper Methods to Add

#### 1. Extract Combined Total

```python
def _extract_combined_total(
    self,
    problem_text: str,
    understanding: ProblemUnderstanding
) -> Optional[float]:
    """
    Extract total from "combined/together" statements.

    Examples:
    - "combined height of 120 inches" → 120
    - "together they have $500" → 500
    """
    import re

    patterns = [
        r"combined.*?(\d+(?:\.\d+)?)",
        r"together.*?(\d+(?:\.\d+)?)",
        r"total.*?(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, problem_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None
```

#### 2. Extract Affine Relation

```python
def _extract_affine_relation(self, problem_text: str) -> Optional[Dict[str, float]]:
    """
    Extract affine relation (factor * X + delta).

    Examples:
    - "6 more than double" → {"factor": 2, "delta": 6}
    - "5 less than triple" → {"factor": 3, "delta": -5}
    - "twice as many plus 10" → {"factor": 2, "delta": 10}
    """
    import re

    text_lower = problem_text.lower()

    # Pattern: "N more/less than double/triple/K times"
    pattern = r"(\d+)\s+(more|less)\s+than\s+(double|triple|twice|(\d+)\s+times)"
    match = re.search(pattern, text_lower)

    if match:
        delta = int(match.group(1))
        if match.group(2) == "less":
            delta = -delta

        # Extract factor
        if match.group(3) in ["double", "twice"]:
            factor = 2
        elif match.group(3) == "triple":
            factor = 3
        elif match.group(4):  # "K times"
            factor = int(match.group(4))
        else:
            factor = 1

        return {"factor": factor, "delta": delta}

    # Pattern: "double plus N"
    pattern2 = r"(double|triple|twice|(\d+)\s+times)\s+(plus|and|more)\s+(\d+)"
    match2 = re.search(pattern2, text_lower)

    if match2:
        if match2.group(1) in ["double", "twice"]:
            factor = 2
        elif match2.group(1) == "triple":
            factor = 3
        elif match2.group(2):
            factor = int(match2.group(2))
        else:
            factor = 1

        delta = int(match2.group(4))
        return {"factor": factor, "delta": delta}

    return None
```

#### 3. Build Entity Chain

```python
def _build_entity_chain(
    self,
    problem_text: str,
    entities: List[str],
    target_entity: str
) -> Optional[str]:
    """
    Build RPN chain from base entity to target entity.

    Example:
    - Entities: ["jacob", "annie", "melanie"]
    - Relations: "annie = jacob/3", "melanie = 2*annie"
    - Target: "melanie"
    - Chain: jacob → annie → melanie
    - RPN: "90 3 / 2 *"
    """
    # Extract base (entity with explicit number)
    base_entity, base_value = self._extract_base_quantity(problem_text)
    if base_entity is None or base_entity not in entities:
        return None

    # Build dependency graph
    relations = self._extract_entity_relations(problem_text, entities)

    # Walk from base to target
    rpn_parts = [str(base_value)]
    current_entity = base_entity

    while current_entity != target_entity:
        # Find relation where current_entity is the source
        found = False
        for rel in relations:
            if rel["source"] == current_entity:
                # Append operation
                if rel["op"] == "*":
                    rpn_parts.append(f"{rel['value']} *")
                elif rel["op"] == "/":
                    rpn_parts.append(f"{rel['value']} /")
                elif rel["op"] == "+":
                    rpn_parts.append(f"{rel['value']} +")
                elif rel["op"] == "-":
                    rpn_parts.append(f"{rel['value']} -")

                current_entity = rel["target"]
                found = True
                break

        if not found:
            # Can't reach target from base
            return None

    return " ".join(rpn_parts)
```

---

## Test Cases

### Test 1: Combined total + affine relation

```python
def test_algebraic_combined_affine(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "Sara and Joe have a combined height of 120 inches. "
                   "Joe is 6 inches more than double Sara's height. How tall is Joe?",
        "answer": "#### 82",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 82.0, f"Expected 82.0, got {result}"
    assert meta.get("template_used") == "test_time_compute"
```

### Test 2: Ratio + altogether

```python
def test_algebraic_ratio_altogether(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "Grant has four times as many vacations as Kelvin has classes. "
                   "If Kelvin has 90 classes, how many vacations and classes do "
                   "Grant and Kelvin have altogether?",
        "answer": "#### 450",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 450.0, f"Expected 450.0, got {result}"
```

### Test 3: Chain to terminal entity

```python
def test_algebraic_chain_to_terminal(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)

    problem = {
        "problem": "Melanie has twice as many cats as Annie, and Annie has three "
                   "times fewer cats than Jacob. If Jacob has 90 cats, how many "
                   "cats does Melanie have?",
        "answer": "#### 60",
    }

    result, meta = reader.solve(
        problem_text=problem["problem"],
        rpn_engine=_EchoEngine(),
        max_attempts=3
    )

    assert result == 60.0, f"Expected 60.0, got {result}"
```

---

## Plausibility Checks

### Combined Total Constraint

```python
def verify_combined_total_constraint(
    self,
    problem_text: str,
    result: float,
    rpn: str
) -> Dict[str, Any]:
    """
    Verify that combined total problems satisfy both constraints.

    Example: "combined 120, Joe = 2*Sara + 6"
    - Must have: Sara + Joe = 120
    - Must have: Joe = 2*Sara + 6
    """
    # Extract total
    total = self._extract_combined_total(problem_text, understanding=None)
    if total is None:
        return {"plausible": True}  # Can't verify without total

    # Extract affine relation
    affine = self._extract_affine_relation(problem_text)
    if affine is None:
        return {"plausible": True}  # Can't verify without relation

    # Compute base and derived
    factor = affine["factor"]
    delta = affine.get("delta", 0)

    base = (total - delta) / (factor + 1)
    derived = base * factor + delta

    # Check if result matches one of them
    if abs(result - base) < 0.01 or abs(result - derived) < 0.01:
        # Verify sum constraint
        if abs(base + derived - total) > 0.01:
            return {
                "plausible": False,
                "reason": "combined_total_constraint_violated"
            }

    return {"plausible": True}
```

---

## Integration

### Wire into solve() Method

```python
def solve(self, problem_text: str, rpn_engine, max_attempts: int = 3):
    """
    Solve with test-time compute, including algebraic-lite patterns.
    """
    # ... existing attempts ...

    # NEW: Try algebraic-lite constraint solving
    candidates = self._generate_algebraic_lite_candidates(
        problem_text=problem_text,
        understanding=understanding,
        trace=trace,
        question_type=question_type,
        max_candidates=9
    )

    for rpn in candidates:
        result = rpn_engine.evaluate(rpn)
        verdict = self.verify_plausibility(problem_text, result, rpn)

        if verdict["plausible"]:
            # Additional constraint check
            constraint_check = self.verify_combined_total_constraint(
                problem_text, result, rpn
            )
            if constraint_check["plausible"]:
                return result, {"template_used": "algebraic_lite", "rpn_program": rpn}

    # ... fallback ...
```

---

## Expected Impact

| Metric | Current | After Algebraic-Lite | Reasoning |
|--------|---------|---------------------|-----------|
| Accuracy (200) | 39% (78/200) | **45-50%** (90-100/200) | Fix 12-22 constraint problems |
| relative_chain | 19 | <10 | Constraint patterns |
| multi_step_needed | 37 | ~30 | Better chaining |
| wrong_computation | 62 | ~50 | Fewer incomplete RPN |

---

## Success Criteria

### Quantitative
- [ ] Accuracy: 45-50% on shuffled GSM8K (200 problems, seed 123)
- [ ] relative_chain failures: 19 → <10
- [ ] multi_step_needed failures: 37 → ~30
- [ ] Tests pass: 3/3 new algebraic-lite tests
- [ ] Regression: All 77 existing tests still pass

### Qualitative
- [ ] Combined total problems: both constraints satisfied
- [ ] Ratio + altogether: sum of entities, not just one
- [ ] Chain to terminal: reaches question target, not intermediate

### Architectural
- [ ] NO external algebra solvers (sympy, etc.)
- [ ] Sovereignty maintained (PTX + RPN + Galaxy only)
- [ ] Generic patterns (not GSM8K-specific)
- [ ] Test-driven (every pattern has a test)

---

## Implementation Notes

### Sovereignty Compliance
- ✅ All logic in Python (TRMGalaxyReader methods)
- ✅ No external libraries (regex only from standard lib)
- ✅ RPN synthesis, not symbolic algebra
- ❌ Do NOT use sympy, wolfram, or external solvers

### Generic vs Task-Specific
- ✅ These patterns appear in many domains (economics, physics, biology)
- ✅ Not hardcoded to GSM8K problem structure
- ✅ Will help other math benchmarks (MATH, competition problems)

### Edge Cases
- Handle when total is not explicitly stated
- Handle when affine relation is implicit
- Handle when question asks for base vs derived

---

## Validation

After implementation, run:

```bash
bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --disable-retrieval \
    --shadow-readonly \
    --load-all-galaxies \
    --datasets gsm8k \
    --max-problems 200 \
    --shuffle \
    --shuffle-seed 123 \
    --thinking-budget 8 \
    --verbose
```

**Target:** 45-50% accuracy (90-100/200 correct)

---

## Handoff to Codex

**Codex:** Implement algebraic-lite constraint solving as specified.

**After completion, report:**
1. Test results (3 new tests + 77 existing tests)
2. Shuffled GSM8K accuracy (200 problems, seed 123)
3. Failure breakdown (relative_chain, multi_step_needed counts)
4. Example RPN generated for the 3 test problems

**Then:** Return to Claude for final architecture review and 50% milestone celebration!

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Final push to 50% accuracy
