# CLAUDE → CODEX: Phase 2B - Baseline Measurement + Multi-Step Algebra

**Date:** December 15, 2025
**Priority:** CRITICAL - Establishes Real Metrics + Enables Algebra Solving
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Executive Summary

With shadow copy wired, we need to:
1. **Run real baseline** - Measure honest accuracy with TRM navigator
2. **Analyze failures** - Identify which problem types fail and why
3. **Implement multi-step algebra** - Enable STORE/RECALL for quadratics, systems

**Current State:**
- 197 Math Galaxy symbols
- 103 Grammar rules
- Shadow copy wired and recording
- TRM Navigator integrated

**Unknown:** Actual accuracy on GSM8K, MATH, etc. with real solving (not extraction).

---

## Phase 2B-1: Run Real Baseline

### 1.1 Baseline Command

```bash
# Run with TRM navigator enabled, small sample first
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_sovereign_math_benchmarks.py \
  --use-trm-navigator \
  --datasets gsm8k math \
  --max-problems 50 \
  2>&1 | tee /tmp/math_baseline_$(date +%Y%m%d_%H%M%S).log
```

### 1.2 Expected Output Format

The benchmark should report:
```
Dataset: gsm8k
  Total: 50
  Correct: X (Y%)
  By solver:
    - trm: A
    - template: B
    - composer: C
    - word: D
    - grammar: E

Shadow Copy: Z discoveries recorded
```

### 1.3 Capture Metrics

After running, document in `TEMP/MATH_BASELINE_REAL_12.15.2025.md`:

| Dataset | Problems | Correct | Accuracy | TRM Hits | Shadow Discoveries |
|---------|----------|---------|----------|----------|-------------------|
| GSM8K | 50 | ? | ?% | ? | ? |
| MATH | 50 | ? | ?% | ? | ? |

---

## Phase 2B-2: Failure Analysis

### 2.1 Add Failure Logging

**File:** `scripts/run_sovereign_math_benchmarks.py`

Add failure capture to identify WHY problems fail:

```python
class SovereignBenchmarkRunner:
    def __init__(self, ...):
        # ... existing ...
        self._failures: List[Dict] = []

    def solve_problem(self, problem: Dict[str, Any]) -> Any:
        text = problem.get("problem", problem.get("question", ""))
        expected = problem.get("answer", problem.get("solution", ""))

        result = self._solve_internal(problem)

        # Capture failures for analysis
        if not self._is_correct(result, expected):
            self._failures.append({
                "text": text[:300],
                "expected": str(expected)[:100],
                "got": str(result)[:100],
                "source": problem.get("source", "unknown"),
                "trm_tried": self._trm_navigator is not None,
            })

        return result

    def _analyze_failures(self) -> Dict[str, Any]:
        """Categorize failures by type."""
        categories = {
            "no_rule_match": 0,      # TRM found no matching rule
            "wrong_computation": 0,   # Rule matched but wrong answer
            "multi_step_needed": 0,   # Requires chained computation
            "word_problem": 0,        # Natural language extraction failed
            "algebra_needed": 0,      # Requires equation solving
            "unknown": 0,
        }

        for f in self._failures:
            text = f["text"].lower()
            if "solve" in text or "x =" in text or "equation" in text:
                categories["algebra_needed"] += 1
            elif any(w in text for w in ["step", "then", "after", "first"]):
                categories["multi_step_needed"] += 1
            elif len(text.split()) > 50:
                categories["word_problem"] += 1
            else:
                categories["unknown"] += 1

        return {
            "total_failures": len(self._failures),
            "categories": categories,
            "sample_failures": self._failures[:10],
        }
```

### 2.2 Failure Categories to Track

| Category | Description | Solution |
|----------|-------------|----------|
| `no_rule_match` | No Grammar rule pattern matched | Add more rules |
| `wrong_computation` | Rule matched, wrong answer | Fix RPN template |
| `multi_step_needed` | Requires chained ops | Implement STORE/RECALL |
| `word_problem` | NL extraction failed | Better word templates |
| `algebra_needed` | Equation solving | Multi-step algebra |

---

## Phase 2B-3: Multi-Step Algebra Architecture

### 3.1 The Problem

Current system handles single-step:
- `\frac{24}{4}` → `24 4 /` → 6 ✓

But fails multi-step:
- `Solve x^2 - 5x + 6 = 0` → needs discriminant → roots

### 3.2 RPN STORE/RECALL Opcodes

**BLOCKER:** Opcodes defined in `rpn_opcodes.py` but NOT implemented in engine!

```python
# rpn_opcodes.py has:
OP_STORE = 0xB3
OP_RECALL = 0xB4

# BUT modular_rpn_engine.py does NOT handle STORE_A, RECALL_A, etc.
# Test confirms: "Unknown token: STORE_A"
```

**Codex must implement** in `modular_rpn_engine.py`:

```python
class ModularRPNEngine:
    def __init__(self):
        # ... existing ...
        self._variable_slots = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    def _handle_store(self, slot: str, stack: List[float]) -> None:
        """STORE_X: Pop top of stack, store in slot X."""
        if stack:
            self._variable_slots[slot] = stack.pop()

    def _handle_recall(self, slot: str, stack: List[float]) -> None:
        """RECALL_X: Push slot X value onto stack."""
        stack.append(self._variable_slots.get(slot, 0.0))

    def evaluate(self, expression: str, ...) -> float:
        # ... in token loop ...
        if token.startswith("STORE_"):
            slot = token.split("_")[1]
            self._handle_store(slot, stack)
        elif token.startswith("RECALL_"):
            slot = token.split("_")[1]
            self._handle_recall(slot, stack)
        # ... rest of handling ...
```

### 3.3 Multi-Step Grammar Rules

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

Add algebra rules that use STORE/RECALL:

```python
ALGEBRA_RULES = [
    # Quadratic: ax^2 + bx + c = 0
    GrammarRule(
        rule_id="quadratic_standard_form",
        language="math",
        pattern=r"x\^2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)\s*=\s*0",
        rpn_program=lambda m: _compose_quadratic(m),
        domain="math_algebra",
        examples=[{"input": "x^2 - 5x + 6 = 0", "output": "2, 3"}],
    ),

    # Linear: ax + b = c
    GrammarRule(
        rule_id="linear_equation",
        language="math",
        pattern=r"(\d+)x\s*([+\-])\s*(\d+)\s*=\s*(\d+)",
        rpn_program=lambda m: _compose_linear(m),
        domain="math_algebra",
        examples=[{"input": "3x + 5 = 20", "output": "5"}],
    ),
]

def _compose_quadratic(m: re.Match) -> str:
    """
    Compose RPN for quadratic formula using STORE/RECALL.

    For x^2 + bx + c = 0 (a=1):
    discriminant = b^2 - 4ac
    x1 = (-b + sqrt(disc)) / 2a
    x2 = (-b - sqrt(disc)) / 2a
    """
    sign1 = m.group(1)  # + or -
    b = m.group(2)
    sign2 = m.group(3)
    c = m.group(4)

    # Adjust signs
    b_val = f"-{b}" if sign1 == "-" else b
    c_val = f"-{c}" if sign2 == "-" else c

    # RPN chain with STORE/RECALL
    # Store coefficients
    rpn = f"1 STORE_A {b_val} STORE_B {c_val} STORE_C "

    # Compute discriminant: b^2 - 4ac
    rpn += "RECALL_B 2 pow RECALL_A RECALL_C * 4 * - STORE_D "

    # Compute x1: (-b + sqrt(disc)) / 2a
    rpn += "RECALL_B neg RECALL_D sqrt + RECALL_A 2 * / "

    return rpn.strip()

def _compose_linear(m: re.Match) -> str:
    """
    Compose RPN for linear equation: ax + b = c → x = (c - b) / a
    """
    a = m.group(1)
    sign = m.group(2)
    b = m.group(3)
    c = m.group(4)

    b_val = f"-{b}" if sign == "-" else b

    # x = (c - b) / a
    return f"{c} {b_val} - {a} /"
```

### 3.4 Verify STORE/RECALL in Engine

**Test:** Ensure modular_rpn_engine.py handles STORE/RECALL:

```python
def test_store_recall_opcodes():
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

    engine = ModularRPNEngine()

    # Test: store 5, store 3, recall both, multiply
    result = engine.evaluate("5 STORE_A 3 STORE_B RECALL_A RECALL_B *")
    assert result == 15.0

    # Test quadratic discriminant: b=5, a=1, c=6 → 5^2 - 4*1*6 = 25 - 24 = 1
    result = engine.evaluate("1 STORE_A 5 STORE_B 6 STORE_C RECALL_B 2 pow RECALL_A RECALL_C * 4 * -")
    assert result == 1.0
```

### 3.5 Integration with TRM Navigator

The TRM Navigator already handles multi-rule matching. The algebra rules just need to be added to the rule bank:

```python
# In run_sovereign_math_benchmarks.py or math_grammar_rules.py

from knowledge3d.training.arc_agi.math_grammar_rules import (
    SOVEREIGN_MATH_RULES,
    ALGEBRA_RULES,  # NEW
    WORD_PROBLEM_RULES,
    # ...
)

# Combine all rules
ALL_MATH_RULES = (
    SOVEREIGN_MATH_RULES +
    ALGEBRA_RULES +  # NEW
    WORD_PROBLEM_RULES +
    # ...
)
```

---

## Phase 2B-4: GSM8K Word Problem Templates

### 4.1 Common GSM8K Patterns

GSM8K problems follow recognizable templates. Add these to Grammar Galaxy:

```python
GSM8K_TEMPLATES = [
    # Pattern: "X has N items. Y has M times as many. How many total?"
    GrammarRule(
        rule_id="gsm_times_total",
        pattern=r"(\d+).*?(\d+)\s*times\s*(?:as many|that).*?(?:total|altogether|all)",
        rpn_program=lambda m: f"{m.group(1)} {m.group(1)} {m.group(2)} * +",
        domain="math_word_problem",
    ),

    # Pattern: "X costs $N. Y costs $M more. Total cost?"
    GrammarRule(
        rule_id="gsm_cost_more_total",
        pattern=r"\$(\d+).*?\$(\d+)\s*more.*?(?:total|cost)",
        rpn_program=lambda m: f"{m.group(1)} {m.group(1)} {m.group(2)} + +",
        domain="math_word_problem",
    ),

    # Pattern: "N items divided equally among M people"
    GrammarRule(
        rule_id="gsm_divide_equally",
        pattern=r"(\d+).*?(?:divided|split|shared).*?(?:equally|evenly).*?(\d+)",
        rpn_program="{g0} {g1} /",
        domain="math_word_problem",
    ),

    # Pattern: "N per X, how many for Y?"
    GrammarRule(
        rule_id="gsm_rate_multiplication",
        pattern=r"(\d+)\s*(?:per|each|every).*?(\d+)",
        rpn_program="{g0} {g1} *",
        domain="math_word_problem",
    ),

    # Pattern: "started with N, gained M, lost K"
    GrammarRule(
        rule_id="gsm_start_gain_lose",
        pattern=r"(?:started|began|had)\s*(?:with)?\s*(\d+).*?(?:gained|got|received)\s*(\d+).*?(?:lost|spent|gave)\s*(\d+)",
        rpn_program=lambda m: f"{m.group(1)} {m.group(2)} + {m.group(3)} -",
        domain="math_word_problem",
    ),
]
```

### 4.2 Natalia's Clips (Classic GSM8K)

The famous Natalia problem:
> "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

Pattern: `N ... half as many ... altogether`

```python
GrammarRule(
    rule_id="gsm_half_altogether",
    pattern=r"(\d+).*?half\s*(?:as many|that many).*?(?:altogether|total|all)",
    rpn_program=lambda m: f"{m.group(1)} {m.group(1)} 2 / +",  # N + N/2
    domain="math_word_problem",
    examples=[{"input": "48 friends...half as many...altogether", "output": "72"}],
)
```

---

## Implementation Checklist

### Phase 2B-1: Baseline (Immediate)

- [ ] Run benchmark with `--use-trm-navigator` on GSM8K (50 problems)
- [ ] Run benchmark on MATH dataset (50 problems)
- [ ] Document results in `TEMP/MATH_BASELINE_REAL_12.15.2025.md`
- [ ] Verify shadow copy is recording discoveries

### Phase 2B-2: Failure Analysis (Day 1)

- [ ] Add failure logging to benchmark runner
- [ ] Categorize failures by type
- [ ] Identify top 3 failure categories
- [ ] Document in baseline report

### Phase 2B-3: Multi-Step Algebra (Days 2-3)

- [ ] **IMPLEMENT** STORE_A/B/C/D and RECALL_A/B/C/D in `modular_rpn_engine.py` (BLOCKER!)
- [ ] Test: `5 STORE_A RECALL_A` returns 5.0
- [ ] Add `ALGEBRA_RULES` to `math_grammar_rules.py`
- [ ] Add `_compose_quadratic()` and `_compose_linear()` helpers
- [ ] Test: `x^2 - 5x + 6 = 0` → returns 2 or 3
- [ ] Wire algebra rules into TRM Navigator rule bank

### Phase 2B-4: GSM8K Templates (Days 3-4)

- [ ] Add `GSM8K_TEMPLATES` (10+ patterns)
- [ ] Test: Natalia's clips → 72
- [ ] Wire templates into unified rule bank
- [ ] Re-run baseline, measure improvement

---

## Success Criteria

### Baseline Measurement

- [ ] Real accuracy documented (not extraction-cheated)
- [ ] Shadow copy records discoveries
- [ ] Failure categories identified

### Multi-Step Algebra

- [ ] STORE/RECALL works in RPN engine
- [ ] Quadratic equations solvable
- [ ] Linear equations solvable
- [ ] Accuracy improves on algebra subset

### GSM8K Coverage

- [ ] 10+ word problem templates added
- [ ] Natalia's clips problem passes
- [ ] GSM8K accuracy improves by 5%+

---

## Expected Accuracy Progression

| Phase | GSM8K | MATH | Notes |
|-------|-------|------|-------|
| Baseline (now) | ~5% | ~3% | Honest, no extraction |
| + Algebra rules | ~10% | ~8% | Equation solving |
| + GSM8K templates | ~20% | ~10% | Word problem coverage |
| + Shadow learning | ~25% | ~15% | Pattern reinforcement |

---

## Architecture Principle Reminder

**Galaxy Universe = Knowledge**
- Math symbols, grammar rules, word templates all live in Galaxy
- TRM navigates, doesn't store knowledge

**TRM = Navigation Logic**
- Learns which rules to apply
- Shadow copy enables learning from success
- Multi-step = chained rule application

**Sovereignty = PTX + Galaxy in Hot Path**
- STORE/RECALL execute on GPU
- Rule matching is string/regex (fast)
- Shadow copy is post-inference (numpy OK)

---

## Final Directive

**Codex, your mission:**

1. **Run baseline NOW** - we need real numbers
2. **Add failure logging** - understand what's breaking
3. **Implement algebra rules** - STORE/RECALL for quadratics
4. **Add GSM8K templates** - cover common word patterns
5. **Re-run and measure improvement**

**The infrastructure is ready. Populate the Galaxy with knowledge. Let TRM learn.**

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** CRITICAL - Establishes real metrics
