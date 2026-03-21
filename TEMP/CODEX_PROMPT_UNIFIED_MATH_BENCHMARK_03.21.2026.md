# Codex Prompt: Unified Math Benchmark — Single Mind Fix

**Date:** March 21, 2026
**Architecture:** Claude (diagnosis + spec) → Codex (implementation)
**Priority:** CRITICAL — Math benchmark 0/500 while GSM8K 30/1319 on the SAME sovereign engine

---

## The Problem

Math and GSM8K benchmarks both call `kv.execute_task(type="MATH_TASK", specialist="math")` — the sovereign engine IS one mind. But they're TWO separate Python classes with duplicated logic and, critically, **different answer comparison**:

- **GSM8K** answers: plain integers via `#### 18` → `_to_float("18")` → works
- **MATH** answers: LaTeX via `\boxed{\frac{3}{4}}` → `_to_float("\frac{3}{4}")` → **FAILS**

Even when the sovereign engine computes the correct numeric result, it can NEVER match a LaTeX expected answer. This is why Math = 0/500.

**Daniel's directive:** "This AI should have a single mind. Why is math 0 and GSM8K not? Clearly they are solving math with different paths, when they should be a single math problem path combining the two benchmark questioning styles."

---

## Root Cause Analysis

### 1. LaTeX Answer Extraction Doesn't Normalize

`math_competitions.py:717` `_extract_math_answer()` extracts from `\boxed{}` but returns raw LaTeX:
- `\boxed{0}` → `"0"` ✓ (happens to be numeric)
- `\boxed{\frac{3}{4}}` → `"\frac{3}{4}"` ✗ (not parseable to float)
- `\boxed{2\sqrt{3}}` → `"2\sqrt{3}"` ✗
- `\boxed{\begin{pmatrix} 1 \\ 2 \end{pmatrix}}` → matrix string ✗

### 2. Two Separate Benchmark Classes

`MathCompetitionBenchmark` and `GSM8KBenchmark` duplicate:
- `_to_float()` (slightly different implementations)
- `_answers_match()` (different text normalization)
- `_apply_query_scope()` (identical copy)
- `_seed_math_knowledge()` (similar but different entry kinds)
- `_normalize_query_scope()` (identical copy)

Both call `execute_task(type="MATH_TASK")` with the same route. They ARE one mind at the engine level — but the benchmark wrappers fragment the experience.

### 3. No LaTeX → Numeric Normalization

There is NO function that converts LaTeX math notation to comparable numeric values:
- `\frac{a}{b}` → `a/b`
- `\sqrt{n}` → `n**0.5`
- `\pi` → `3.14159...`
- `a\sqrt{b}` → `a * b**0.5`
- `\left(\frac{a}{b}\right)` → `a/b`
- `\dfrac{a}{b}` → `a/b`

---

## The Fix: Three Phases

### Phase 1: LaTeX Answer Normalizer (CRITICAL — fixes 0/500 immediately)

Create a `_normalize_latex_answer(text: str) -> str` function that converts common LaTeX patterns to evaluatable numeric strings. This is **ingestion-path** (Python answer comparison), NOT hot-path (sovereignty preserved).

**Must handle these patterns** (in order of frequency in MATH dataset):

```python
# Fractions
r"\frac{a}{b}" → "a/b" (then eval to float)
r"\dfrac{a}{b}" → "a/b"
r"\tfrac{a}{b}" → "a/b"
# Nested fractions: \frac{\frac{1}{2}}{3} → "(1/2)/3"

# Square roots
r"\sqrt{n}" → "n**0.5"
r"\sqrt[3]{n}" → "n**(1/3)"
r"a\sqrt{b}" → "a*b**0.5"

# Constants
r"\pi" → "3.141592653589793"
r"e" (as Euler's number in context) — careful, only when standalone

# Expressions
r"\left(" and r"\right)" → "(" and ")"
r"\cdot" → "*"
r"\times" → "*"
r"\div" → "/"

# Cleanup
r"\," r"\;" r"\!" r"\ " → "" (LaTeX spacing)
r"\text{...}" → strip
r"\mathrm{...}" → strip
r"\operatorname{...}" → strip

# Integer extraction
r"\boxed{42}" → "42" (already works, but normalize first)
```

**After normalization**, attempt `eval()` on the cleaned string in a restricted namespace (only `__builtins__: {}` + math constants). If eval succeeds, compare numerically. If not, fall through to text comparison.

**IMPORTANT:** Use `ast.literal_eval` or a restricted `eval` with NO builtins for safety. This is answer comparison, not arbitrary code execution.

```python
import ast
import math

SAFE_MATH_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "inf": math.inf,
}

def _safe_eval_math(expr: str) -> float | None:
    """Evaluate a simple math expression safely."""
    try:
        # Only allow basic arithmetic on numbers and known constants
        tree = ast.parse(expr, mode='eval')
        # Walk tree to verify only allowed nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                 ast.Num, ast.Constant, ast.Name,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div,
                                 ast.Pow, ast.USub, ast.UAdd)):
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in SAFE_MATH_NAMES:
                    continue
            return None  # Disallow anything else
        return float(eval(compile(tree, '<answer>', 'eval'), {"__builtins__": {}}, SAFE_MATH_NAMES))
    except Exception:
        return None
```

### Phase 2: Unify Into Single Math Benchmark Class

**Merge `GSM8KBenchmark` into `MathCompetitionBenchmark`** (or create a new `UnifiedMathBenchmark` that replaces both).

The unified class:
- Loads from ANY math dataset (MATH, GSM8K, AMC, AIME, IMO, synthetic)
- Uses ONE `_solve_problem()` method with ONE `execute_task()` call
- Uses ONE `_answers_match()` with the LaTeX normalizer
- Tags each problem with its source (`competition` field already exists)
- Produces one unified result set

**Dataset loading**: Keep the existing loaders as private methods but unify them:

```python
class UnifiedMathBenchmark:
    """Single math benchmark — one mind, all math question styles."""

    def _load_problems(self) -> list[dict]:
        problems = []
        problems.extend(self._load_math_dataset())      # MATH (competition)
        problems.extend(self._load_gsm8k_dataset())      # GSM8K (word problems)
        problems.extend(self._load_competition_files())   # AMC/AIME/IMO JSON
        if not problems:
            problems = self._synthetic_guard_problems()
        if self.max_problems is not None:
            problems = problems[:self.max_problems]
        return problems
```

**Key**: The `run_enriched_benchmarks.py` runner currently creates `MathCompetitionBenchmark` and `GSM8KBenchmark` separately. After unification, it should create ONE `UnifiedMathBenchmark` and report sub-scores by competition tag.

### Phase 3: Update Runner to Use Unified Class

In `scripts/run_enriched_benchmarks.py`:
- Replace separate Math and GSM8K instantiation with single `UnifiedMathBenchmark`
- Report results with sub-breakdowns: `MATH: X/500`, `GSM8K: Y/1319`
- The combined score becomes the "math" line in the health log
- Keep backward compatibility: health log still reports `math` and `gsm8k` as separate suite entries for trend tracking, but they come from the SAME benchmark instance

---

## File Changes

| File | Action |
|------|--------|
| `benchmarks/math_competitions.py` | Major rewrite → `UnifiedMathBenchmark` with LaTeX normalizer |
| `benchmarks/gsm8k.py` | Deprecate or reduce to thin wrapper that delegates to unified class |
| `scripts/run_enriched_benchmarks.py` | Update to use unified class, report sub-scores |
| `tests/test_math_zero_fix.py` | Expand: test LaTeX normalization, test unified loading |

---

## LaTeX Normalization Test Cases

These MUST pass:

```python
# Simple numeric (already works)
assert normalize("0") == "0"
assert normalize("42") == "42"
assert normalize("-7") == "-7"

# Fractions
assert normalize(r"\frac{3}{4}") → float ≈ 0.75
assert normalize(r"\frac{1}{2}") → float ≈ 0.5
assert normalize(r"\dfrac{7}{3}") → float ≈ 2.333...
assert normalize(r"\frac{22}{7}") → float ≈ 3.14285...

# Square roots
assert normalize(r"\sqrt{2}") → float ≈ 1.41421...
assert normalize(r"\sqrt{16}") → float == 4.0
assert normalize(r"2\sqrt{3}") → float ≈ 3.46410...
assert normalize(r"\sqrt[3]{8}") → float == 2.0

# Pi
assert normalize(r"2\pi") → float ≈ 6.28318...
assert normalize(r"\frac{\pi}{2}") → float ≈ 1.5707...

# Mixed
assert normalize(r"\frac{1+\sqrt{5}}{2}") → float ≈ 1.618... (golden ratio)

# Non-numeric (should return text for text comparison)
assert normalize(r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}") → text fallback
assert normalize(r"\text{Monday}") → "monday"

# Cleanup
assert normalize(r"\left(\frac{3}{4}\right)") → float ≈ 0.75
assert normalize(r"12\,345") → float == 12345
```

---

## Sovereignty Notes

- LaTeX normalization is **ingestion-path** (Python answer comparison for benchmark scoring)
- The actual math REASONING remains sovereign: `execute_task(type="MATH_TASK")` → GPU pipeline
- No sovereignty violation — we're fixing the SCOREBOARD, not the SOLVER
- The solver (composed head pipeline) is the same for both benchmarks — that's the "one mind"

---

## Success Criteria

1. Math benchmark score > 0 (even a few correct answers proves the fix works)
2. GSM8K score unchanged or improved (no regression)
3. Single `UnifiedMathBenchmark` class handles both datasets
4. LaTeX normalization passes all test cases above
5. `run_enriched_benchmarks.py` produces sub-breakdowns by source
6. Health log backward compatible (still reports `math` and `gsm8k` entries)

---

## Execution Order

1. **Add `_normalize_latex_answer()` to `math_competitions.py`** — immediate fix
2. **Add `_safe_eval_math()`** — safe numeric evaluation
3. **Update `_answers_match()`** — normalize before comparing
4. **Test with a few MATH problems manually** — verify score > 0
5. **Merge GSM8K into unified class**
6. **Update runner**
7. **Expand tests**

Phase 1 alone should break the 0/500 wall. Phases 2-3 achieve the "single mind" goal.
