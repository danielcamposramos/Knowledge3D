# Codex Math Sovereignty Audit — 2026-02-12

## Scope
Requested audit: verify whether current math solving follows sovereign path
`Question -> Grammar/Math Galaxy query -> RPN composition -> PTX execution`
or falls back to Python parsing/execution.

## Executive Finding
Current benchmark math solving is **not** using Galaxy-first RPN/PTX execution.  
It is predominantly a Python regex + safe `eval` pipeline with a few hardcoded derivative heuristics.

- Routing is correct (`math` specialist), but solver path is Python-centric.
- Galaxy queries are executed, but their retrieved entries are **not used** to compute the final answer.
- No active use of PTX math kernels in `run_all_benchmarks.py` math path.

---

## 1) Actual Call Path (Current)

### Benchmark entry point
`benchmarks/math_competitions.py`:
- calls `navigate_and_compose(...)` [benchmarks/math_competitions.py:213]
- calls `navigator.query(...)` for patterns [benchmarks/math_competitions.py:231]
- computes prediction from `navigator.execute(composed)` [benchmarks/math_competitions.py:238]

### Composition step
`knowledge3d/knowledgeverse/trm_navigator.py`:
- `compose(...)` always returns `program_type="math_expression"` with raw question text [knowledge3d/knowledgeverse/trm_navigator.py:363]
- `patterns` are only counted (`patterns_used`) and not compiled into executable math program [knowledge3d/knowledgeverse/trm_navigator.py:367]

### Execution step
`knowledge3d/knowledgeverse/trm_navigator.py`:
- `execute(...)` routes math to `_solve_math(expression, use_enriched=...)` [knowledge3d/knowledgeverse/trm_navigator.py:381]
- `_solve_math(...)`:
  - tries derivative heuristic parser (`_solve_derivative_prompt`) [knowledge3d/knowledgeverse/trm_navigator.py:725]
  - otherwise extracts arithmetic-like substring via regex [knowledge3d/knowledgeverse/trm_navigator.py:813]
  - evaluates with AST + Python `eval` [knowledge3d/knowledgeverse/trm_navigator.py:826]

---

## 2) Sovereignty Deviation Details

### A) Python parsing dominates hot math path
Evidence:
- regex extraction and parsing (`re.search`, `re.findall`) [knowledge3d/knowledgeverse/trm_navigator.py:746, 758, 764, 815, 821]
- AST parse + Python eval [knowledge3d/knowledgeverse/trm_navigator.py:828, 834]

### B) Galaxy retrieval is not used for solving
Evidence:
- `patterns = navigator.query(...)` is executed [benchmarks/math_competitions.py:231]
- final answer still comes from `navigator.execute(composed)` [benchmarks/math_competitions.py:238]
- composed program for math is raw text expression, not RPN from retrieved symbols [knowledge3d/knowledgeverse/trm_navigator.py:364]

### C) PTX/RPN runtime not in active benchmark math solve loop
- Active path files (`benchmarks/math_competitions.py`, `knowledge3d/knowledgeverse/trm_navigator.py`) contain no PTX runtime invocation for math solving.
- Existing PTX/runtime modules exist elsewhere, but are not wired into this benchmark loop.

### D) No external SymPy/SciPy in active path (good), but still non-sovereign math execution
- Active solver does **not** call sympy/scipy in this path.
- However, reliance on Python regex+eval still violates intended sovereign hot path.

---

## 3) Concrete Trace Example

Question tested:
`If 2x + 3 = 11, what is x?`

Observed:
- composed program type: `math_expression`
- composed expression: original question text
- route: `specialist=math`, galaxies `['Math','Grammar']`
- result: `3.0`

Root cause:
- `_extract_arithmetic_expr` captured `'+ 3'` from question text [knowledge3d/knowledgeverse/trm_navigator.py:815]
- `_safe_eval('+ 3') -> 3.0` [knowledge3d/knowledgeverse/trm_navigator.py:826]

So the solver returns a parser artifact, not equation solution (`x=4`).

---

## 4) Why Math Accuracy Stalls at 0%

This behavior explains previous diagnostics:
- `route_specialist_counts.math == 100%` (routing works)
- high `predicted_none_rate` and wrong numeric predictions (solver extraction weak)
- no meaningful lift from richer Math Galaxy content because retrieval is disconnected from execution.

---

## 5) Architectural Gap vs Intended Design

### Intended
`Question -> Grammar/Math retrieval -> RPN template composition -> PTX execution -> answer`

### Actual
`Question -> regex extraction / heuristics -> Python eval -> answer`

Main gap: **no solver-time compilation from retrieved Math/Grammar entries into executable RPN/PTX plan**.

---

## 6) Recommended Fix Sequence (Foundation First)

1. **Introduce sovereign math execution plan object**
   - `program_type = "math_rpn"` when route domain is math and suitable templates are found.
   - include explicit tokenized operands/operators and provenance (entry IDs).

2. **Compile from Galaxy entries (not raw text)**
   - query Grammar for equation structure (`linear_eq`, `quadratic`, etc.)
   - query Math for operator templates/symbol bindings
   - produce RPN sequence from templates.

3. **Execute via PTX/RPN runtime**
   - replace `_safe_eval` for enriched mode with PTX-backed evaluator.
   - keep Python safe-eval only as debug fallback behind explicit non-sovereign flag.

4. **Telemetry gates**
   - add `math_sovereign_path_rate` (fraction solved via math_rpn/PTX)
   - add `math_python_fallback_rate`
   - stage gate: fail if sovereign path rate < target.

5. **Regression harness**
   - include deterministic linear-equation sanity set (e.g., `2x+3=11`, `3x-5=10`) validating exact answer + provenance chain.

---

## 7) Immediate Week 22.2 Implication

Recommendation: hold broad ARC parameter tuning until this foundational math sovereignty gap is corrected for the math track.

- ARC oracle rescue can proceed in parallel.
- But for math benchmark credibility and K3D sovereignty claims, this solver path must be migrated from Python parsing/eval to Galaxy->RPN->PTX.

---

## Quick Verdict

- **Routing:** OK
- **Knowledge presence:** OK
- **Math solving path sovereignty:** **NOT OK (currently Python-centric)**
- **Primary action:** rewire math execution to sovereign RPN/PTX using retrieved Galaxy templates.
