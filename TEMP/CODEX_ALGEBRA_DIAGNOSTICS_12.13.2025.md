# CODEX BRIEFING: Algebra Solver Diagnostics

**Date:** December 13, 2025
**Priority:** HIGH - Need data before adding more rules
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Problem

Algebra solver scaffolding is complete but scores unchanged:
- MATH: 2.15% (no improvement)
- AMC-AIME: 2.85% (no improvement)

**Hypothesis:** The classifier/strategies aren't matching real problem formats.

---

## Task 1: Sample Problems Diagnostic Script

Create a diagnostic script that shows us what's actually happening:

**File:** `scripts/diagnose_algebra_solver.py`

```python
#!/usr/bin/env python3
"""
Diagnose why algebra solver isn't improving scores.
Shows problem formats, classification results, and failure modes.
"""

import json
import random
from pathlib import Path


def load_samples(dataset: str, n: int = 10):
    """Load n sample problems from a dataset."""
    base = Path("/K3D/K3D_llama_cpp/datasets")

    if dataset == "math":
        path = base / "math/data/train.jsonl"
        problems = []
        with open(path) as f:
            for line in f:
                problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    elif dataset == "amc_aime":
        path = base / "AMC-AIME/data"
        problems = []
        for jsonl in path.glob("*.jsonl"):
            with open(jsonl) as f:
                for line in f:
                    problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    elif dataset == "omni_math":
        path = base / "Omni-MATH/Omni-Math.jsonl"
        problems = []
        with open(path) as f:
            for line in f:
                problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    return []


def diagnose_problem(problem: dict, solver):
    """Run full diagnostic on a single problem."""
    # Extract problem text
    text = problem.get("problem", problem.get("question", ""))[:500]
    answer = problem.get("answer", problem.get("solution", ""))

    # Classify
    classification = solver.classifier.classify(text)

    # Try to solve
    result, metadata = solver.solve(text)

    return {
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "ground_truth": str(answer)[:100],
        "classification": {
            "type": classification.problem_type,
            "subtype": classification.subtype,
            "confidence": classification.confidence,
            "coefficients": classification.coefficients,
        },
        "strategy": metadata.get("strategy", "none"),
        "result": result,
        "variables_found": metadata.get("variables", {}),
    }


def main():
    from knowledge3d.training.math_benchmarks.algebra_solver import AlgebraSolver

    solver = AlgebraSolver()

    print("=" * 80)
    print("ALGEBRA SOLVER DIAGNOSTICS")
    print("=" * 80)

    for dataset in ["math", "amc_aime", "omni_math"]:
        print(f"\n{'=' * 40}")
        print(f"DATASET: {dataset.upper()}")
        print("=" * 40)

        samples = load_samples(dataset, n=5)

        classification_counts = {}
        strategy_counts = {}
        solved_count = 0

        for i, problem in enumerate(samples):
            diag = diagnose_problem(problem, solver)

            print(f"\n--- Problem {i+1} ---")
            print(f"Text: {diag['text_preview']}")
            print(f"Ground Truth: {diag['ground_truth']}")
            print(f"Classification: {diag['classification']['type']}/{diag['classification']['subtype']} "
                  f"(conf={diag['classification']['confidence']:.2f})")
            print(f"Coefficients found: {diag['classification']['coefficients']}")
            print(f"Strategy: {diag['strategy']}")
            print(f"Result: {diag['result']}")

            # Track stats
            ctype = diag['classification']['type']
            classification_counts[ctype] = classification_counts.get(ctype, 0) + 1
            strategy_counts[diag['strategy']] = strategy_counts.get(diag['strategy'], 0) + 1
            if diag['result'] is not None:
                solved_count += 1

        print(f"\n--- {dataset.upper()} SUMMARY ---")
        print(f"Classification distribution: {classification_counts}")
        print(f"Strategy distribution: {strategy_counts}")
        print(f"Solved: {solved_count}/{len(samples)}")


if __name__ == "__main__":
    main()
```

---

## Task 2: Run Diagnostics and Report

```bash
PYTHONPATH=. python3 scripts/diagnose_algebra_solver.py 2>&1 | tee /tmp/algebra_diagnostics.txt
```

**Report back:**
1. What problem types are most common in MATH/AMC-AIME?
2. What are problems being classified as?
3. Are coefficients being extracted correctly?
4. What patterns are we missing?

---

## Task 3: Identify Top 5 Missing Patterns

Based on the diagnostics, identify the top 5 patterns we need to add to the classifier.

Example expected findings:
- "Many MATH problems use `\frac{a}{b}` which we don't parse"
- "AMC problems often ask 'find the value of...' which we classify as expression"
- "Coefficients aren't extracted from LaTeX like `2x^2 - 3x + 1`"

---

## Task 4: Quick Wins

After identifying gaps, add the **easiest** missing patterns first:

1. **LaTeX fraction extraction**: `\frac{num}{denom}` → coefficients
2. **Find the value patterns**: "find the value of", "what is", "compute"
3. **LaTeX quadratic parsing**: `ax^2 + bx + c` with LaTeX formatting

---

## Success Criteria

After running diagnostics:
1. We understand WHY the solver isn't matching problems
2. We have concrete patterns to add
3. We can prioritize improvements by impact

---

## Architecture Notes

- Diagnostics only - no benchmark runs yet
- Focus on understanding the data before adding more code
- Keep everything sovereign - no external libs

---

**Codex:** Run the diagnostic script, analyze the output, and report:
1. Classification distribution for MATH and AMC-AIME
2. Top 3-5 problem patterns we're missing
3. Proposed fixes (which patterns to add to classifier)
