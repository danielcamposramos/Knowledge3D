# CODEX: Template Diagnostic - Why 9 Hits but 0.20% Accuracy?

**Date:** December 14, 2025
**Priority:** HIGH - Templates hit but produce wrong answers
**Partner:** Claude (Architecture Analysis) -> Codex (Implementation)

---

## The Problem

Templates are now hitting (9 vs 4 before), but accuracy remains at 0.20%.

**This means:** Templates match patterns but produce WRONG answers.

---

## Diagnostic Task

Create a diagnostic script to understand:
1. Which templates are hitting?
2. What numbers are being extracted?
3. What RPN is being generated?
4. What answer is computed vs expected?

**File:** `scripts/diagnose_template_matches.py`

```python
#!/usr/bin/env python3
"""
Diagnose template matching to understand why hits don't convert to accuracy.
"""

import json
import re
from pathlib import Path

from knowledge3d.training.math_benchmarks.math_templates import get_all_templates
from knowledge3d.training.math_benchmarks.number_words import normalize_number_words
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


def extract_answer(answer_str: str) -> float | None:
    """Extract numeric answer from GSM8K answer field."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*([\d,.-]+)", answer_str)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def diagnose_templates(limit: int = 20):
    """Diagnose template matching on first N GSM8K problems."""

    templates = get_all_templates()
    engine = ModularRPNEngine()

    gsm_path = Path("/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/train.jsonl")
    if not gsm_path.exists():
        print(f"Dataset not found: {gsm_path}")
        return

    with open(gsm_path) as f:
        problems = [json.loads(line) for line in f][:limit]

    hits = 0
    correct = 0

    for i, p in enumerate(problems):
        question = p.get("question", "")
        answer_str = p.get("answer", "")
        expected = extract_answer(answer_str)

        print(f"\n{'='*60}")
        print(f"PROBLEM {i+1}: {question[:80]}...")
        print(f"EXPECTED: {expected}")

        # Try each template
        for rule in templates:
            # Try original text
            m = re.search(rule.pattern, question, re.IGNORECASE | re.DOTALL)

            # Try normalized text
            if not m:
                normalized = normalize_number_words(question)
                m = re.search(rule.pattern, normalized, re.IGNORECASE | re.DOTALL)

            if m:
                # Build RPN
                rpn = rule.rpn_program
                captures = []
                for idx, group in enumerate(m.groups()):
                    clean = str(group).replace(",", "").strip()
                    captures.append(clean)
                    rpn = rpn.replace(f"{{{idx}}}", clean)

                if not is_valid_rpn(rpn):
                    continue

                try:
                    result = engine.evaluate(rpn)
                except Exception as e:
                    print(f"  TEMPLATE: {rule.rule_id}")
                    print(f"  CAPTURES: {captures}")
                    print(f"  RPN: {rpn}")
                    print(f"  ERROR: {e}")
                    continue

                hits += 1
                is_correct = expected is not None and abs(result - expected) < 0.01
                if is_correct:
                    correct += 1

                print(f"  TEMPLATE: {rule.rule_id}")
                print(f"  PATTERN: {rule.pattern[:60]}...")
                print(f"  MATCH: '{m.group(0)[:60]}...'")
                print(f"  CAPTURES: {captures}")
                print(f"  RPN: {rpn}")
                print(f"  RESULT: {result}")
                print(f"  CORRECT: {'YES' if is_correct else 'NO'}")

                # Only show first matching template
                break
        else:
            print("  NO TEMPLATE MATCH")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {hits} hits, {correct} correct out of {len(problems)} problems")
    print(f"Accuracy from templates: {100*correct/len(problems):.1f}%")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    diagnose_templates(limit)
```

---

## Run Diagnostic

```bash
PYTHONPATH=. python scripts/diagnose_template_matches.py 50
```

---

## What to Look For

1. **Wrong Number Extraction:**
   - Pattern matches but captures wrong numbers
   - Example: "48 of her friends" captures "48" but also matches another number later

2. **Wrong Operation Order:**
   - RPN has operands in wrong order
   - Example: "12/60" vs "60/12"

3. **Missing Steps:**
   - Template handles only part of the problem
   - Example: Problem needs 3 steps, template does 1

4. **Overly Greedy Patterns:**
   - `.*?` matches too much, grabbing wrong numbers
   - Example: Pattern intended for "half...altogether" matches different structure

---

## Potential Fixes

Based on diagnostic findings:

### Fix 1: More Specific Number Anchors

Instead of:
```python
r"(\d+\.?\d*).*?half"  # Greedy - might skip the right number
```

Use:
```python
r"(?:sold|gave|had)\s*(?:clips?\s*to\s*)?(\d+)"  # Context-specific
```

### Fix 2: Position-Aware Captures

```python
# Capture the LAST number before "half", not the first
r".*?(\d+)[^\d]*half\s*(?:as many)"
```

### Fix 3: Multi-Step Template Chaining

For problems needing 3+ steps, use scratchpad:

```python
# Step 1: Extract and compute intermediate
# Step 2: Use intermediate for final computation
```

---

## Expected Diagnostic Output

```
PROBLEM 1: Natalia sold clips to 48...
EXPECTED: 72.0
  TEMPLATE: gsm_half_altogether
  CAPTURES: ['48']
  RPN: 48 DUP 2 / +
  RESULT: 72.0
  CORRECT: YES

PROBLEM 2: Weng earns $12 an hour...
EXPECTED: 10.0
  TEMPLATE: gsm_hourly_minutes
  CAPTURES: ['12', '50']
  RPN: 12 60 / 50 *
  RESULT: 10.0
  CORRECT: YES

PROBLEM 3: Betty is saving money...
EXPECTED: 5.0
  NO TEMPLATE MATCH (4-step problem)
```

---

## Success Criteria

1. Diagnostic script runs and shows template behavior
2. Identify WHY 9 hits → ~1 correct
3. Fix patterns based on findings
4. Re-run benchmark to measure improvement

---

## Key Insight

**Templates are matching but extracting wrong numbers or applying wrong operations.**

The diagnostic will reveal the exact failure mode so we can fix it.
