# CODEX: Quick Fix - inf/NaN Guard

**Priority:** BLOCKING - Training aborts at ~4500 problems
**Time:** 5 minutes

---

## Problem

Large factorials (e.g., 100!) produce `inf` values which crash `MathOutputAdapter._format_for_benchmark()` with OverflowError.

---

## Fix

**File:** `knowledge3d/training/math_benchmarks/math_output_adapter.py`

Add inf/NaN guard to `_format_for_benchmark` and `_to_number`:

```python
import math  # Add at top

def _format_for_benchmark(self, answer: Any, source: str) -> str:
    if answer is None:
        return ""
    # Guard inf/NaN
    if isinstance(answer, float) and (math.isinf(answer) or math.isnan(answer)):
        return ""
    # ... rest unchanged
```

And fix `_to_number`:

```python
def _to_number(self, value: Any):
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isinf(f) or math.isnan(f):
            return None
        return f
    # ... rest unchanged
```

---

## After Fix

Re-run training:
```bash
PYTHONPATH=. python3 scripts/train_math_benchmarks.py --epochs 1
```

Report final scores for all 5 datasets.
