# CODEX: Critical Tier-1 PTX Bug - Empty Stack on Simple Arithmetic

**Date:** December 14, 2025
**Priority:** CRITICAL - This is blocking all template-based math solving
**Partner:** Claude (Root Cause Analysis) -> Codex (Fix Implementation)

---

## Root Cause Identified

**The Tier-1 PTX kernel produces empty stack for simple arithmetic programs.**

```python
# These FAIL on Tier-1:
'12 60 /'   → ERROR: Tier‑1 GPU execution produced empty stack
'2 3 +'     → ERROR: Tier‑1 GPU execution produced empty stack

# This WORKS (routes to Tier-2 due to DUP):
'48 DUP 2 / +'  → 72.0 ✓
```

The tier routing logic in `tiered_rpn.py` line 376-379:
```python
stack_ops = {50, 51, 52, 53, 54, 55}
# Route stack-heavy programs to Tier‑2 for stability
if op_set & stack_ops:
    tier = 2
```

Programs with DUP/SWAP/etc. go to Tier-2 which works. Simple arithmetic goes to buggy Tier-1.

---

## Impact

- Templates like `{0} {1} *` route to Tier-1 and FAIL
- Only templates with DUP (like `{0} DUP 2 / +`) work
- This explains why 9 templates "hit" but produce wrong results

---

## Solution Options

### Option A: Quick Workaround (Immediate Fix)

Modify tier routing to ALWAYS use Tier-2 for benchmark programs:

**File:** `knowledge3d/cranium/bridges/tiered_rpn.py`

```python
def _determine_tier(self, op_codes: Sequence[int]) -> int:
    """Return tier index (1-3) for given op-code sequence."""
    iterable = [int(op) for op in op_codes]

    key = tuple(iterable)
    if key == self._tier_cache_key:
        return self._tier_cache_value

    ternary_ops = {0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76}
    has_tier3 = any(
        (op not in ternary_ops) and (op >= self.MATRIX_OPCODE_THRESHOLD or op == 0x02)
        for op in iterable
    )
    if has_tier3:
        tier = 3
    else:
        # WORKAROUND: Always use Tier-2 until Tier-1 PTX bug is fixed
        # Tier-1 produces empty stack for simple arithmetic programs
        tier = 2

    self._tier_cache_key = key
    self._tier_cache_value = tier
    return tier
```

### Option B: Fix Tier-1 PTX Kernel (Proper Fix)

The bug is likely in `modular_rpn_kernel_lite.ptx`. The kernel doesn't properly push literal values onto the stack when only literals and binary ops are present.

Debug steps:
1. Check if `OP_LITERAL` (opcode 0) properly increments stack size
2. Verify stack pointer handling after pushing literals
3. Ensure binary ops (10, 11, 12, 13) read from correct stack positions

**File:** `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`

Look for the literal push handler and verify it updates the stack size header correctly.

### Option C: Template RPN Workaround

Modify all templates to include a no-op stack operation:

```python
# Instead of:
rpn_program="{0} {1} *"

# Use:
rpn_program="{0} DUP DROP {1} *"  # DUP DROP is a no-op but routes to Tier-2
```

---

## Recommended Approach

1. **Immediate:** Apply Option A (force Tier-2) to unblock benchmarks
2. **Later:** Investigate and fix Tier-1 PTX kernel (Option B)

---

## Verification Test

After fix, run:

```python
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

engine = ModularRPNEngine()
tests = ['12 60 /', '2 3 +', '6 2 /', '10 5 -', '3 4 *']
for rpn in tests:
    result = engine.evaluate(rpn)
    print(f'{rpn} => {result}')
```

Expected:
```
12 60 / => 0.2
2 3 + => 5.0
6 2 / => 3.0
10 5 - => 5.0
3 4 * => 12.0
```

---

## Success Criteria

1. Simple arithmetic programs execute without error
2. Template hits convert to correct answers
3. gsm8k accuracy improves significantly (target: 10%+)

---

## Files to Modify

**For Option A (Quick Fix):**
- `knowledge3d/cranium/bridges/tiered_rpn.py` - Force Tier-2

**For Option B (Proper Fix):**
- `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx` - Fix literal push
- `knowledge3d/cranium/bridges/lightweight_rpn.py` - Verify stack handling
