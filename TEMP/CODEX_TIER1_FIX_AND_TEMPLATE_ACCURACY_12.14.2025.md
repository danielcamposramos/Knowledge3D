# CODEX: Tier-1 PTX Fix + Template Accuracy Enhancement

**Date:** December 14, 2025
**Priority:** HIGH
**Partner:** Claude (Root Cause Analysis) -> Codex (Implementation)

---

## Executive Summary

Three tasks to improve math benchmark accuracy:
1. **Fix Tier-1 PTX kernel** - Currently produces empty stack for simple arithmetic
2. **Add retry-on-empty-stack fallback** - Safety net in case Tier-1 regresses
3. **Improve template capture accuracy** - Patterns match but extract wrong numbers

Current status after Tier-2 workaround:
- template: 173 hits, gsm8k: 3.00%, overall: 6.40%
- fail: 0 (all programs execute)

---

## Task 1: Fix Tier-1 PTX Kernel

### Root Cause Analysis

**File:** `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`

The Tier-1 (lite) kernel produces `size=0` in the header for simple programs like `12 60 /`.

**Key Code Paths Analyzed:**

1. **Literal Push (opcode 0)** at $L__BB0_126 (line 405-427):
   - Loads scalar from global memory ✓
   - Increments scalar_index ✓
   - Stores to shared stack ✓
   - Increments stack_size ✓

2. **Binary Ops (10-13)** at $L__BB0_44 (line 124-150) → $L__BB0_53:
   - Pops two values ✓
   - Computes operation ✓
   - Pushes result at $L__BB0_54 ✓
   - Sets stack_size = original - 1 ✓

3. **Writeback** at $L__BB0_133 (line 1028-1034):
   - Writes {0, stack_size, error_code, 0} to d_state ✓

**The PTX structure matches Tier-2, but something causes empty stack.**

### Fix Strategy: Use Tier-2 PTX as Reference

The working Tier-2 kernel (`modular_rpn_kernel.ptx`) has identical structure but more extensive opcode coverage. Compare key sections:

**Tier-2 Literal Push (working):**
```ptx
// After loading scalar, incrementing index:
ld.shared.u32 	%r_size, [stack_size];
setp.gt.u32 	%p_overflow, %r_size, 63;
@%p_overflow bra $ERROR;
// Store and increment:
shl.b32 	%r_offset, %r_size, 4;
add.s32 	%r_addr, stack_base, %r_offset;
st.shared.v4.f32 	[%r_addr], {%f_scalar, 0, 0, 0};
add.s32 	%r_new_size, %r_size, 1;
st.shared.u32 	[stack_size], %r_new_size;
```

### Specific Fix Points in Tier-1 PTX

**Line 568-576 ($L__BB0_129) - Literal Push Store:**
```ptx
$L__BB0_129:
	shl.b32 	%r243, %r71, 4;
	mov.u32 	%r244, stack_base;
	add.s32 	%r245, %r244, %r243;
	mov.f32 	%f205, 0f00000000;
	st.shared.v4.f32 	[%r245], {%f220, %f205, %f205, %f205};
	add.s32 	%r246, %r71, 1;
	st.shared.u32 	[stack_size], %r246;
	bra.uni 	$L__BB0_131;
```

**Potential Issue:** %r71 is loaded at line 419 BEFORE the overflow check. If the check fails and we branch to error, but somehow still reach $L__BB0_129 through a different path, we'd have stale %r71.

**Verify:** Add `bar.sync 0;` before the store to ensure shared memory coherence:

```ptx
$L__BB0_129:
	bar.sync 	0;        // ADD: Ensure stack_size is visible
	shl.b32 	%r243, %r71, 4;
	...
```

### Alternative: Regenerate Lite PTX from CUDA Source

If direct PTX editing is risky, regenerate from the CUDA source:

1. Find `modular_rpn_kernel_lite.cu` (source file)
2. Compile with verbose output to identify differences
3. Or create a minimal test kernel to isolate the bug

---

## Task 2: Add Retry-on-Empty-Stack Fallback

**File:** `knowledge3d/cranium/bridges/tiered_rpn.py`

Add automatic retry on Tier-2 if Tier-1 returns empty stack:

```python
def execute_single(
    self,
    instance_id: int,
    op_codes: Sequence[int],
    scalars: Sequence[float],
    vectors: Sequence[Sequence[float]],
    *,
    matrices: Optional[Iterable[float]] = None,
) -> float:
    """Compatibility wrapper with automatic tier fallback."""
    if not (0 <= instance_id < self.MAX_INSTANCES):
        raise ValueError(f"Invalid instance_id {instance_id}")

    tier = self._determine_tier(op_codes)
    previous = self._last_tier[instance_id]

    # Reset if tier changed
    if previous != tier:
        self._reset_tier(previous, instance_id)

    # Try execution with fallback
    try:
        result = self._execute_on_tier(tier, instance_id, op_codes, scalars, vectors, matrices)
        self._last_tier[instance_id] = tier
        self._tier_counts[tier] += 1
        return float(result)
    except RuntimeError as e:
        if "empty stack" in str(e) and tier == 1:
            # Tier-1 failed with empty stack - retry on Tier-2
            self._tier1.reset_instance(instance_id)
            result = self._execute_on_tier(2, instance_id, op_codes, scalars, vectors, matrices)
            self._last_tier[instance_id] = 2
            self._tier_counts[2] += 1
            # Log for diagnostics
            self._tier1_fallback_count = getattr(self, '_tier1_fallback_count', 0) + 1
            return float(result)
        raise

def _execute_on_tier(self, tier: int, instance_id: int, op_codes, scalars, vectors, matrices) -> float:
    """Execute on specified tier."""
    if tier == 1:
        return self._tier1.execute_single(instance_id, op_codes, scalars, vectors)
    elif tier == 2:
        return self._tier2.execute_single(instance_id, list(op_codes), list(scalars), list(vectors))
    else:
        remapped = [0x005E if op == 0x0064 else int(op) for op in op_codes]
        matrices_seq = list(matrices) if matrices is not None else []
        return self._tier3.execute_scalar(instance_id, remapped, list(scalars), vectors, matrices_seq)

def _reset_tier(self, tier: int, instance_id: int) -> None:
    """Reset instance on specified tier."""
    if tier == 1:
        self._tier1.reset_instance(instance_id)
    elif tier == 2:
        self._tier2.reset_instance(instance_id)
    elif tier == 3:
        self._tier3.reset_instance(instance_id)
```

---

## Task 3: Improve Template Capture Accuracy

Templates hit (173) but many produce wrong answers. Issue: patterns match but capture wrong numbers.

### Problem Example

For problem: "Mark has 48 cookies. He eats 12 and gives 15 away. How many left?"

Template `gsm_has_gets_more` might match:
- Pattern: `(?:has|have|had)\s*(\d+).*?(\d+)\s*more`
- Captures: ('48', '15') - WRONG! Should capture for subtraction, not addition.

### Solution 1: Pattern Specificity Ordering

Already implemented (sort by length), but add domain-specific ordering:

```python
def get_all_templates() -> List[GrammarRule]:
    """Get templates with specificity and domain ordering."""
    templates = []
    templates.extend(get_gsm8k_templates())
    templates.extend(get_expanded_templates())

    # Sort by: (1) pattern length desc, (2) capture count desc, (3) rule_id
    templates.sort(
        key=lambda r: (len(r.pattern), r.pattern.count('('), r.rule_id),
        reverse=True
    )
    return templates
```

### Solution 2: Validate Capture Context

Add context validation to ensure captures are semantically correct:

```python
def _apply_template(self, rule, text: str):
    """Apply template with capture validation."""
    import re as _re

    try:
        m = _re.search(rule.pattern, text, _re.IGNORECASE | _re.DOTALL)
        if not m:
            normalized = normalize_number_words(text)
            m = _re.search(rule.pattern, normalized, _re.IGNORECASE | _re.DOTALL)
        if not m:
            return None

        # Validate captures: check numbers appear in expected semantic positions
        if not self._validate_captures(rule, m, text):
            return None

        rpn_program = rule.rpn_program
        for idx, group in enumerate(m.groups()):
            clean = str(group).replace(",", "").strip()
            rpn_program = rpn_program.replace(f"{{{idx}}}", clean)

        if not is_valid_rpn(rpn_program):
            return None
        return self.engine.evaluate(rpn_program)
    except Exception:
        return None

def _validate_captures(self, rule, match, text: str) -> bool:
    """Validate that captured numbers appear in expected semantic positions."""
    # Skip validation for simple patterns
    if len(match.groups()) < 2:
        return True

    rule_id = rule.rule_id
    captures = match.groups()

    # For subtraction rules, first capture should appear before subtraction keywords
    if 'sub' in rule_id or 'minus' in rule_id or 'less' in rule_id:
        sub_keywords = ['minus', 'subtract', 'less', 'spent', 'ate', 'gave', 'lost', 'uses']
        for kw in sub_keywords:
            if kw in text.lower():
                kw_pos = text.lower().find(kw)
                cap0_pos = text.find(str(captures[0]))
                if cap0_pos > kw_pos:
                    return False  # First capture appears after subtraction keyword - suspicious

    return True
```

### Solution 3: Multi-Number Disambiguation

When multiple numbers exist, choose based on problem structure:

```python
def _find_best_capture(self, text: str, pattern_type: str) -> Tuple[str, ...]:
    """Find best number captures based on problem structure."""
    import re

    # Extract all numbers with positions
    numbers = [(m.group(), m.start()) for m in re.finditer(r'\d+\.?\d*', text)]

    if pattern_type == 'half_altogether':
        # For "X ... half ... total": X should be the FIRST significant number
        # Filter out numbers that are part of other constructs
        significant = [n for n, pos in numbers if float(n) > 1]
        if significant:
            return (significant[0],)

    elif pattern_type == 'rate_time':
        # For "earns $X per hour ... Y minutes":
        # X is after $, Y is before "minutes"
        hourly_match = re.search(r'\$(\d+\.?\d*)', text)
        minute_match = re.search(r'(\d+)\s*minutes?', text)
        if hourly_match and minute_match:
            return (hourly_match.group(1), minute_match.group(1))

    return tuple()
```

### Solution 4: Answer Sanity Check

After computing, check if answer is reasonable:

```python
def _sanity_check_answer(self, result: float, text: str) -> bool:
    """Check if answer is reasonable given the problem."""
    # Extract all numbers from problem
    import re
    numbers = [float(n.replace(',', '')) for n in re.findall(r'\d+\.?\d*', text) if n]

    if not numbers:
        return True

    max_num = max(numbers)
    min_num = min(n for n in numbers if n > 0)

    # Answer should be in reasonable range
    if result < 0 and 'negative' not in text.lower():
        return False  # Suspicious negative result

    if result > max_num * 1000:
        return False  # Answer seems too large

    if result > 0 and result < min_num / 1000:
        return False  # Answer seems too small

    return True
```

---

## Implementation Order

1. **Immediate (keeps progress):** Add retry-on-empty-stack fallback (Task 2)
2. **Next:** Add capture validation and sanity checks (Task 3)
3. **When time permits:** Deep-dive Tier-1 PTX fix (Task 1)

---

## Success Criteria

1. Retry fallback catches any Tier-1 regressions
2. Template accuracy improves (target: 50%+ of hits are correct)
3. gsm8k reaches 10%+
4. Overall benchmark reaches 12%+

---

## Files to Modify

| File | Task | Priority |
|------|------|----------|
| `tiered_rpn.py` | Add retry fallback | HIGH |
| `run_sovereign_math_benchmarks.py` | Add capture validation | HIGH |
| `math_templates.py` | Add specificity ordering | MEDIUM |
| `modular_rpn_kernel_lite.ptx` | Fix Tier-1 | MEDIUM |

---

## Testing Commands

```bash
# Verify Tier-2 workaround still works
PYTHONPATH=. python3 -c "
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
e = ModularRPNEngine()
print('12 60 / =', e.evaluate('12 60 /'))  # Should be 0.2
print('2 3 + =', e.evaluate('2 3 +'))      # Should be 5.0
"

# Run benchmark
PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py --limit 100

# Check template accuracy on sample
PYTHONPATH=. python3 scripts/diagnose_template_matches.py 20
```
