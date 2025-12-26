# CODEX: Tier-1 Opcode Encoding Bug Fix

**Date:** December 14, 2025
**Priority:** CRITICAL
**Partner:** Claude (Root Cause) -> Codex (Implementation)

---

## Root Cause

**File:** `knowledge3d/cranium/bridges/lightweight_rpn.py`

The `_encode_uint16` function returns integers that are then wrapped with `bytes()`, producing SINGLE BYTES instead of uint16 values.

**Current (BROKEN):**
```python
# Line 179
codes_bytes = bytes(self._encode_uint16(op_codes))
# For op_codes = [0, 0, 12]:
#   _encode_uint16([0, 0, 12]) → [0, 0, 12]
#   bytes([0, 0, 12]) → b'\x00\x00\x0c' (3 bytes!)
```

**PTX kernel expects uint16 (2 bytes each):**
```ptx
ld.global.nc.u16 	%rs1, [%rd32];  // Reads 16-bit opcode
```

For 3 opcodes, kernel expects 6 bytes but receives only 3. First opcode reads as `0x0c00` (wrong!), subsequent reads are garbage.

---

## The Fix

Use `ctypes.c_uint16` array like Tier-2 does:

**File:** `knowledge3d/cranium/bridges/lightweight_rpn.py`

Replace lines 179-189 in `_execute_gpu`:

```python
def _execute_gpu(
    self,
    instance_id: int,
    op_codes: list[int],
    scalars: list[float],
    vectors: list[list[float]],
) -> float:
    # Encode opcodes as proper uint16 array (like Tier-2)
    OpArray = ctypes.c_uint16 * len(op_codes)
    op_arr = OpArray(*op_codes)

    scalars_bytes = self._encode_f32(scalars)
    vectors_bytes = self._encode_f32_flat(vectors)

    d_codes = self._ensure_scratch_buffer("codes", ctypes.sizeof(op_arr))
    d_scalars = self._ensure_scratch_buffer("scalars", len(scalars_bytes))
    d_vectors = self._ensure_scratch_buffer("vectors", len(vectors_bytes))

    loader.memcpy_htod(d_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
    loader.memcpy_htod(d_scalars, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(scalars_bytes))), len(scalars_bytes))
    loader.memcpy_htod(d_vectors, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(vectors_bytes))), len(vectors_bytes))

    # ... rest unchanged
```

---

## Also Remove Unused Method

The `_encode_uint16` method is no longer needed after this fix. Remove lines 345-349:

```python
# DELETE THIS:
@staticmethod
def _encode_uint16(values: Iterable[int]) -> list[int]:
    out: list[int] = []
    for v in values:
        out.append(int(v) & 0xFFFF)
    return out
```

Also update the cache comparison in `execute_single` to use a different approach (or remove caching for now).

---

## Verification

After fix, run:

```bash
PYTHONPATH=. python3 -c "
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
e = ModularRPNEngine()

# Force Tier-1 by using simple ops
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine
t = TieredRPNEngine()

# Simple arithmetic should work on Tier-1 now
result = e.evaluate('2 3 +')
print(f'2 3 + = {result}')  # Should be 5.0

result = e.evaluate('12 60 /')
print(f'12 60 / = {result}')  # Should be 0.2

print('Tier stats:', t.get_stats())
"
```

**Expected:**
```
2 3 + = 5.0
12 60 / = 0.2
Tier stats: {1: 2, 2: 0, 3: 0, 'tier1_fallbacks': 0}
```

---

## Impact

This fixes ALL Tier-1 execution failures. The PTX kernel was never the problem - it's the Python bridge encoding opcodes incorrectly.

No PTX modifications needed. The kernel code is correct.
