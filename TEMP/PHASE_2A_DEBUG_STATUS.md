# Phase 2A Debug Status Report
**Date:** 2025-11-18
**Status:** PTX kernel has runtime memory access error (error 716: misaligned address)

---

## What We Fixed ✅

### 1. Test File Syntax Errors
- **File:** `tests/test_procedural_drawing_performance.py`
- **Issue:** Line 92 had erroneous backslash before f-string quote
- **Fix:** Removed backslash: `f"{i*0.05}..."` instead of `f\"{i*0.05}..."`
- **Issue:** Line 66 had incorrect import path for `LatencyGuard`
- **Fix:** Changed from `sovereign_bridges` to `ptx_runtime.latency_guard`

### 2. PTX Compilation Errors
- **File:** `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`

**Error 1:** `mul.lo` arguments mismatch (lines 100, 158)
- **Root Cause:** Trying to write u32 result into u64 register
- **Fix:** Added `seg_offset_u32` temporary register, compute offset as u32, then convert to u64
```ptx
mul.lo.u32 seg_offset_u32, seg_count, 16;
cvt.u64.u32 seg_offset, seg_offset_u32;
```

**Error 2:** `cvt` requires rounding modifier (lines 128-129)
- **Fix:** Added `.rn` (round to nearest)
```ptx
cvt.rn.f32.u32 t, i;
cvt.rn.f32.u32 tmp, segs;
```

### 3. Vector Load Alignment Issues
- **Issue:** Used `ld.global.v2.f32` and `ld.global.v4.f32` which require 8/16-byte alignment
- **Root Cause:** Bytecode may not be aligned to vector boundaries
- **Fix:** Replaced with scalar loads
```ptx
// Before (misaligned):
ld.global.v2.f32 {curx, cury}, [addr];

// After (safe):
ld.global.f32 curx, [addr + 0];
ld.global.f32 cury, [addr + 4];
```

### 4. Parameter Passing
- **File:** `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` (lines 172-177)
- **Issue:** Device pointers wrapped in `ctypes.c_uint64(d_ptr.value)`
- **Root Cause:** Incorrect API usage - loader.launch expects raw device pointers
- **Fix:** Pass device pointers directly
```python
# Before (wrong):
params=[
    ctypes.c_uint64(d_bytecode.value),
    ...
]

# After (correct):
params=[
    d_bytecode,  # Pass pointer directly
    ctypes.c_uint32(bytecode.nbytes),  # Only wrap scalars
    ...
]
```

### 5. PTX Address Space Conversion
- **Issue:** Pointers loaded from parameters need explicit conversion to global address space
- **Fix:** Added `cvta.to.global.u64` instructions (lines 54-56)
```ptx
.reg .u64 prog_ptr_param, seg_ptr_param, cnt_ptr_param;
ld.param.u64 prog_ptr_param, [d_rpn_program];
ld.param.u64 seg_ptr_param, [d_segments_out];
ld.param.u64 cnt_ptr_param, [d_segment_count];

// Convert to global address space
cvta.to.global.u64 prog_ptr, prog_ptr_param;
cvta.to.global.u64 seg_ptr, seg_ptr_param;
cvta.to.global.u64 cnt_ptr, cnt_ptr_param;
```

---

## Current Blocker ❌

### Runtime Error: Misaligned Address (CUDA Error 716)

**Symptoms:**
- PTX compiles successfully with `ptxas`
- Module loads successfully (`cuModuleLoadData` succeeds)
- Kernel launches without error
- **FAILS during `cuCtxSynchronize()` with error 716**

**What Works:**
```bash
# PTX compilation
ptxas --gpu-name=sm_86 pixel_genesis_universal_primitive.ptx -o /dev/null  # ✓ OK

# Module loading (isolated)
python -c "from knowledge3d.cranium.sovereign import loader; ..."  # ✓ OK
```

**What Fails:**
```bash
# Kernel execution
pytest tests/test_procedural_drawing_performance.py::test_gpu_rpn_operand_decoding  # ✗ FAIL
# Error: RuntimeError: Sovereign loader error: misaligned address
```

**Test Case That Fails:**
```python
program = "-0.5 -0.5 MOVE 0.5 0.5 LINE STROKE"
result = bridge.execute_rpn_gpu(program, width=64, height=64)
# Fails at loader.synchronize()
```

**Bytecode Generated:**
```
Hex: 640000003f0000003f 650000003f0000003f 6a
     ^^-MOVE--^^x=0.5^^y=0.5 ^^-LINE--^^x=0.5^^y=0.5 ^^STROKE
Total: 19 bytes
```

---

## Likely Root Causes

Based on error 716 (misaligned address) occurring during **execution** (not loading):

### Theory 1: Global Memory Access Misalignment
**Location:** PTX lines 103-106, 161-164 (segment writes)
```ptx
st.global.f32 [addr + 0], curx;
st.global.f32 [addr + 4], cury;
st.global.f32 [addr + 8], fx1;
st.global.f32 [addr + 12], fy1;
```

**Hypothesis:**
- `addr` is computed as `seg_ptr + seg_offset`
- `seg_offset = seg_count * 16` (should be 16-byte aligned)
- BUT: Maybe `seg_ptr` itself isn't aligned?

**Check Needed:**
1. Verify GPU malloc returns aligned pointers (should be at least 256-byte aligned)
2. Add alignment assertion in PTX before first store
3. Check if `cvta.to.global.u64` produces correct address

### Theory 2: Parameter Stride Mismatch
**Location:** PTX parameter declarations (lines 25-31)
```ptx
.entry execute_drawing_rpn(
    .param .u64 d_rpn_program,
    .param .u32 program_length,
    .param .u64 d_segments_out,
    .param .u64 d_segment_count,
    .param .u32 segments_per_curve,
    .param .f32 ternary_hint
)
```

**Hypothesis:**
- Parameter packing/alignment may be wrong
- CUDA driver expects specific alignment for parameters
- Float parameter at end might cause misalignment

**Test:**
- Try removing `ternary_hint` parameter (make it const 0.0 in PTX)
- Try changing order (put all u64 first, then u32, then f32)

### Theory 3: Bytecode Read Beyond Bounds
**Location:** PTX line 76 (load opcode byte)
```ptx
ld.global.u8 opcode, [addr];
```

**Hypothesis:**
- Loop continues reading past bytecode end
- Reads garbage opcode (e.g., 0x6A = STROKE)
- STROKE has no operands, loops back
- Eventually reads completely invalid memory

**Test:**
- Add explicit END opcode (0xFF) to bytecode
- Check loop termination: `setp.ge.u32 p0, idx, prog_len;` (line 68)
- Add bounds checking before every `ld.global`

---

## Recommended Next Steps for Codex

### Option A: Add PTX Debugging (Quick Win)
1. **Add segment count guard at start of OPC_LINE:**
```ptx
OPC_LINE:
    // Fail fast if segment buffer full
    setp.ge.u32 p0, seg_count, 4096;
    @p0 bra LOOP;  // Skip if buffer full

    // ... rest of LINE logic
```

2. **Add alignment check before first store:**
```ptx
OPC_LINE:
    // ... (after computing addr)

    // DEBUG: Check alignment (addr must be 16-byte aligned)
    and.b64 addr_check, addr, 15;  // addr & 0xF
    setp.ne.u64 p_misalign, addr_check, 0;
    @p_misalign bra WRITE_COUNT;  // Abort if misaligned
```

3. **Test with minimal bytecode:**
```python
# Test MOVE only (no LINE)
program = "0.5 0.5 MOVE"
result = bridge.execute_rpn_gpu(program, width=64, height=64)
# Should write NO segments, just update position
```

### Option B: Simplify Kernel (Nuclear Option)
1. **Remove all operand decoding temporarily**
2. **Make kernel just count opcodes:**
```ptx
LOOP:
    setp.ge.u32 p0, idx, prog_len;
    @p0 bra WRITE_COUNT;

    ld.global.u8 opcode, [addr];
    add.u32 idx, idx, 1;
    add.u32 seg_count, seg_count, 1;  // Just count bytes
    bra LOOP;

WRITE_COUNT:
    st.global.u32 [cnt_ptr], seg_count;  // Return byte count
```

3. **Verify this minimal kernel works**
4. **Incrementally add back operand decoding**

### Option C: Use CUDA Memcheck (Definitive Answer)
```bash
compute-sanitizer --tool memcheck python debug_kernel_exec.py
```
This will pinpoint EXACT line in PTX where misaligned access occurs.

**NOTE:** Tried running memcheck but output was truncated. Need to:
1. Run with `--log-file memcheck.log`
2. Parse log for first ERROR line
3. Look at PTX line number + instruction

---

## Files Modified

1. `tests/test_procedural_drawing_performance.py` - Fixed syntax errors
2. `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx` - Fixed compilation errors, added cvta, replaced vector loads
3. `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Fixed parameter passing

## Files for Reference

- `knowledge3d/cranium/kernels/procedural_glyph_rasterizer.ptx` - Working example with proper cvta usage
- `knowledge3d/cranium/bridges/procedural_glyph_bridge.py` - Working bridge with correct parameter passing
- `knowledge3d/cranium/sovereign/loader.py` - CUDA loader API

---

## Validation Checklist (Once Fixed)

- [ ] PTX compiles: `ptxas pixel_genesis_universal_primitive.ptx`
- [ ] Module loads: `loader.load_module_from_file(...)`
- [ ] Kernel executes: `loader.synchronize()` succeeds
- [ ] Segment count > 0 for "MOVE LINE" program
- [ ] Operand decoding test passes
- [ ] QUAD tessellation test passes
- [ ] Latency < 100µs (may need optimization)

---

**Next Session:** Focus on Option A (add PTX debugging) or Option C (memcheck analysis) to pinpoint exact misaligned access.
