# Codex Next Action: Phase 2A GPU RPN Debugging

**Priority:** HIGH
**Blocker:** CUDA error 716 (misaligned address) during kernel execution
**Context:** See `TEMP/PHASE_2A_DEBUG_STATUS.md` for full debug history

---

## Current Situation

✅ **What Works:**
- PTX compiles successfully
- PTX module loads successfully
- Parameter passing fixed
- Vector loads replaced with scalar loads
- Address space conversion added (cvta.to.global.u64)

❌ **What Fails:**
- Kernel execution fails at `cuCtxSynchronize()` with error 716
- Test: `-0.5 -0.5 MOVE 0.5 0.5 LINE STROKE`

---

## Your Task

**Goal:** Fix the misaligned address error and get at least `test_gpu_rpn_operand_decoding` passing.

### Approach 1: Add PTX Assertions (Recommended - Fast)

Add runtime checks to PTX kernel to identify exact failure point:

**Step 1:** Add segment buffer bounds check
```ptx
OPC_LINE:
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 fx1, [addr + 0];
    ld.global.f32 fy1, [addr + 4];
    add.u32 idx, idx, 8;

    // GUARD: Check if segment buffer full BEFORE computing address
    setp.ge.u32 p0, seg_count, 4096;
    @p0 bra LOOP;  // ← Already exists at line 98

    // NEW: Verify seg_count is sane
    setp.gt.u32 p_insane, seg_count, 10000;  // Paranoid check
    @p_insane bra WRITE_COUNT;  // Abort if corrupted

    mul.lo.u32 seg_offset_u32, seg_count, 16;
    // ... rest of store logic
```

**Step 2:** Test with MOVE-only (no stores)
```python
# In debug script or test
program = "0.5 0.5 MOVE"  # Only updates position, writes NO segments
result = bridge.execute_rpn_gpu(program, width=64, height=64)
# If this passes, issue is in LINE logic
# If this fails, issue is in MOVE operand reading
```

**Step 3:** Add address alignment verification
```ptx
OPC_LINE:
    // ... (after computing addr for seg_ptr + seg_offset)

    // DEBUG: Verify 16-byte alignment
    .reg .u64 addr_test;
    and.b64 addr_test, addr, 15;  // addr & 0xF
    setp.ne.u64 p_bad, addr_test, 0;
    @p_bad bra WRITE_COUNT;  // Silent abort if misaligned

    st.global.f32 [addr + 0], curx;
    // ...
```

### Approach 2: Simplify Kernel (Bisect Failure)

Temporarily gut the kernel to isolate the issue:

**Version A:** Just count opcodes (no operand reads)
```ptx
LOOP:
    setp.ge.u32 p0, idx, prog_len;
    @p0 bra WRITE_COUNT;

    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.u8 opcode, [addr];  // ← Does THIS fail?
    add.u32 idx, idx, 1;
    add.u32 seg_count, seg_count, 1;
    bra LOOP;
```

**Version B:** Read MOVE operands (no LINE stores)
```ptx
OPC_MOVE:
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 curx, [addr + 0];  // ← Does THIS fail?
    ld.global.f32 cury, [addr + 4];
    add.u32 idx, idx, 8;
    mov.f32 startx, curx;
    mov.f32 starty, cury;
    bra LOOP;

OPC_LINE:
    // Skip operand read, just count
    add.u32 idx, idx, 8;
    add.u32 seg_count, seg_count, 1;
    bra LOOP;
```

**Test each version** to narrow down failure point.

### Approach 3: Use CUDA Memcheck (Definitive)

Run with compute-sanitizer to get exact line number:

```bash
cd /path/to/Knowledge3D
conda activate k3d-cranium
compute-sanitizer --tool memcheck --log-file memcheck.log \
  python -m pytest tests/test_procedural_drawing_performance.py::test_gpu_rpn_operand_decoding -xvs

# Parse log
grep -A10 "Invalid" memcheck.log
# Look for PTX line number and instruction
```

**Output format:**
```
========= Invalid __global__ read of size 4
=========     at 0x000001a0 in execute_drawing_rpn
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f8a12345678 is misaligned
```

Then cross-reference `0x000001a0` with PTX to find exact instruction.

---

## Likely Culprits (My Bet)

**1. Parameter layout issue** (25% confidence)
- Try swapping parameter order: all u64 first, then u32, then f32
- Try removing `ternary_hint` parameter entirely

**2. Bounds check off-by-one** (30% confidence)
- Loop terminates at `idx >= prog_len`, but MOVE reads at idx+0..7
- If `prog_len = 9` and `idx = 1`, we read bytes 1-8 (valid)
- But what if opcode reading increments idx before checking?

**3. seg_ptr not aligned** (20% confidence)
- GPU malloc should return 256-byte aligned pointers
- But maybe loader.gpu_malloc has a bug?
- **Check:** Print `d_segments.value & 0xFF` in Python (should be 0)

**4. cvta.to.global.u64 produces wrong address** (15% confidence)
- Maybe we need `.global` directive on pointer registers?
- Compare with working kernels (procedural_glyph_rasterizer.ptx)

**5. STROKE opcode not handled** (10% confidence)
- Bytecode ends with 0x6A (STROKE)
- PTX has `bra LOOP` at line 80 for unknown opcodes
- This should be safe, but maybe loop doesn't terminate?

---

## Success Criteria

**Minimum (Phase 2A Pass):**
- [ ] `test_gpu_rpn_operand_decoding` passes (MOVE + LINE work)
- [ ] `test_gpu_quad_bezier` passes (QUAD tessellation works)
- [ ] No crashes, no misaligned address errors

**Nice to Have:**
- [ ] Latency < 100µs (may need buffer reuse optimization)
- [ ] All 4 tests pass

---

## Quick Wins to Try First

1. **Test MOVE-only:** `program = "0.5 0.5 MOVE"`
   - If passes: Issue is in LINE store logic
   - If fails: Issue is in MOVE operand read or parameter passing

2. **Remove ternary_hint parameter:**
   - Change PTX to have only 5 params
   - Change Python to pass only 5 params
   - Simplifies parameter layout

3. **Print addr before first store:**
   - Can't printf in PTX, but can store addr to d_count temporarily:
   ```ptx
   OPC_LINE:
       // ... compute addr
       st.global.u64 [cnt_ptr], addr;  // DEBUG: write addr instead of count
       bra WRITE_COUNT;  // Skip actual store
   ```
   - Read addr in Python, check if aligned

4. **Check GPU malloc alignment:**
   ```python
   d_test = loader.gpu_malloc(1024)
   print(f"Alignment: {d_test.value & 0xFF:#x}")  # Should be 0x00
   loader.gpu_free(d_test)
   ```

---

## Files to Modify

- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx` - Add guards/assertions
- `debug_kernel_exec.py` - Test minimal cases
- `tests/test_procedural_drawing_performance.py` - Maybe add simpler test cases

## Files for Reference

- `TEMP/PHASE_2A_DEBUG_STATUS.md` - Full debug history
- `knowledge3d/cranium/kernels/procedural_glyph_rasterizer.ptx` - Working PTX example
- `knowledge3d/cranium/sovereign/loader.py` - CUDA API

---

## Expected Time

- **Quick approach (1-2 hours):** Add assertions, test MOVE-only, print alignment
- **Thorough approach (3-4 hours):** Bisect kernel, run memcheck, fix root cause

---

## Reporting Back

When done, update:
1. `TEMP/PHASE_2A_DEBUG_STATUS.md` - Add "RESOLVED" section with fix
2. `TEMP/PROCEDURAL_DRAWING_PHASE_2A_COMPLETION_PROMPT.md` - Update status
3. Run all 4 tests: `pytest tests/test_procedural_drawing_performance.py -v`
4. Report latency numbers from passing tests

**If still blocked:** Document which approaches you tried, what errors you got, and any new theories.

---

**Go forth and debug! 🔧**
