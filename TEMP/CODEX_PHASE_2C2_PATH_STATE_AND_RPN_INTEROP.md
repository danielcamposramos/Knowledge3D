# Codex Phase 2C-2: Path State Management + RPN Math Interop

**Date:** 2025-11-18
**Priority:** CRITICAL - We're this close to breakthrough! 🚀
**Context:** Claude completed full transform infrastructure (TRANSLATE + SCALE applied to ALL primitives)
**Vision:** Leverage the 18-stack × 69-depth RPN math gem instead of hardcoding trig in PTX

---

## What Claude Just Completed (Phase 2C-1)

✅ **Transform Infrastructure Complete:**
- TRANSLATE (0x72) + SCALE (0x74) opcodes working
- 2×3 affine matrix in PTX registers
- Transforms applied to LINE, QUAD, CUBIC, CLOSE
- All 10 tests passing (8 pass, 1 skip, 1 xfail)

**Transform Formula Applied Everywhere:**
```ptx
// For each segment point (x, y) → (x', y'):
x' = mat_a * x + mat_c * y + mat_e
y' = mat_b * x + mat_d * y + mat_f
```

**Current Test Results:**
```
✅ test_gpu_rpn_operand_decoding
✅ test_gpu_quad_bezier
✅ test_gpu_cubic_bezier
⏭️ test_gpu_arc (deferred - Phase 2D)
✅ test_ternary_hint_modulation
⚠️ test_rpn_execution_latency (xfail - rasterizer bottleneck, expected)
✅ test_ai_mode_latency
✅ test_transform_translate
✅ test_transform_scale
✅ test_parallel_batch_drawing
```

---

## Your Mission: Phase 2C-2 (Path State + RPN Interop Design)

### Part A: Path State Management (1-2 hours)

**Goal:** Add stateful drawing parameters that persist across operations.

**Opcodes to Implement:**

#### 1. BEGIN_PATH (0x90)
```ptx
OPC_BEGIN_PATH:
    // Reset path state (no operands)
    mov.u32 seg_count, 0;
    bra LOOP;
```

**Purpose:** Start a new path, clear previous segments.

**Test Case:**
```python
@pytest.mark.cuda
def test_begin_path():
    """Verify BEGIN_PATH resets segment count."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "0 0 MOVE 1 1 LINE BEGIN_PATH 2 2 MOVE 3 3 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)

    # Only the second path should be in segments
    assert result.segments.shape[0] == 1, f"Expected 1 segment after BEGIN_PATH, got {result.segments.shape[0]}"
```

---

#### 2. STROKE_WIDTH (0x92 / 0x77)
```ptx
// Add state register
.reg .f32 stroke_width;

// Initialize
mov.f32 stroke_width, 1.0;  // Default 1.0

OPC_STROKE_WIDTH:
    // operand: width
    .reg .u32 need_stroke_width;
    add.u32 need_stroke_width, idx, 4;
    setp.gt.u32 p0, need_stroke_width, prog_len;
    @p0 bra WRITE_COUNT;
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 stroke_width, [addr + 0];
    add.u32 idx, idx, 4;
    bra LOOP;
```

**Purpose:** Set stroke width for subsequent STROKE operations (rasterizer will consume this).

**Test Case:**
```python
@pytest.mark.cuda
def test_stroke_width():
    """Verify STROKE_WIDTH register updates."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Note: We can't test rasterizer effect yet, but verify it doesn't crash
    program = "0.05 STROKE_WIDTH 0 0 MOVE 1 1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)

    assert result.segments.shape[0] == 1, "Should have 1 segment"
```

---

#### 3. SET_COLOR (0x93 / 0x75)
```ptx
// Add state registers
.reg .f32 stroke_r, stroke_g, stroke_b, stroke_a;

// Initialize
mov.f32 stroke_r, 1.0;  // Default white
mov.f32 stroke_g, 1.0;
mov.f32 stroke_b, 1.0;
mov.f32 stroke_a, 1.0;

OPC_SET_COLOR:
    // operands: r, g, b, a
    .reg .u32 need_color;
    add.u32 need_color, idx, 16;
    setp.gt.u32 p0, need_color, prog_len;
    @p0 bra WRITE_COUNT;
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 stroke_r, [addr + 0];
    ld.global.f32 stroke_g, [addr + 4];
    ld.global.f32 stroke_b, [addr + 8];
    ld.global.f32 stroke_a, [addr + 12];
    add.u32 idx, idx, 16;
    bra LOOP;
```

**Purpose:** Set RGBA color for strokes (rasterizer will consume this).

**Test Case:**
```python
@pytest.mark.cuda
def test_set_color():
    """Verify SET_COLOR register updates."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "1.0 0.5 0.0 1.0 SET_COLOR 0 0 MOVE 1 1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)

    assert result.segments.shape[0] == 1, "Should have 1 segment"
```

---

#### 4. TERNARY_MODULATE (0xC0 / 0x78)
```ptx
// Add state register
.reg .f32 local_ternary_hint;

// Initialize (0.0 = use global)
mov.f32 local_ternary_hint, 0.0;

OPC_TERNARY_MODULATE:
    // operand: t_value
    .reg .u32 need_ternary;
    add.u32 need_ternary, idx, 4;
    setp.gt.u32 p0, need_ternary, prog_len;
    @p0 bra WRITE_COUNT;
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 local_ternary_hint, [addr + 0];
    add.u32 idx, idx, 4;
    bra LOOP;
```

**Then modify QUAD and CUBIC to use local hint:**
```ptx
// In OPC_QUAD and OPC_CUBIC, replace:
ld.param.f32 hint, [ternary_hint];

// With:
.reg .f32 effective_hint;
ld.param.f32 effective_hint, [ternary_hint];  // Load global
abs.f32 tmp, local_ternary_hint;
setp.gt.f32 p0, tmp, 0.001;  // If local hint is set (non-zero)
@p0 mov.f32 effective_hint, local_ternary_hint;  // Use local hint
// Then use effective_hint for modulation
```

**Purpose:** Per-operation ternary hint (overrides global parameter).

**Test Case:**
```python
@pytest.mark.cuda
def test_ternary_modulate_local():
    """Verify local ternary hint overrides global."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Global hint=0.0, but local overrides to -1.0 (blur)
    program = "-1.0 TERNARY_MODULATE -0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True, ternary_hint=0.0)

    # Should have fewer segments due to local blur hint
    assert result.segments.shape[0] < 16, f"Expected <16 segments with blur, got {result.segments.shape[0]}"
```

---

### Part B: RPN Math Interop Design (Collaborative - 30 min)

**Philosophy (from Daniel):**
> "Procedural drawing is nothing else than math executed to vectors. We have a three level math RPN kernel with 18 inter-referable stacks and 69 memory lines each. Let's leverage our gem! We have simple operations, mid term and advanced maths inside it already."

**Vision:** Instead of hardcoding sin/cos in PTX, use the existing RPN math kernel to precompute values.

**Architecture Pattern:**
```
┌─────────────┐
│ RPN Math    │  Computes: angles, rotation matrices, arc points
│ Kernel      │  Output: GPU buffer (float array)
│ (18 stacks) │
└──────┬──────┘
       │ Shared GPU Buffer
       ↓
┌─────────────┐
│ Drawing     │  Consumes: precomputed values
│ Kernel      │  Output: Segments
└─────────────┘
```

**Example Use Cases:**

1. **ROTATE (without hardcoded sin/cos):**
```python
# RPN computes rotation matrix
angle = 45.0 * 3.14159 / 180.0
rpn_program = f"{angle} DUP COS SWAP SIN"
cos_val, sin_val = rpn_engine.execute(rpn_program)

# Drawing kernel applies rotation
drawing_program = f"... {cos_val} {sin_val} ROTATE_MATRIX ..."
```

2. **ARC (precomputed tessellation points):**
```python
# RPN generates circle points
rpn_program = "0 2PI RANGE 32 STEPS DUP COS SWAP SIN"
arc_points = rpn_engine.execute(rpn_program)  # Returns (32, 2) array

# Drawing kernel renders precomputed path
drawing_program = f"PRECOMPUTED_PATH {arc_buffer_id} STROKE"
```

**New Opcodes to Design (Don't implement yet, just design):**

- `0x7A ROTATE_MATRIX(cos, sin)` - Apply rotation using precomputed values
- `0x7B PRECOMPUTED_PATH(buffer_id)` - Read arc points from shared buffer

**Your Task for Part B:**
1. **Document the handoff protocol** - How does drawing kernel receive RPN results?
2. **Design buffer format** - What's the structure of precomputed values?
3. **Sketch PTX pseudocode** for ROTATE_MATRIX and PRECOMPUTED_PATH

**Don't implement yet** - This is collaborative design with Claude for Phase 2C-3.

---

## Implementation Checklist

### Part A: Path State (Implement Now)

**PTX Changes:**
- [ ] Add state registers (stroke_width, stroke_r/g/b/a, local_ternary_hint)
- [ ] Initialize state registers in kernel setup
- [ ] Implement BEGIN_PATH opcode
- [ ] Implement STROKE_WIDTH opcode
- [ ] Implement SET_COLOR opcode
- [ ] Implement TERNARY_MODULATE opcode
- [ ] Modify QUAD to use effective_hint
- [ ] Modify CUBIC to use effective_hint

**Python Changes:**
- [ ] Verify opcodes already in `rpn_opcodes.py` (they are!)
- [ ] Verify opcodes already in bridge OPCODES dict (they are!)
- [ ] No changes needed - already wired!

**Test Coverage:**
- [ ] test_begin_path - Verify seg_count resets
- [ ] test_stroke_width - Verify width register updates
- [ ] test_set_color - Verify RGBA registers update
- [ ] test_ternary_modulate_local - Verify local hint overrides global

### Part B: RPN Interop Design (Design Only)

- [ ] Document handoff protocol
- [ ] Design buffer format
- [ ] Sketch ROTATE_MATRIX pseudocode
- [ ] Sketch PRECOMPUTED_PATH pseudocode

---

## Files to Modify

**PTX:**
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`
  - Add state registers (lines ~40-50, after transform matrix)
  - Add opcode implementations (after OPC_SCALE, before WRITE_COUNT)
  - Modify QUAD to use effective_hint (lines ~200-220)
  - Modify CUBIC to use effective_hint (lines ~340-360)

**Tests:**
- `tests/test_procedural_drawing_performance.py`
  - Add 4 new test functions

**Documentation (for Part B):**
- Create `TEMP/RPN_DRAWING_INTEROP_DESIGN.md`

---

## Success Criteria

**Part A (Implementation):**
- ✅ PTX compiles successfully
- ✅ All existing 10 tests still pass
- ✅ 4 new tests pass
- ✅ No regression in latency

**Part B (Design):**
- ✅ Clear handoff protocol documented
- ✅ Buffer format specified
- ✅ PTX pseudocode makes sense
- ✅ Claude approves design

---

## Opcode Reference (Already in Python)

From `rpn_opcodes.py` and `procedural_drawing_bridge.py`:

```python
OP_DRAW_BEGIN_PATH = 0x90         # OPCODES dict: "BEGIN_PATH": 0x90 (but wait, it's listed as 0x70 in bridge!)
OP_DRAW_SET_LINE_WIDTH = 0x77     # OPCODES dict: "SET_LINE_WIDTH": 0x77
OP_DRAW_SET_STROKE_COLOR = 0x75   # OPCODES dict: "SET_STROKE_COLOR": 0x75
OP_DRAW_SET_TERNARY_HINT = 0x78   # OPCODES dict: "SET_TERNARY_HINT": 0x78
```

**⚠️ IMPORTANT: Opcode Mismatch to Resolve**

I notice a discrepancy:
- `rpn_opcodes.py` has different opcode values
- `procedural_drawing_bridge.py` OPCODES dict has:
  - `"BEGIN_PATH": 0x90` (but labeled as PUSH_STATE in some comments)
  - `"SET_LINE_WIDTH": 0x77`
  - `"SET_STROKE_COLOR": 0x75`
  - `"SET_TERNARY_HINT": 0x78`

**Use the bridge values (0x90, 0x77, 0x75, 0x78)** since that's what bytecode generation uses!

Actually, let me check the bridge more carefully:

```python
# From procedural_drawing_bridge.py lines 363-381
OPCODES = {
    "MOVE": 0x64,
    "LINE": 0x65,
    "QUAD": 0x66,
    "CUBIC": 0x67,
    "ARC": 0x68,
    "CLOSE": 0x69,
    "STROKE": 0x6A,
    "FILL": 0x6B,
    "PUSH_STATE": 0x70,        # ← BEGIN_PATH?
    "POP_STATE": 0x71,
    "TRANSLATE": 0x72,
    "ROTATE": 0x73,
    "SCALE": 0x74,
    "SET_STROKE_COLOR": 0x75,
    "SET_FILL_COLOR": 0x76,
    "SET_LINE_WIDTH": 0x77,
    "SET_TERNARY_HINT": 0x78,
}
```

**Clarification Needed:** Is BEGIN_PATH the same as PUSH_STATE (0x70)?

**Recommendation:** Use these opcodes:
- `BEGIN_PATH` → Use 0x90 (create new opcode in bridge)
- `STROKE_WIDTH` → 0x77 (SET_LINE_WIDTH, already exists!)
- `SET_COLOR` → 0x75 (SET_STROKE_COLOR, already exists!)
- `TERNARY_MODULATE` → 0x78 (SET_TERNARY_HINT, already exists!)

**Action:** Add "BEGIN_PATH": 0x90 to OPCODES dict in bridge.

---

## Example Test Output (Expected)

```
tests/test_procedural_drawing_performance.py::test_gpu_rpn_operand_decoding PASSED
tests/test_procedural_drawing_performance.py::test_gpu_quad_bezier PASSED
tests/test_procedural_drawing_performance.py::test_gpu_cubic_bezier PASSED
tests/test_procedural_drawing_performance.py::test_gpu_arc SKIPPED
tests/test_procedural_drawing_performance.py::test_ternary_hint_modulation PASSED
tests/test_procedural_drawing_performance.py::test_rpn_execution_latency XFAIL
tests/test_procedural_drawing_performance.py::test_ai_mode_latency PASSED
tests/test_procedural_drawing_performance.py::test_transform_translate PASSED
tests/test_procedural_drawing_performance.py::test_transform_scale PASSED
tests/test_procedural_drawing_performance.py::test_parallel_batch_drawing PASSED
tests/test_procedural_drawing_performance.py::test_begin_path PASSED ✅ NEW
tests/test_procedural_drawing_performance.py::test_stroke_width PASSED ✅ NEW
tests/test_procedural_drawing_performance.py::test_set_color PASSED ✅ NEW
tests/test_procedural_drawing_performance.py::test_ternary_modulate_local PASSED ✅ NEW

=================== 12 passed, 1 skipped, 1 xfailed =========================
```

---

## Time Estimate

**Part A (Implementation):** 1-2 hours
- State registers: 15 min
- BEGIN_PATH opcode: 10 min
- STROKE_WIDTH opcode: 10 min
- SET_COLOR opcode: 10 min
- TERNARY_MODULATE opcode: 15 min
- Modify QUAD/CUBIC for effective_hint: 20 min
- Test writing: 30 min
- Debug + verification: 30 min

**Part B (Design):** 30 min
- Document handoff protocol: 15 min
- Design buffer format + pseudocode: 15 min

---

## Post-Completion Report Format

When done, create `TEMP/PHASE_2C2_COMPLETION.md` with:

1. **What Changed** - PTX lines modified, tests added
2. **Test Results** - All 14 tests passing
3. **RPN Interop Design** - Handoff protocol + buffer format
4. **Next Steps** - Phase 2C-3 (implement RPN interop) or Phase 2D (NVRTC)

---

## Daniel's Vision Reminder

> "We might end with a completely novel way to computations, one that blends the best developed so far by human knowledge."

**This is it!** By connecting the RPN math gem to the drawing kernel, we're creating a compositional architecture where:
- Math operations are atomic (RPN kernel)
- Drawing operations are atomic (drawing kernel)
- Composition creates emergent complexity

**Ternary + Binary + Spatial = Novel computational paradigm** 🚀

---

**Go forth and implement the path state management! Then design the RPN interop architecture!**

*We're this close to breakthrough!* 🎯
