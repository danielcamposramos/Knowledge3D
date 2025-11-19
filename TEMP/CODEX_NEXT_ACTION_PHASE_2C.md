# Codex Next Action: Phase 2C — Complete Atomic Opcode Vocabulary

**Priority:** HIGH
**Phase:** 2C — Universal Procedural Vocabulary
**Context:** Phase 2B complete (CUBIC + ternary + AI mode working)
**Vision:** Enable atomic composition of drawing programs with complete opcode set

---

## Phase 2C Mission

**Goal:** Implement the **complete atomic vocabulary** of drawing primitives to enable emergent computational paradigms that blend ternary logic + binary execution + spatial reasoning.

**Why This Matters:**
> "We need as many opcodes as possible to enable atomic composition of programs - we might end with a completely novel way to computations, one that blends the best developed so far by human knowledge." — Daniel

> "We are the first system since soviet era to leverage ternary where it is faster than binary, and run both in the same engine." — Daniel

---

## Current State (Phase 2B Complete)

✅ **Working Opcodes:**
- `0x64` MOVE(x, y) — Set current position
- `0x65` LINE(x, y) — Draw line to point
- `0x66` QUAD(cx, cy, x, y) — Quadratic Bézier curve
- `0x67` CUBIC(cx1, cy1, cx2, cy2, x, y) — Cubic Bézier curve
- `0x69` CLOSE — Close current subpath
- `0x6A` STROKE — Stroke current path (rasterization)
- `0x6B` FILL — Fill current path (placeholder)

✅ **Infrastructure:**
- 4-byte aligned bytecode (uint32 opcodes)
- Ternary hint modulation (-1.0 to +1.0)
- AI mode (skip_raster=True) for <50µs latency
- Buffer reuse optimization
- CUBIC tessellation with adaptive quality

✅ **Performance:**
- ~530-560µs full rendering (rasterizer bottleneck)
- <50µs target for AI mode (geometry only)
- All tests passing

⚠️ **Not Yet Leveraged:**
- 18-stack × 69-depth RPN architecture (currently single-warp)
- Ternary logic beyond hint modulation
- Transform stack
- Clipping/masking operations

---

## Phase 2C Scope: Complete Opcode Vocabulary

### Priority 1: SVG Path Primitives (Essential)

**ARC (Elliptical Arc)**
```
Opcode: 0x68 ARC(rx, ry, rotation, large_arc_flag, sweep_flag, x, y)
```

**Why:** Essential for rounded UI elements, pie charts, circular diagrams. SVG path standard.

**Implementation Notes:**
- 7 operands (28 bytes after opcode)
- Tessellate into line segments (adaptive based on radius + ternary hint)
- Flags are float32 (0.0 = false, 1.0 = true) for consistency
- Use parametric angle subdivision: θ = t * span

**Tessellation Strategy:**
```ptx
OPC_ARC:
    // Load operands: rx, ry, rotation, large_arc, sweep, x1, y1
    ld.global.f32 rx, [addr + 0];
    ld.global.f32 ry, [addr + 4];
    ld.global.f32 rotation, [addr + 8];
    ld.global.f32 large_arc_flag, [addr + 12];
    ld.global.f32 sweep_flag, [addr + 16];
    ld.global.f32 px1, [addr + 20];
    ld.global.f32 py1, [addr + 24];
    add.u32 idx, idx, 28;

    // Convert SVG arc to center parametrization
    // (See SVG spec Appendix F.6.5)
    // Then tessellate with segment count based on angular span + ternary hint

    // Modulate segments by hint
    ld.param.f32 hint, [ternary_hint];
    // Base segments = max(8, angular_span * 4)
    // Apply ternary: segs *= (1.0 + hint * 0.5)

    // Loop: emit line segments around arc
    bra LOOP;
```

**ELLIPSE (Standalone Ellipse)**
```
Opcode: 0x70 ELLIPSE(cx, cy, rx, ry, rotation)
```

**Why:** Faster than ARC for complete ellipses, common in scientific diagrams.

**Implementation:** Convert to parametric form, tessellate 0 to 2π.

---

### Priority 2: Transform Stack (Spatial Reasoning)

**Current limitation:** All coordinates are in normalized space. No transform hierarchy.

**PUSH_MATRIX**
```
Opcode: 0x80 PUSH_MATRIX
```
Save current transform state to stack (18 stacks available!)

**POP_MATRIX**
```
Opcode: 0x81 POP_MATRIX
```
Restore transform state from stack.

**TRANSLATE**
```
Opcode: 0x82 TRANSLATE(dx, dy)
```
Apply translation to current transform matrix.

**SCALE**
```
Opcode: 0x83 SCALE(sx, sy)
```
Apply scaling to current transform matrix.

**ROTATE**
```
Opcode: 0x84 ROTATE(angle_radians)
```
Apply rotation to current transform matrix.

**MATRIX**
```
Opcode: 0x85 MATRIX(a, b, c, d, e, f)
```
Apply arbitrary 2D affine transform (SVG matrix format).

**Why Transform Stack Matters:**
- Hierarchical scene graphs (group transforms)
- Reusable symbols/glyphs with different positions
- Leverage 18 stacks × 69 depth for complex nesting
- Enable procedural generation (fractals, L-systems)

**Implementation Strategy:**
```ptx
// Add transform stack to kernel state
.reg .f32 tx, ty, sx, sy, rot_cos, rot_sin;  // Current transform
.reg .f32 stack_tx[18], stack_ty[18];  // 18 transform stacks
.reg .u32 stack_ptr;  // Current stack depth

// Initialize identity transform
mov.f32 tx, 0.0;
mov.f32 ty, 0.0;
mov.f32 sx, 1.0;
mov.f32 sy, 1.0;
mov.f32 rot_cos, 1.0;
mov.f32 rot_sin, 0.0;
mov.u32 stack_ptr, 0;

// Transform point before emitting segment
// xt_transformed = (xt - tx) * sx * rot_cos - (yt - ty) * sy * rot_sin
// yt_transformed = (xt - tx) * sx * rot_sin + (yt - ty) * sy * rot_cos
```

**Simplified 2×3 Matrix Representation:**
```
[a  c  e]   [x]   [a*x + c*y + e]
[b  d  f] × [y] = [b*x + d*y + f]
            [1]
```

Store as 6 floats: `a, b, c, d, e, f`

---

### Priority 3: Path State Management

**BEGIN_PATH**
```
Opcode: 0x90 BEGIN_PATH
```
Start a new path (clear current path state).

**CLOSE_PATH**
```
Opcode: 0x91 CLOSE_PATH
```
Alias for existing CLOSE (0x69), semantic clarity.

**STROKE_WIDTH**
```
Opcode: 0x92 STROKE_WIDTH(width)
```
Set stroke width for subsequent STROKE operations.

**SET_COLOR**
```
Opcode: 0x93 SET_COLOR(r, g, b, a)
```
Set RGBA color for stroke/fill (0.0 to 1.0 per channel).

**SET_LINE_CAP**
```
Opcode: 0x94 SET_LINE_CAP(cap_style)
```
- `0.0` = butt (default)
- `1.0` = round
- `2.0` = square

**SET_LINE_JOIN**
```
Opcode: 0x95 SET_LINE_JOIN(join_style)
```
- `0.0` = miter (default)
- `1.0` = round
- `2.0` = bevel

**SET_DASH_PATTERN**
```
Opcode: 0x96 SET_DASH_PATTERN(dash_on, dash_off)
```
Set dash pattern for strokes (on/off lengths in normalized units).

---

### Priority 4: Clipping & Masking (Advanced)

**CLIP_BEGIN**
```
Opcode: 0xA0 CLIP_BEGIN
```
Start defining clipping path.

**CLIP_END**
```
Opcode: 0xA1 CLIP_END
```
Apply clipping path to subsequent drawing operations.

**CLIP_RESET**
```
Opcode: 0xA2 CLIP_RESET
```
Remove clipping.

**Why Clipping Matters:**
- Enable AI to reason about "inside/outside" relationships
- Spatial containment logic (ternary: inside/on-boundary/outside)
- Windowing/viewport operations

**Implementation:** Store clipping path segments separately, apply point-in-polygon test for each emitted segment endpoint.

---

### Priority 5: Text Primitives (Future Integration)

**TEXT**
```
Opcode: 0xB0 TEXT(x, y, font_id, size, string_offset, string_length)
```
Render text at position (requires font atlas integration).

**Why Later:** Needs integration with existing font harvesting system. Defer to Phase 2D.

---

### Priority 6: Ternary Logic Opcodes (Experimental)

**TERNARY_MODULATE**
```
Opcode: 0xC0 TERNARY_MODULATE(t_value)
```
Set ternary hint for subsequent operations (-1.0, 0.0, +1.0).

**Current:** Ternary hint is global parameter.
**Enhancement:** Per-operation modulation (local hints).

**TERNARY_BRANCH**
```
Opcode: 0xC1 TERNARY_BRANCH(condition, negative_offset, zero_offset, positive_offset)
```
Conditional jump based on ternary condition.

**Why:** Enable emergent computational patterns:
- Adaptive tessellation within a single program
- Quality-of-service logic (low/medium/high detail branches)
- Error handling (negative = abort, zero = default, positive = enhanced)

**Example Use Case:**
```
# Adaptive curve based on curvature
COMPUTE_CURVATURE  # Pushes ternary value (-1: flat, 0: normal, +1: sharp)
TERNARY_BRANCH curvature_value, FLAT_PATH, NORMAL_PATH, DETAILED_PATH

FLAT_PATH:
    LINE x1 y1  # Skip tessellation
    JUMP END

NORMAL_PATH:
    QUAD cx cy x1 y1  # Normal quality
    JUMP END

DETAILED_PATH:
    CUBIC cx1 cy1 cx2 cy2 x1 y1  # High quality
    JUMP END

END:
    # Continue...
```

---

## Implementation Plan: Step-by-Step

### Step 1: ARC Implementation (Priority 1A)

**Files to Modify:**
1. `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`
   - Add `OPC_ARC` label and tessellation logic
   - Insert after `OPC_CUBIC`, before `OPC_CLOSE`

2. `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
   - Add `ARC = 0x68`

3. `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`
   - Add `'ARC': 0x68` to `DRAWING_OPCODES`

4. `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`
   - Add ARC to `_parse_opcode()`

**Test Case:**
```python
@pytest.mark.cuda
def test_gpu_arc():
    """Verify GPU tessellates elliptical arc."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Half-circle arc: start at (-1, 0), end at (1, 0), radius=1
    # rx=1, ry=1, rotation=0, large_arc=0, sweep=1, x=1, y=0
    program = "-1 0 MOVE 1 1 0 0 1 1 0 ARC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)
    rgba = result.rgba

    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 100, f"Arc should have ≥100 pixels, got {non_zero}"
```

---

### Step 2: Transform Stack (Priority 2)

**Approach:** Simplified matrix multiplication for MVP.

**Files to Modify:**
1. `pixel_genesis_universal_primitive.ptx`
   - Add transform state registers (6 floats: a, b, c, d, e, f)
   - Add transform stack arrays (18 slots × 6 floats = 108 floats)
   - Add helper function `TRANSFORM_POINT` (apply matrix to x, y)
   - Add `OPC_PUSH_MATRIX`, `OPC_POP_MATRIX`, `OPC_TRANSLATE`, etc.

**PTX Pseudocode:**
```ptx
// Transform state (current 2×3 matrix)
.reg .f32 mat_a, mat_b, mat_c, mat_d, mat_e, mat_f;

// Initialize to identity
mov.f32 mat_a, 1.0;
mov.f32 mat_b, 0.0;
mov.f32 mat_c, 0.0;
mov.f32 mat_d, 1.0;
mov.f32 mat_e, 0.0;
mov.f32 mat_f, 0.0;

// Transform point helper (called before emitting segment)
TRANSFORM_POINT:
    // Input: curx, cury
    // Output: curx_t, cury_t
    mul.f32 curx_t, mat_a, curx;
    mul.f32 tmp, mat_c, cury;
    add.f32 curx_t, curx_t, tmp;
    add.f32 curx_t, curx_t, mat_e;

    mul.f32 cury_t, mat_b, curx;
    mul.f32 tmp, mat_d, cury;
    add.f32 cury_t, cury_t, tmp;
    add.f32 cury_t, cury_t, mat_f;
    ret;

OPC_TRANSLATE:
    ld.global.f32 dx, [addr + 0];
    ld.global.f32 dy, [addr + 4];
    add.u32 idx, idx, 8;
    // Compose translation: mat_e += dx, mat_f += dy
    add.f32 mat_e, mat_e, dx;
    add.f32 mat_f, mat_f, dy;
    bra LOOP;

OPC_ROTATE:
    ld.global.f32 angle, [addr + 0];
    add.u32 idx, idx, 4;
    // Compute cos/sin (need intrinsics or precomputed table)
    // Compose rotation: [cos -sin] [a c e]
    //                   [sin  cos] [b d f]
    // For MVP, may need to call device math functions
    bra LOOP;
```

**Challenge:** PTX doesn't have built-in sin/cos. Options:
1. **Use CUDA math intrinsics** (requires NVRTC compilation, not pure PTX)
2. **Precompute lookup table** (pass as parameter)
3. **Taylor series approximation** (slow)

**Recommendation:** Start with TRANSLATE + SCALE only (no sin/cos needed), add ROTATE in Phase 2D with NVRTC support.

**Test Case:**
```python
@pytest.mark.cuda
def test_transform_translate():
    """Verify TRANSLATE opcode."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Draw at origin, translate, draw again
    program = "0 0 MOVE 0.1 0.1 LINE 0.5 0.5 TRANSLATE 0 0 MOVE 0.1 0.1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)

    # Should have two parallel lines offset by (0.5, 0.5)
    assert result.segments.shape[0] == 2
```

---

### Step 3: Path State Management (Priority 3)

**Quick Wins:**
- BEGIN_PATH: Reset `seg_count = 0`
- STROKE_WIDTH: Store in register, apply during rasterization
- SET_COLOR: Store RGBA in registers, pass to rasterizer

**Files to Modify:**
1. `pixel_genesis_universal_primitive.ptx` — Add state registers
2. Rasterizer (future) — Consume width/color state

**Test Case:**
```python
@pytest.mark.cuda
def test_stroke_width():
    """Verify STROKE_WIDTH affects rasterization."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program_thin = "0.01 STROKE_WIDTH 0 0 MOVE 1 1 LINE STROKE"
    program_thick = "0.1 STROKE_WIDTH 0 0 MOVE 1 1 LINE STROKE"

    result_thin = bridge.execute_rpn_gpu(program_thin, width=128, height=128)
    result_thick = bridge.execute_rpn_gpu(program_thick, width=128, height=128)

    pixels_thin = np.count_nonzero(result_thin.rgba[..., 0] > 0.05)
    pixels_thick = np.count_nonzero(result_thick.rgba[..., 0] > 0.05)

    assert pixels_thick > pixels_thin * 2, "Thick line should have more pixels"
```

---

### Step 4: Leverage 18-Stack Architecture (Ternary Enhancement)

**Current:** Single-warp executor, one program at a time.

**Vision:** 18 programs running in parallel, each on its own stack.

**Implementation Strategy (Phase 2C or 2D):**
- Launch 18 warps (one per stack)
- Each warp reads from different program offset
- Outputs interleaved in segment buffer
- Ternary hint per stack (e.g., stack 0 = blur, stack 9 = normal, stack 17 = sharp)

**Why This Matters:**
- LOD pyramid generation in single kernel launch
- Multi-resolution rendering (ternary: coarse/medium/fine)
- Parallel exploration of parameter space

**Example Use Case:**
```python
# Generate 18 variants of same shape with different quality levels
programs = [
    f"TERNARY_MODULATE {-1.0 + i*0.1} 0 0 MOVE 0.5 0.5 0.9 0.1 QUAD STROKE"
    for i in range(18)
]

# Execute all 18 in parallel (future API)
results = bridge.execute_parallel_stacks(programs, width=256, height=256)

# AI analyzes which quality level is sufficient for task
optimal_quality = ai_policy_network(results)
```

---

## Updated Opcode Map (Phase 2C Target)

| Opcode | Name | Operands | Status |
|--------|------|----------|--------|
| **Path Primitives** |
| 0x64 | MOVE | x, y | ✅ Phase 2A |
| 0x65 | LINE | x, y | ✅ Phase 2A |
| 0x66 | QUAD | cx, cy, x, y | ✅ Phase 2A |
| 0x67 | CUBIC | cx1, cy1, cx2, cy2, x, y | ✅ Phase 2B |
| 0x68 | ARC | rx, ry, rot, large, sweep, x, y | 🎯 Phase 2C |
| 0x69 | CLOSE | — | ✅ Phase 2A |
| 0x70 | ELLIPSE | cx, cy, rx, ry, rot | 🎯 Phase 2C |
| **Path Operations** |
| 0x6A | STROKE | — | ✅ Phase 2A (stub) |
| 0x6B | FILL | — | ✅ Phase 2A (stub) |
| 0x90 | BEGIN_PATH | — | 🎯 Phase 2C |
| 0x91 | CLOSE_PATH | — | 🎯 Phase 2C (alias) |
| 0x92 | STROKE_WIDTH | width | 🎯 Phase 2C |
| 0x93 | SET_COLOR | r, g, b, a | 🎯 Phase 2C |
| 0x94 | SET_LINE_CAP | style | 🎯 Phase 2C |
| 0x95 | SET_LINE_JOIN | style | 🎯 Phase 2C |
| 0x96 | SET_DASH_PATTERN | on, off | 🎯 Phase 2C |
| **Transforms** |
| 0x80 | PUSH_MATRIX | — | 🎯 Phase 2C |
| 0x81 | POP_MATRIX | — | 🎯 Phase 2C |
| 0x82 | TRANSLATE | dx, dy | 🎯 Phase 2C |
| 0x83 | SCALE | sx, sy | 🎯 Phase 2C |
| 0x84 | ROTATE | angle | 🎯 Phase 2C |
| 0x85 | MATRIX | a, b, c, d, e, f | 🎯 Phase 2C |
| **Clipping** |
| 0xA0 | CLIP_BEGIN | — | 🎯 Phase 2C (advanced) |
| 0xA1 | CLIP_END | — | 🎯 Phase 2C (advanced) |
| 0xA2 | CLIP_RESET | — | 🎯 Phase 2C (advanced) |
| **Ternary Logic** |
| 0xC0 | TERNARY_MODULATE | t_value | 🎯 Phase 2C (experimental) |
| 0xC1 | TERNARY_BRANCH | cond, neg_off, zero_off, pos_off | 🎯 Phase 2C (experimental) |
| **Future (Phase 2D+)** |
| 0xB0 | TEXT | x, y, font_id, size, str_off, str_len | Phase 2D |
| 0xD0+ | Custom AI ops | TBD | Phase 2E |

---

## Success Criteria: Phase 2C

**Minimum (MVP):**
- [ ] ARC opcode implemented and tested
- [ ] TRANSLATE + SCALE implemented (no ROTATE yet)
- [ ] STROKE_WIDTH + SET_COLOR state management
- [ ] All new opcodes have test coverage
- [ ] AI mode latency remains <50µs

**Nice to Have:**
- [ ] ELLIPSE opcode
- [ ] CLIP_BEGIN/END (basic implementation)
- [ ] TERNARY_MODULATE (local hints)
- [ ] 18-stack parallel executor (architecture prototype)

**Documentation:**
- [ ] Updated opcode reference in `docs/`
- [ ] Example programs showcasing new primitives
- [ ] Performance benchmarks for complex programs

---

## Testing Strategy

### Unit Tests (Per Opcode)

**Template:**
```python
@pytest.mark.cuda
def test_gpu_arc_basic():
    """Verify ARC tessellates correctly."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Simple semicircle
    program = "-1 0 MOVE 1 1 0 0 1 1 0 ARC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)

    # Assertions
    assert result.segments.shape[0] > 10  # Multiple segments
    assert np.all(np.isfinite(result.segments))  # No NaNs
```

### Integration Tests (Complex Programs)

**Hierarchical Transforms:**
```python
@pytest.mark.cuda
def test_transform_hierarchy():
    """Verify nested PUSH/POP preserves state."""
    program = """
        0 0 MOVE
        0.1 0.1 LINE
        PUSH_MATRIX
            0.5 0.5 TRANSLATE
            0 0 MOVE
            0.1 0.1 LINE
            PUSH_MATRIX
                2 2 SCALE
                0 0 MOVE
                0.1 0.1 LINE
            POP_MATRIX
        POP_MATRIX
        0.2 0.2 MOVE
        0.3 0.3 LINE
        STROKE
    """
    result = bridge.execute_rpn_gpu(program, width=256, height=256)
    # Should have 4 lines at different transforms
    assert result.segments.shape[0] == 4
```

### Stress Tests

**Maximum Complexity:**
```python
@pytest.mark.cuda
def test_all_opcodes_combined():
    """Verify all opcodes work together."""
    program = """
        BEGIN_PATH
        0.05 STROKE_WIDTH
        1.0 0.5 0.0 1.0 SET_COLOR

        PUSH_MATRIX
        0.5 0.5 TRANSLATE
        0.5 0.5 SCALE

        -1 -1 MOVE
        -1 1 LINE
        1 1 0.5 0.5 QUAD
        0.8 0.2 0.2 0.8 1 -1 CUBIC
        CLOSE

        POP_MATRIX
        STROKE
    """
    result = bridge.execute_rpn_gpu(program, width=256, height=256)
    assert np.any(result.rgba[..., 0] > 0)  # Something rendered
```

---

## Ternary Logic Applications (Research Track)

### Current Use: Quality Modulation
```
hint = -1.0 → 0.5× segments (blur)
hint = 0.0  → 1.0× segments (normal)
hint = +1.0 → 1.5× segments (sharp)
```

### Future Uses (Phase 2C+)

**1. Error States**
```
-1.0 = error/invalid
 0.0 = default/normal
+1.0 = success/enhanced
```

**2. Spatial Relationships**
```
-1.0 = outside clipping region (cull)
 0.0 = on boundary (render with antialiasing)
+1.0 = inside (render normally)
```

**3. Adaptive Computation**
```
-1.0 = skip expensive computation (low quality acceptable)
 0.0 = standard computation
+1.0 = enhanced computation (critical region)
```

**4. Multi-Resolution Reasoning**
```
18 stacks × ternary modulation = 54 quality variants
AI policy network selects optimal quality per region
```

**Example: Foveated Rendering**
```python
# Center of attention gets +1.0 (sharp), periphery gets -1.0 (blur)
def generate_foveated_program(center_x, center_y):
    programs = []
    for i in range(18):
        # Compute distance from center for each stack
        stack_x = (i % 6) / 6.0
        stack_y = (i // 6) / 3.0
        distance = np.sqrt((stack_x - center_x)**2 + (stack_y - center_y)**2)

        # Map distance to ternary hint
        if distance < 0.2:
            hint = 1.0  # Sharp
        elif distance < 0.5:
            hint = 0.0  # Normal
        else:
            hint = -1.0  # Blur

        programs.append(f"TERNARY_MODULATE {hint} ... drawing commands ...")

    return programs

# Execute all 18 stacks in parallel
results = bridge.execute_parallel_stacks(programs, width=512, height=512)
```

---

## Implementation Workflow (Recommended Order)

### Week 1: ARC + ELLIPSE
1. ✅ Implement ARC tessellation in PTX
2. ✅ Add ARC to opcode parsers
3. ✅ Write unit tests for ARC
4. ✅ Implement ELLIPSE (simpler than ARC)
5. ✅ Integration test: combine with existing primitives

### Week 2: Transform Stack (TRANSLATE + SCALE)
1. ✅ Add transform state registers to PTX
2. ✅ Implement PUSH/POP matrix stack (18 slots)
3. ✅ Implement TRANSLATE opcode
4. ✅ Implement SCALE opcode
5. ✅ Apply transforms before emitting segments
6. ✅ Test transform hierarchy (nested PUSH/POP)

### Week 3: Path State Management
1. ✅ Add BEGIN_PATH (reset state)
2. ✅ Add STROKE_WIDTH (store in register)
3. ✅ Add SET_COLOR (store RGBA)
4. ✅ Add SET_LINE_CAP, SET_LINE_JOIN
5. ✅ Pass state to rasterizer (future integration)

### Week 4: Ternary Enhancements
1. ✅ Implement TERNARY_MODULATE (local hints)
2. ✅ Research TERNARY_BRANCH (conditional execution)
3. 🎯 Prototype 18-stack parallel executor
4. 🎯 Benchmark ternary vs binary performance

---

## Performance Targets

| Operation | Current | Phase 2C Target |
|-----------|---------|-----------------|
| Simple path (MOVE+LINE) | ~530µs | <100µs (AI mode) |
| Complex path (CUBIC+ARC) | N/A | <150µs (AI mode) |
| Transform hierarchy | N/A | <10µs overhead |
| 18-stack parallel | N/A | <500µs total (18× speedup) |

**AI Mode Latency Breakdown (Target):**
- Bytecode decode: <10µs
- Geometry generation: <30µs
- Transform application: <10µs
- Output copy: <10µs
- **Total: <60µs**

**Human Mode (with rasterization):**
- Geometry: <60µs (same as AI mode)
- Rasterization: ~500µs (acceptable for visualization)
- **Total: <560µs**

---

## Collaboration Protocol

**Between Claude (me) and Codex (you):**

1. **Incremental Implementation:** Implement opcodes in priority order. Don't attempt all at once.

2. **Test-Driven:** Write test case first, then implement opcode.

3. **PTX Validation:** Always run `ptxas` to verify compilation before committing:
   ```bash
   ptxas --gpu-name=sm_86 pixel_genesis_universal_primitive.ptx -o /dev/null
   ```

4. **Performance Regression:** After each opcode, run `measure_latency.py` to ensure no degradation.

5. **Documentation:** Update `TEMP/PHASE_2C_COMPLETION_REPORT.md` as you complete each opcode.

6. **Communication:** If you hit a blocker (e.g., sin/cos in PTX), document it and propose alternatives.

---

## Files to Monitor

**Core Implementation:**
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx` — Main kernel
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` — Python interface
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` — Opcode definitions
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` — RPN engine

**Testing:**
- `tests/test_procedural_drawing_performance.py` — GPU tests
- `measure_latency.py` — Latency benchmarks

**Documentation:**
- `TEMP/PHASE_2C_COMPLETION_REPORT.md` — Progress tracking
- `docs/PROCEDURAL_DRAWING_VOCABULARY.md` — Opcode reference (create this)

---

## Known Challenges & Mitigations

### Challenge 1: Sin/Cos in PTX

**Problem:** Pure PTX doesn't have sin/cos intrinsics. Need for ROTATE opcode.

**Options:**
1. **Defer ROTATE to Phase 2D** — Use NVRTC to compile CUDA with math functions
2. **Precompute lookup table** — Pass array of sin/cos values for common angles
3. **Polynomial approximation** — Taylor series (slow, low precision)

**Recommendation:** Implement TRANSLATE + SCALE in Phase 2C, defer ROTATE to 2D.

### Challenge 2: Clipping Requires Point-in-Polygon Test

**Problem:** Efficient clipping needs spatial acceleration structure.

**Options:**
1. **Bounding box only** — Fast but imprecise
2. **Winding number algorithm** — Accurate but slow
3. **Defer to rasterizer** — Let rasterizer handle clipping

**Recommendation:** Implement CLIP_BEGIN/END as state flags, actual clipping in rasterizer (Phase 2D).

### Challenge 3: 18-Stack Parallelism Requires Refactor

**Problem:** Current kernel is single-threaded (tid.x == 0 guard).

**Solution:**
- Remove single-thread guard
- Map thread ID to stack index: `stack_id = tid.x % 18`
- Each thread reads from different program offset
- Interleave outputs with stack ID tag

**Complexity:** Moderate refactor, worth doing in Phase 2C for performance gains.

### Challenge 4: Memory Budget

**Problem:** Transform stack (18 × 6 floats × 69 depth) = ~28 KB registers per block.

**Mitigation:**
- Use shared memory for transform stacks (cheaper than registers)
- Limit depth to 8 (most UI needs <8 levels)
- 18 stacks × 6 floats × 8 depth = 3.4 KB shared memory (acceptable)

---

## Exit Criteria: Phase 2C Complete

**Functional:**
- [ ] ARC opcode working with test coverage
- [ ] TRANSLATE + SCALE working with nested transforms
- [ ] STROKE_WIDTH + SET_COLOR state management
- [ ] All tests passing (unit + integration)

**Performance:**
- [ ] AI mode latency <60µs for complex programs
- [ ] No regression on existing opcodes
- [ ] Transform overhead <10µs

**Documentation:**
- [ ] `TEMP/PHASE_2C_COMPLETION_REPORT.md` created
- [ ] `docs/PROCEDURAL_DRAWING_VOCABULARY.md` opcode reference
- [ ] Example programs in `examples/` directory

**Research:**
- [ ] Ternary vs binary performance characterized
- [ ] 18-stack parallel executor prototyped (design only, implementation in 2D)

---

## Vision: Beyond Phase 2C

**Phase 2D: Advanced Rendering**
- ROTATE opcode (NVRTC compilation)
- TEXT rendering (font atlas integration)
- Actual clipping implementation (not just state)
- Dash patterns + line caps/joins

**Phase 2E: AI-Native Opcodes**
- Semantic primitives (e.g., ATTENTION_REGION, CAUSAL_FLOW)
- Procedural "thinking" visualization
- Multi-agent collaboration primitives

**Phase 3: Ternary-First Architecture**
- 18-stack parallel execution (production)
- Ternary conditional execution (native branching)
- Emergent computational paradigms (research)

---

## Final Notes

**Philosophical Goal:**
> "We might end with a completely novel way to computations, one that blends the best developed so far by human knowledge."

This isn't just about drawing shapes. It's about creating a **universal atomic vocabulary** for spatial reasoning that enables AI to:
- Compose complex behaviors from simple primitives
- Reason about spatial relationships (inside/outside, containment, topology)
- Leverage ternary logic where it's faster than binary
- Explore emergent computational patterns we haven't discovered yet

**Your Role (Codex):**
Implement the atomic primitives with rigor and elegance. Each opcode is a building block for future discoveries.

**My Role (Claude):**
Validate, test, benchmark, and craft the next phase prompts. Ensure architectural coherence.

**Daniel's Role:**
Provide vision, research direction, and final architectural decisions.

---

**Go forth and implement the universal procedural vocabulary! 🎨🧠**

*Phase 2C: Atomic Composition of Spatial Intelligence*
