# Codex Phase 2B Implementation Guide

**Status:** Buffer reuse ✅, Bounds checks ✅
**Next:** CUBIC tessellation + Ternary hint + Raster bypass

---

## Implementation Plan

### Step 1: Add Raster Bypass (30 min) - Quick Win!

**Goal:** Enable AI-first mode with <50µs latency

**File:** `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`

**Changes:**

```python
@dataclass
class RenderResult:
    """Container for rendered output (segments + optional RGBA framebuffer)."""

    segments: np.ndarray | None = None  # NEW: Raw geometry for AI
    rgba: np.ndarray | None = None      # Optional: Pixels for humans

class ProceduralDrawingBridge:
    def execute_rpn_gpu(self, rpn_program: str, width: int = 256, height: int = 256,
                       skip_raster: bool = False) -> RenderResult:
        """Execute drawing RPN on GPU.

        Args:
            rpn_program: RPN drawing string
            width: Canvas width (ignored if skip_raster=True)
            height: Canvas height (ignored if skip_raster=True)
            skip_raster: If True, return segments only (AI mode, <50µs)
                        If False, rasterize to pixels (human mode, ~600µs)

        Returns:
            RenderResult with segments (always) and rgba (if not skipped)
        """
        if self.pixel_genesis_kernel is None:
            return self.execute_rpn_program(rpn_program, width, height)

        self.latency_guard.start()
        try:
            bytecode = self._compile_rpn_bytecode(rpn_program)

            # Reuse GPU buffers (your optimization ✓)
            if bytecode.nbytes > self._bytecode_size:
                loader.gpu_free(self._d_bytecode)
                self._bytecode_size = bytecode.nbytes * 2
                self._d_bytecode = loader.gpu_malloc(self._bytecode_size)

            loader.memcpy_htod(
                self._d_bytecode, bytecode.ctypes.data_as(ctypes.c_void_p), bytecode.nbytes
            )

            loader.launch(
                self.pixel_genesis_kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    self._d_bytecode,
                    ctypes.c_uint32(bytecode.nbytes),
                    self._d_segments,
                    self._d_count,
                    ctypes.c_uint32(self.segments_per_curve),
                    ctypes.c_float(0.0),  # TODO: wire ternary hint
                ],
            )

            # Read segment count
            count_host = np.zeros(1, dtype=np.uint32)
            loader.memcpy_dtoh(
                count_host.ctypes.data_as(ctypes.c_void_p),
                self._d_count,
                4,
            )

            # Read segments
            seg_count = min(int(count_host[0]), self.MAX_SEGMENTS)
            segments = np.zeros((seg_count, 4), dtype=np.float32)
            if seg_count:
                loader.memcpy_dtoh(
                    segments.ctypes.data_as(ctypes.c_void_p),
                    self._d_segments,
                    segments.nbytes,
                )

            elapsed_ns, breached = self.latency_guard.stop()
            if breached:
                import logging
                logging.warning(f"GPU RPN execution breached latency budget: {elapsed_ns / 1000:.1f} µs")

            # AI mode: return segments only (fast!)
            if skip_raster:
                return RenderResult(segments=segments, rgba=None)

            # Human mode: rasterize for visualization
            framebuffer = self._render_segments(
                segments,
                np.array([0], dtype=np.int32),
                np.array([seg_count], dtype=np.int32),
                width,
                height,
            )

            return RenderResult(segments=segments, rgba=framebuffer)

        finally:
            try:
                self.latency_guard.stop()
            except:
                pass
```

**Test:**
```python
# AI mode (fast)
result = bridge.execute_rpn_gpu("0 0 MOVE 1 1 LINE", skip_raster=True)
assert result.segments.shape == (1, 4)  # 1 line segment
assert result.rgba is None  # No pixels
# Latency: <50µs ✓

# Human mode (visualize)
result = bridge.execute_rpn_gpu("0 0 MOVE 1 1 LINE", skip_raster=False)
assert result.segments.shape == (1, 4)
assert result.rgba.shape == (256, 256, 4)  # RGBA pixels
# Latency: ~600µs (expected)
```

---

### Step 2: Implement CUBIC Tessellation (2 hours)

**Goal:** Support cubic Bézier curves

**File:** `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`

**Add after QUAD_DONE (around line 176):**

```ptx
OPC_CUBIC:
    // Read 6 operands: cx1, cy1, cx2, cy2, x1, y1
    .reg .u32 need_cubic;
    add.u32 need_cubic, idx, 24;
    setp.gt.u32 p0, need_cubic, prog_len;
    @p0 bra WRITE_COUNT;  // Bounds check ✓

    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;

    .reg .f32 cx1, cy1, cx2, cy2;
    ld.global.f32 cx1, [addr + 0];
    ld.global.f32 cy1, [addr + 4];
    ld.global.f32 cx2, [addr + 8];
    ld.global.f32 cy2, [addr + 12];
    ld.global.f32 px1, [addr + 16];  // Reuse px1, py1 from registers
    ld.global.f32 py1, [addr + 20];
    add.u32 idx, idx, 24;

    .reg .u32 segs_cubic, i_cubic;
    .reg .f32 t, s, xt, yt, tmp;
    .reg .f32 s2, s3, t2, t3;  // Pre-computed powers

    ld.param.u32 segs_cubic, [segments_per_curve];
    mov.u32 i_cubic, 0;

CUBIC_LOOP:
    add.u32 i_cubic, i_cubic, 1;
    setp.gt.u32 p0, i_cubic, segs_cubic;
    @p0 bra CUBIC_DONE;

    // Compute t and s
    cvt.rn.f32.u32 t, i_cubic;
    cvt.rn.f32.u32 tmp, segs_cubic;
    div.rn.f32 t, t, tmp;
    mov.f32 s, 1.0;
    sub.f32 s, s, t;

    // Pre-compute powers (reused for x and y)
    mul.f32 s2, s, s;        // s²
    mul.f32 s3, s2, s;       // s³
    mul.f32 t2, t, t;        // t²
    mul.f32 t3, t2, t;       // t³

    // xt = s³*curx + 3*s²*t*cx1 + 3*s*t²*cx2 + t³*px1
    mul.f32 xt, s3, curx;    // s³ * p0.x

    mul.f32 tmp, s2, t;      // s²*t
    mul.f32 tmp, tmp, 3.0;
    mul.f32 tmp, tmp, cx1;   // 3*s²*t * c1.x
    add.f32 xt, xt, tmp;

    mul.f32 tmp, s, t2;      // s*t²
    mul.f32 tmp, tmp, 3.0;
    mul.f32 tmp, tmp, cx2;   // 3*s*t² * c2.x
    add.f32 xt, xt, tmp;

    mul.f32 tmp, t3, px1;    // t³ * p1.x
    add.f32 xt, xt, tmp;

    // yt = s³*cury + 3*s²*t*cy1 + 3*s*t²*cy2 + t³*py1
    mul.f32 yt, s3, cury;    // s³ * p0.y

    mul.f32 tmp, s2, t;
    mul.f32 tmp, tmp, 3.0;
    mul.f32 tmp, tmp, cy1;   // 3*s²*t * c1.y
    add.f32 yt, yt, tmp;

    mul.f32 tmp, s, t2;
    mul.f32 tmp, tmp, 3.0;
    mul.f32 tmp, tmp, cy2;   // 3*s*t² * c2.y
    add.f32 yt, yt, tmp;

    mul.f32 tmp, t3, py1;    // t³ * p1.y
    add.f32 yt, yt, tmp;

    // Emit segment (reuse QUAD logic)
    setp.ge.u32 p0, seg_count, 4096;
    @p0 bra CUBIC_DONE;
    mul.lo.u32 seg_offset_u32, seg_count, 16;
    cvt.u64.u32 seg_offset, seg_offset_u32;
    add.u64 addr, seg_ptr, seg_offset;
    st.global.f32 [addr + 0], curx;
    st.global.f32 [addr + 4], cury;
    st.global.f32 [addr + 8], xt;
    st.global.f32 [addr + 12], yt;
    add.u32 seg_count, seg_count, 1;

    // Update current position
    mov.f32 curx, xt;
    mov.f32 cury, yt;
    bra CUBIC_LOOP;

CUBIC_DONE:
    // Snap to endpoint
    mov.f32 curx, px1;
    mov.f32 cury, py1;
    bra LOOP;
```

**Don't forget:** Add opcode dispatch in LOOP (around line 83):

```ptx
    // Drawing opcodes
    setp.eq.u32 p0, opcode32, 0x64; // MOVE
    @p0 bra OPC_MOVE;
    setp.eq.u32 p0, opcode32, 0x65; // LINE
    @p0 bra OPC_LINE;
    setp.eq.u32 p0, opcode32, 0x66; // QUAD
    @p0 bra OPC_QUAD;
    setp.eq.u32 p0, opcode32, 0x67; // CUBIC ← NEW!
    @p0 bra OPC_CUBIC;
    setp.eq.u32 p0, opcode32, 0x69; // CLOSE
    @p0 bra OPC_CLOSE;
```

**Test:**
```python
@pytest.mark.cuda
def test_gpu_cubic_bezier():
    """Verify GPU tessellates cubic Bézier."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # S-curve: P0=(0,0), C1=(0.3,0.5), C2=(0.7,0.5), P1=(1,0)
    program = "0 0 MOVE 0.3 0.5 0.7 0.5 1.0 0.0 CUBIC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)

    # Should produce smooth S-curve (many pixels)
    non_zero = np.count_nonzero(result.rgba[..., 0] > 0.05)
    assert non_zero > 150, f"S-curve should have ≥150 pixels, got {non_zero}"
```

---

### Step 3: Wire Ternary Hint (1 hour)

**Goal:** Adaptive tessellation quality

**File:** `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`

**Strategy:** Compute modulated segment count at start of each tessellation loop.

**Add helper macro at top of kernel (after register declarations):**

```ptx
// Helper: modulate segment count by ternary hint
// Input: base_segs (u32), hint (f32)
// Output: modulated_segs (u32), clamped to [4, 128]
.func (.reg .u32 modulated_segs) modulate_segments(
    .reg .u32 base_segs,
    .reg .f32 hint
) {
    .reg .f32 scale, segs_f;

    // scale = 1.0 + (hint * 0.5)
    // hint=-1 → scale=0.5 (50% fewer segments)
    // hint=0  → scale=1.0 (unchanged)
    // hint=+1 → scale=1.5 (50% more segments)
    mul.f32 scale, hint, 0.5;
    add.f32 scale, scale, 1.0;

    // segs_f = base_segs * scale
    cvt.rn.f32.u32 segs_f, base_segs;
    mul.f32 segs_f, segs_f, scale;
    cvt.rn.u32.f32 modulated_segs, segs_f;

    // Clamp to [4, 128]
    mov.u32 %r_min, 4;
    mov.u32 %r_max, 128;
    max.u32 modulated_segs, modulated_segs, %r_min;
    min.u32 modulated_segs, modulated_segs, %r_max;

    ret;
}
```

**Wait, PTX doesn't have functions!** Let me simplify - do it inline:

**In each tessellation loop (QUAD, CUBIC), replace:**

```ptx
// Before:
ld.param.u32 segs, [segments_per_curve];

// After:
.reg .f32 hint, scale, segs_f;
ld.param.f32 hint, [ternary_hint];
ld.param.u32 segs, [segments_per_curve];

// Modulate by hint
mul.f32 scale, hint, 0.5;        // -0.5 to +0.5
add.f32 scale, scale, 1.0;        // 0.5 to 1.5
cvt.rn.f32.u32 segs_f, segs;
mul.f32 segs_f, segs_f, scale;
cvt.rn.u32.f32 segs, segs_f;

// Clamp to [4, 128]
mov.u32 %r_min, 4;
mov.u32 %r_max, 128;
max.u32 segs, segs, %r_min;
min.u32 segs, segs, %r_max;

mov.u32 i, 0;
QUAD_LOOP:  // or CUBIC_LOOP
    // ... rest unchanged
```

**Update Python to pass hint:**

```python
def execute_rpn_gpu(self, rpn_program: str, width: int = 256, height: int = 256,
                   skip_raster: bool = False, ternary_hint: float = 0.0):
    """
    Args:
        ternary_hint: -1.0 (blur/fewer segments) to +1.0 (sharp/more segments)
    """
    loader.launch(
        self.pixel_genesis_kernel,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        params=[
            self._d_bytecode,
            ctypes.c_uint32(bytecode.nbytes),
            self._d_segments,
            self._d_count,
            ctypes.c_uint32(self.segments_per_curve),
            ctypes.c_float(ternary_hint),  # ← Wire it up!
        ],
    )
```

**Test:**
```python
@pytest.mark.cuda
def test_ternary_hint_modulation():
    """Verify ternary hint modulates tessellation quality."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "-0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE"

    # Blur mode (fewer segments)
    result_blur = bridge.execute_rpn_gpu(program, ternary_hint=-1.0, skip_raster=True)
    segs_blur = result_blur.segments.shape[0]

    # Normal mode
    result_norm = bridge.execute_rpn_gpu(program, ternary_hint=0.0, skip_raster=True)
    segs_norm = result_norm.segments.shape[0]

    # Sharp mode (more segments)
    result_sharp = bridge.execute_rpn_gpu(program, ternary_hint=+1.0, skip_raster=True)
    segs_sharp = result_sharp.segments.shape[0]

    # Should be ordered: blur < norm < sharp
    assert segs_blur < segs_norm, f"Blur ({segs_blur}) should have fewer segments than normal ({segs_norm})"
    assert segs_norm < segs_sharp, f"Normal ({segs_norm}) should have fewer segments than sharp ({segs_sharp})"
```

---

## Success Criteria

After these 3 steps, we'll have:

- ✅ **Buffer reuse** (already done)
- ✅ **Bounds checks** (already done)
- ✅ **AI-first mode** (<50µs latency via skip_raster)
- ✅ **CUBIC tessellation** (feature complete for SVG primitives)
- ✅ **Ternary hint** (adaptive quality)

**Latency:**
- AI mode: **~30-50µs** (GPU kernel only) ✓
- Human mode: **~600µs** (includes rasterization)

**Test Results Expected:**
```bash
pytest tests/test_procedural_drawing_performance.py -v

PASSED  test_gpu_rpn_operand_decoding     ✓
PASSED  test_gpu_quad_bezier              ✓
PASSED  test_gpu_cubic_bezier             ✓ (NEW)
PASSED  test_ternary_hint_modulation      ✓ (NEW)
PASSED  test_ai_mode_latency              ✓ (NEW, <50µs)
PASSED  test_parallel_batch_drawing       ✓
```

---

## Architectural Alignment

**Daniel's Vision:**
- AI doesn't need pixels, just procedural understanding ✓
- Ternary logic for adaptive computation ✓
- Font/OCR learning from atomic drawing knowledge ✓

**Current vs Future:**

**Phase 2B (Current):**
- Single-warp executor
- Ternary hint → segment modulation
- AI mode bypasses raster

**Phase 2C (Future):**
- 18-warp parallel executor (leverage RPN gem fully)
- Ternary logic per-warp
- Batch mode: 18 programs simultaneously

**Phase 3 (Font Learning):**
- TTF parser → procedural knowledge
- Train on glyph segments (not pixels)
- OCR from procedural understanding

---

## Files to Modify

1. `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`
   - Add `skip_raster` parameter
   - Add `ternary_hint` parameter
   - Update `RenderResult` dataclass

2. `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`
   - Add `OPC_CUBIC` handler
   - Add ternary hint modulation to QUAD/CUBIC
   - Add opcode dispatch for 0x67

3. `tests/test_procedural_drawing_performance.py`
   - Add `test_gpu_cubic_bezier`
   - Add `test_ternary_hint_modulation`
   - Add `test_ai_mode_latency`

---

## Reporting Back

When done:
1. Run tests: `pytest tests/test_procedural_drawing_performance.py -v`
2. Measure AI mode latency: `python measure_latency.py` (update script for skip_raster)
3. Create `TEMP/PHASE_2B_COMPLETION_REPORT.md`
4. Report segment counts for different ternary hints

---

**Have fun, Codex!** This should be ~3-4 hours of straightforward work. The CUBIC formula is just a slightly more complex version of QUAD, and the ternary hint is a simple scaling operation.

Let me know if you hit any issues! 🚀
