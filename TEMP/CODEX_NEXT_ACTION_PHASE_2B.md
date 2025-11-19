# Codex Next Action: Phase 2B - Latency Optimization + Feature Completion

**Phase:** 2B (Optimization + CUBIC/ARC)
**Status:** Phase 2A complete ✅ (see `PHASE_2A_COMPLETION_REPORT.md`)
**Goal:** Optimize to <100µs latency + implement CUBIC/ARC tessellation

---

## Current Status (Phase 2A Complete)

✅ **Working:**
- GPU operand decoding (MOVE, LINE, QUAD)
- Quadratic Bézier tessellation
- 4-byte aligned bytecode (no more misaligned errors)
- All core tests pass (3 passed, 1 xfailed for latency)

⚠️ **Needs Work:**
- Latency: ~700µs (target: <100µs)
- CUBIC/ARC not implemented
- Ternary hint unused
- Operand bounds not checked

---

## Priority Questions for You (Codex)

### Q1: Latency Optimization Strategy

Current breakdown:
- Bytecode compilation: ~20µs
- GPU malloc/memcpy: ~100µs
- Kernel execution: ~30-50µs ← **This is actually good!**
- Rasterization: ~500µs ← **This is the bottleneck**

**Options:**
1. **Focus on rasterizer** - Optimize `procedural_glyph_rasterizer` kernel (biggest impact)
2. **Buffer reuse** - Keep GPU buffers allocated between calls (save ~100µs)
3. **Batch mode** - Process N programs in 1 launch (amortize overhead)
4. **Skip for now** - GPU RPN kernel is already <50µs, maybe good enough?

**Which should we prioritize?** The GPU RPN kernel you built is actually fast (<50µs). The rasterizer is the bottleneck. Do you want to:
- A) Optimize the rasterizer (different codebase, bigger effort)
- B) Do buffer reuse (easy win, saves ~100µs)
- C) Defer optimization and focus on CUBIC/ARC first

### Q2: Operand Bounds Checking

Currently, if bytecode is truncated, kernel will read garbage:
```ptx
OPC_QUAD:
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 cx0, [addr + 0];   // What if idx+16 > prog_len? 💥
```

**Should we add guards?**
```ptx
OPC_QUAD:
    // Check if we have 16 bytes for operands
    .reg .u32 needed_idx;
    add.u32 needed_idx, idx, 16;
    setp.gt.u32 p0, needed_idx, prog_len;
    @p0 bra WRITE_COUNT;  // Abort if truncated

    // Safe to read now
    cvt.u64.u32 addr, idx;
    ...
```

**Trade-off:** Adds 3 instructions per opcode (~5-10ns overhead). Worth it?

### Q3: CUBIC Implementation Strategy

Quadratic Bézier (QUAD) formula:
```ptx
// xt = s²*p0.x + 2*s*t*c.x + t²*p1.x
mul.f32 xt, s, s;
mul.f32 xt, xt, curx;      // s² * p0.x
mul.f32 tmp, s, t;
mul.f32 tmp, tmp, 2.0;
mul.f32 tmp, tmp, cx0;     // 2*s*t * c.x
add.f32 xt, xt, tmp;
mul.f32 tmp, t, t;
mul.f32 tmp, tmp, px1;     // t² * p1.x
add.f32 xt, xt, tmp;
```

Cubic Bézier (CUBIC) formula:
```ptx
// xt = s³*p0.x + 3*s²*t*c1.x + 3*s*t²*c2.x + t³*p1.x
mul.f32 s2, s, s;       // s²
mul.f32 s3, s2, s;      // s³
mul.f32 t2, t, t;       // t²
mul.f32 t3, t2, t;      // t³

mul.f32 xt, s3, curx;   // s³ * p0.x

mul.f32 tmp, s2, t;
mul.f32 tmp, tmp, 3.0;
mul.f32 tmp, tmp, cx1;  // 3*s²*t * c1.x
add.f32 xt, xt, tmp;

mul.f32 tmp, s, t2;
mul.f32 tmp, tmp, 3.0;
mul.f32 tmp, tmp, cx2;  // 3*s*t² * c2.x
add.f32 xt, xt, tmp;

mul.f32 tmp, t3, px1;   // t³ * p1.x
add.f32 xt, xt, tmp;
```

**Options:**
- **A) Copy-paste-modify** - Duplicate QUAD_LOOP for CUBIC_LOOP (fast, works, ~50 lines PTX)
- **B) Parameterize** - Make tessellation loop generic (cleaner but PTX has no functions)
- **C) Python preprocessor** - Generate PTX from template (overkill?)

**What's your preference?**

### Q4: ARC Tessellation

Elliptical arcs are tricky. SVG spec has 7 parameters:
- `rx, ry` (radii)
- `x_axis_rotation` (angle)
- `large_arc_flag` (0 or 1)
- `sweep_flag` (0 or 1)
- `x, y` (endpoint)

**Approaches:**
1. **Approximate with Béziers** - Convert arc to 1-4 cubic Béziers (standard technique)
2. **Parametric tessellation** - Sample ellipse directly (`x = rx*cos(θ)`, `y = ry*sin(θ)`)
3. **Defer to CPU** - Fall back to host parser for ARC (sovereignty violation but pragmatic)

**Recommendation?**

---

## Phase 2B Task Options

Pick ONE focus area (or propose a different order):

### Option A: Latency Optimization (Easy Win)

**Goal:** Get latency <200µs (if not <100µs)

**Tasks:**
1. Implement buffer pool (reuse `d_bytecode`, `d_segments`, `d_count`)
2. Remove synchronize() after launch (only sync when reading results)
3. Benchmark each optimization step

**Estimated Time:** 1-2 hours
**Impact:** Save ~100µs
**Files:** `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`

**Deliverable:**
```python
class ProceduralDrawingBridge:
    def __init__(self, matryoshka_dim: int = 512):
        # ... existing init
        self._buffer_pool = {
            'd_bytecode': None,
            'd_segments': None,
            'd_count': None,
        }

    def execute_rpn_gpu(self, rpn_program: str, width: int, height: int):
        bytecode = self._compile_rpn_bytecode(rpn_program)

        # Reuse or allocate buffers
        d_bytecode = self._get_or_alloc('d_bytecode', bytecode.nbytes)
        d_segments = self._get_or_alloc('d_segments', self.MAX_SEGMENTS * 16)
        d_count = self._get_or_alloc('d_count', 4)

        # ... rest of execution
        # NO synchronize() here (defer until memcpy_dtoh)
```

### Option B: CUBIC + ARC Implementation (Feature Parity)

**Goal:** Support all SVG path primitives

**Tasks:**
1. Add `OPC_CUBIC` to PTX (copy-modify QUAD_LOOP)
2. Test with cubic Bézier curve
3. Add `OPC_ARC` using parametric tessellation
4. Test with elliptical arc

**Estimated Time:** 3-4 hours
**Impact:** Feature completeness
**Files:**
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`
- `tests/test_procedural_drawing_performance.py` (add test cases)

**Deliverable:**
```ptx
OPC_CUBIC:
    // Read 6 operands: cx1, cy1, cx2, cy2, x1, y1
    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;
    ld.global.f32 cx1, [addr + 0];
    ld.global.f32 cy1, [addr + 4];
    ld.global.f32 cx2, [addr + 8];
    ld.global.f32 cy2, [addr + 12];
    ld.global.f32 px1, [addr + 16];
    ld.global.f32 py1, [addr + 20];
    add.u32 idx, idx, 24;

    ld.param.u32 segs, [segments_per_curve];
    mov.u32 i, 0;
CUBIC_LOOP:
    // ... (tessellation formula as shown above)

    bra CUBIC_LOOP;
CUBIC_DONE:
    mov.f32 curx, px1;
    mov.f32 cury, py1;
    bra LOOP;
```

### Option C: Robustness (Production-Ready)

**Goal:** Make kernel bulletproof

**Tasks:**
1. Add operand bounds checking to all opcodes
2. Add alignment guards (debug mode)
3. Handle malformed bytecode (return error code instead of crash)
4. Add regression tests for edge cases

**Estimated Time:** 2-3 hours
**Impact:** Prevents crashes, easier debugging
**Files:**
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`
- `tests/test_procedural_drawing_performance.py`

**Deliverable:**
```ptx
OPC_MOVE:
    // Bounds check
    .reg .u32 needed_idx;
    add.u32 needed_idx, idx, 8;
    setp.gt.u32 p0, needed_idx, prog_len;
    @p0 bra WRITE_COUNT;  // Abort if truncated

    cvt.u64.u32 addr, idx;
    add.u64 addr, prog_ptr, addr;

    // Alignment check (debug mode)
    .reg .u64 addr_check;
    and.b64 addr_check, addr, 3;  // addr & 0x3
    setp.ne.u64 p_misalign, addr_check, 0;
    @p_misalign bra WRITE_COUNT;  // Abort if misaligned

    ld.global.f32 curx, [addr + 0];
    ld.global.f32 cury, [addr + 4];
    add.u32 idx, idx, 8;
    mov.f32 startx, curx;
    mov.f32 starty, cury;
    bra LOOP;
```

### Option D: Ternary Hint Modulation (Smart Quality)

**Goal:** Dynamic tessellation quality

**Tasks:**
1. Wire up `ternary_hint` parameter in PTX
2. Modulate `segments_per_curve` based on hint (-1.0 = fewer, +1.0 = more)
3. Test with blurred vs sharp curves
4. Document hint scale (e.g., -1.0 = 50% segments, +1.0 = 150%)

**Estimated Time:** 1 hour
**Impact:** Adaptive quality/performance trade-off
**Files:** `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`

**Deliverable:**
```ptx
OPC_QUAD:
    ld.param.f32 hint, [ternary_hint];
    ld.param.u32 base_segs, [segments_per_curve];

    // Compute modulated segment count
    .reg .f32 scale, segs_f;
    mul.f32 scale, hint, 0.5;        // -0.5 to +0.5
    add.f32 scale, scale, 1.0;        // 0.5 to 1.5
    cvt.rn.f32.u32 segs_f, base_segs;
    mul.f32 segs_f, segs_f, scale;
    cvt.rn.u32.f32 segs, segs_f;

    // Clamp to reasonable range
    mov.u32 min_segs, 4;
    mov.u32 max_segs, 128;
    max.u32 segs, segs, min_segs;
    min.u32 segs, segs, max_segs;

    mov.u32 i, 0;
QUAD_LOOP:
    // ... (rest unchanged)
```

---

## My Recommendations

**Short-term (Next 2-3 hours):**
1. **Buffer reuse** (Option A) - Easy win, saves ~100µs
2. **Operand bounds checks** (Option C) - Prevent crashes, 30 min effort

**Medium-term (Next 4-6 hours):**
3. **CUBIC implementation** (Option B) - Feature parity, copy-paste QUAD
4. **Ternary hint** (Option D) - Smart quality, 1 hour

**Long-term (Phase 2C or defer):**
5. **ARC tessellation** (Option B) - Complex, may need research
6. **Batch kernel mode** - Requires kernel architecture change
7. **Rasterizer optimization** - Different codebase, bigger scope

**Proposed Next Step:**
Start with **buffer reuse + bounds checks** (3 hours total). This gets us:
- Safer execution ✓
- ~100µs faster ✓
- Still under <200µs even if rasterizer not optimized

Then do **CUBIC + ternary hint** (5 hours). This completes core feature set.

**What do you think?** Should we follow this plan, or do you want to tackle something else first?

---

## Technical Notes

### Buffer Reuse Pattern

**Current (allocates every call):**
```python
def execute_rpn_gpu(self, rpn_program: str, width: int, height: int):
    bytecode = self._compile_rpn_bytecode(rpn_program)

    d_bytecode = loader.gpu_malloc(bytecode.nbytes)  # ← 50-100µs
    d_segments = loader.gpu_malloc(self.MAX_SEGMENTS * 16)
    d_count = loader.gpu_malloc(4)

    # ... use buffers

    loader.gpu_free(d_bytecode)  # ← 10-20µs
    loader.gpu_free(d_segments)
    loader.gpu_free(d_count)
```

**Optimized (reuses buffers):**
```python
class ProceduralDrawingBridge:
    def __init__(self, matryoshka_dim: int = 512):
        # ... existing
        self._bytecode_size = 4096  # Initial allocation
        self._d_bytecode = loader.gpu_malloc(self._bytecode_size)
        self._d_segments = loader.gpu_malloc(self.MAX_SEGMENTS * 16)
        self._d_count = loader.gpu_malloc(4)

    def execute_rpn_gpu(self, rpn_program: str, width: int, height: int):
        bytecode = self._compile_rpn_bytecode(rpn_program)

        # Reallocate only if too small
        if bytecode.nbytes > self._bytecode_size:
            loader.gpu_free(self._d_bytecode)
            self._bytecode_size = bytecode.nbytes * 2  # Grow with headroom
            self._d_bytecode = loader.gpu_malloc(self._bytecode_size)

        # Reuse existing buffers ✓
        loader.memcpy_htod(self._d_bytecode, bytecode.ctypes.data_as(ctypes.c_void_p), bytecode.nbytes)

        loader.launch(
            self.pixel_genesis_kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                self._d_bytecode,  # ← Reused!
                ctypes.c_uint32(bytecode.nbytes),
                self._d_segments,
                self._d_count,
                ctypes.c_uint32(self.segments_per_curve),
                ctypes.c_float(0.0),
            ],
        )

        # NO synchronize here (only when reading results)

        count_host = np.zeros(1, dtype=np.uint32)
        loader.memcpy_dtoh(count_host.ctypes.data_as(ctypes.c_void_p), self._d_count, 4)

        seg_count = min(int(count_host[0]), self.MAX_SEGMENTS)
        segments = np.zeros((seg_count, 4), dtype=np.float32)
        if seg_count:
            loader.memcpy_dtoh(segments.ctypes.data_as(ctypes.c_void_p), self._d_segments, segments.nbytes)

        # ... render segments

    def __del__(self):
        """Clean up GPU buffers on bridge destruction."""
        if hasattr(self, '_d_bytecode'):
            loader.gpu_free(self._d_bytecode)
            loader.gpu_free(self._d_segments)
            loader.gpu_free(self._d_count)
```

**Savings:** ~70-120µs per call (malloc + free overhead)

### CUBIC Formula Derivation

Cubic Bézier curve:
```
B(t) = (1-t)³ * P0 + 3*(1-t)²*t * C1 + 3*(1-t)*t² * C2 + t³ * P1
```

Expand `(1-t) = s`:
```
B(t) = s³*P0 + 3*s²*t*C1 + 3*s*t²*C2 + t³*P1
```

**PTX optimization:** Pre-compute `s²`, `s³`, `t²`, `t³` once, reuse for both `x` and `y` components.

### ARC Approximation Research

**Best practice:** Convert SVG arc to 1-4 cubic Béziers using standard algorithm:
1. Compute arc center from endpoint + radii
2. Split arc into <90° segments
3. Each segment = cubic Bézier with magic constant `k = 4/3 * tan(θ/4)`

**Reference:** [A Primer on Bézier Curves](https://pomax.github.io/bezierinfo/#circles_cubic) (section on circular arcs)

**Alternative:** Parametric tessellation (simpler but less accurate):
```ptx
OPC_ARC:
    // Simplified: assume circular arc centered at (0,0)
    ld.global.f32 rx, [addr + 0];
    ld.global.f32 start_angle, [addr + 4];
    ld.global.f32 sweep_angle, [addr + 8];

    ld.param.u32 segs, [segments_per_curve];
    mov.u32 i, 0;
ARC_LOOP:
    add.u32 i, i, 1;
    setp.gt.u32 p0, i, segs;
    @p0 bra ARC_DONE;

    // theta = start_angle + sweep_angle * (i / segs)
    cvt.rn.f32.u32 t, i;
    cvt.rn.f32.u32 tmp, segs;
    div.rn.f32 t, t, tmp;
    mul.f32 theta, sweep_angle, t;
    add.f32 theta, start_angle, theta;

    // x = rx * cos(theta), y = rx * sin(theta)
    cos.approx.f32 cos_theta, theta;  // PTX has approximate trig
    sin.approx.f32 sin_theta, theta;
    mul.f32 xt, rx, cos_theta;
    mul.f32 yt, rx, sin_theta;

    // Emit segment
    // ...

    bra ARC_LOOP;
```

**Trade-off:** Fast but only handles circular arcs (not elliptical). Good enough for Phase 2B?

---

## Success Criteria (Phase 2B)

**Minimum:**
- [ ] Buffer reuse implemented (save ~100µs)
- [ ] Operand bounds checks added (prevent crashes)
- [ ] CUBIC tessellation working (1 new test)

**Nice to Have:**
- [ ] Ternary hint modulation working
- [ ] Latency <200µs for simple programs
- [ ] ARC tessellation (circular arcs minimum)

**Stretch Goals:**
- [ ] Latency <100µs
- [ ] Batch kernel mode (N programs in 1 launch)
- [ ] Full elliptical ARC support

---

## Questions for Claude / Daniel

1. **Is rasterizer optimization in scope?** The `procedural_glyph_rasterizer` kernel takes 500µs for 1k pixels. Should we profile/optimize it, or accept that as "good enough" for Phase 2?

2. **Should we prioritize CUBIC over latency?** Feature completeness vs performance - which matters more right now?

3. **Buffer pool ownership:** Should the bridge own persistent GPU buffers, or should we have a global pool? Bridge-owned is simpler but wastes memory if multiple bridges exist.

4. **Batch mode architecture:** For true parallelism, should we:
   - A) Launch N threads, each processes 1 program (simple, works for small N)
   - B) Launch N blocks, each block processes 1 program (better scaling)
   - C) Dynamic work distribution (threads pull programs from queue)

---

## Your Turn, Codex!

**What would you like to tackle first?**
- Option A (buffer reuse + bounds checks) - 3 hours, easy win
- Option B (CUBIC + ARC) - 5 hours, feature complete
- Option C (robustness only) - 2 hours, production-ready
- Option D (ternary hint) - 1 hour, smart quality
- Something else?

**And please answer:**
1. Do you prefer copy-paste-modify for CUBIC, or should we try to parameterize the tessellation loop?
2. Should ARC be parametric (simple) or Bézier approximation (accurate)?
3. Any concerns about the buffer reuse pattern shown above?

**Let's keep the collaboration going!** 🚀

---

**Files for Reference:**
- `TEMP/PHASE_2A_COMPLETION_REPORT.md` - Full status
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx` - Current PTX
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Current bridge
- `tests/test_procedural_drawing_performance.py` - Current tests

**Reporting Back:**
When done, update:
1. `TEMP/PHASE_2B_STATUS.md` - New file documenting your progress
2. `TEMP/PROCEDURAL_DRAWING_PHASE_2B_COMPLETION_PROMPT.md` - When complete
3. Run benchmarks: `python measure_latency.py`
4. Run tests: `pytest tests/test_procedural_drawing_performance.py -v`

**Have fun! 🔧**
