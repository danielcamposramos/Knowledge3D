# Codex Next Round: Procedural Drawing - Full Opcode Coverage + Training Validation

**Date:** 2025-11-18
**From:** Claude (Swarm Partner)
**Previous Session:** Excellent foundation with `rpn_executor.cu` basic opcodes + bridge integration ✅

---

## What You Built (Solid Work!) ✅

Your previous session delivered production-ready foundations:
1. ✅ **GPU RPN executor** - `rpn_executor.cu` with MOVE/LINE/SET_COLOR/SET_LINE_WIDTH
2. ✅ **PTX compilation** - Kernel builds cleanly, loads in bridge
3. ✅ **Bridge integration** - `execute_rpn_bytecode_gpu()` method working
4. ✅ **Test coverage** - `test_rpn_executor_gpu.py` validates basic path
5. ✅ **Performance tuning** - 26ms latency guard (realistic for 3060)
6. ✅ **Dataset builder** - Font → RPN → NPZ bytecode pipeline complete

**No complaints - this is excellent incremental sovereignty!**

---

## What Claude Added (While You Were Away) 🎨

I filled complementary gaps to set you up for the next phase:

### 1. **ProceduralDrawingSpecialist** ✅
**File:** `knowledge3d/cranium/specialists/procedural_drawing_specialist.py`

**What it does:**
- Wraps your RPN executor + bridges for adaptive swarm training
- Implements cross-modal learning: text ("A") ≈ visual (RPN glyph execution)
- Training loop: For each (char, rpn_bytecode):
  1. Text embedding via RPNEmbeddingEngine
  2. Visual embedding via RPN execute → FractalEmitter
  3. Contrastive loss: Pull text/visual together
  4. Update specialist adapter in swarm

**Integration:** Ready to consume your RPN datasets (JSONL + NPZ)

---

### 2. **Ternary Classification Utilities** ✅
**File:** `knowledge3d/cranium/ternary_utils.py`

**What it does:**
- Setun-inspired ternary logic (-1, 0, +1) for efficient decisions
- Font weight: `classify_font_weight(550) → 0` (normal)
- Stroke complexity: `classify_stroke_complexity(15) → 0` (medium)
- Routing: `ternary_route(0.85) → +1` (accept with high confidence)

**Use cases:**
- Matryoshka dimension selection: -1 → 64-dim, 0 → 512-dim, +1 → 2048-dim
- Style routing: Apply ternary weight to stroke width
- Device-side gates in kernels (your next task!)

---

### 3. **Training Script Integration** ✅
**File:** `scripts/train_adaptive_swarm.py` (modified)

**New mode:** `--mode procedural_drawing`

**Usage:**
```bash
bash scripts/k3d_env.sh run python3 scripts/train_adaptive_swarm.py \
  --mode procedural_drawing \
  --rpn-dataset /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \
  --epochs 10 \
  --matryoshka-dim 512 \
  --batch-size 32
```

**What happens:**
- Loads your JSONL dataset
- Compiles RPN to bytecode via your bridge
- Trains specialist on text-visual alignment
- Saves checkpoint with metrics

---

### 4. **Implementation Documentation** ✅
**File:** `docs/research/Procedural_Drawing_Implementation.md`

**Contents:**
- 4-stage architecture (you completed Stage 1, we're in Stage 2)
- File manifest, workflow, known issues
- Success metrics for each stage
- Connection to atomic cognition vision

---

## Your Next Mission: Full RPN Opcode Coverage 🎯

**Goal:** Complete Stage 2 (Device-Side RPN Evaluation) by extending `rpn_executor.cu` with missing opcodes.

### **Priority 1: Curve Opcodes** (This Session)

Extend `knowledge3d/cranium/kernels/rpn_executor.cu` with:

#### **QUAD Opcode (Quadratic Bézier)**
```cuda
case OP_QUAD: {
    // Stack: x0 y0 cx cy x1 y1
    // Pop 6 values: (x0,y0) start, (cx,cy) control, (x1,y1) end
    float y1 = stack[--stack_top];
    float x1 = stack[--stack_top];
    float cy = stack[--stack_top];
    float cx = stack[--stack_top];
    float y0 = current_y;  // Current pen position
    float x0 = current_x;

    // Approximate with segments (tunable via QUAD_SEGMENTS constant)
    const int QUAD_SEGMENTS = 16;  // Match ProceduralDrawingBridge quality
    for (int i = 1; i <= QUAD_SEGMENTS; i++) {
        float t = (float)i / QUAD_SEGMENTS;
        float mt = 1.0f - t;

        // Quadratic Bézier formula: (1-t)²·P0 + 2(1-t)t·Pc + t²·P1
        float x = mt*mt*x0 + 2*mt*t*cx + t*t*x1;
        float y = mt*mt*y0 + 2*mt*t*cy + t*t*y1;

        // Emit line segment from (current_x, current_y) to (x, y)
        if (segment_count < max_segments) {
            output_segments[segment_count * 9 + 0] = current_x;
            output_segments[segment_count * 9 + 1] = current_y;
            output_segments[segment_count * 9 + 2] = x;
            output_segments[segment_count * 9 + 3] = y;
            output_segments[segment_count * 9 + 4] = current_color[0];  // R
            output_segments[segment_count * 9 + 5] = current_color[1];  // G
            output_segments[segment_count * 9 + 6] = current_color[2];  // B
            output_segments[segment_count * 9 + 7] = current_color[3];  // A
            output_segments[segment_count * 9 + 8] = current_line_width;
            segment_count++;
        }

        current_x = x;
        current_y = y;
    }
    break;
}
```

#### **CUBIC Opcode (Cubic Bézier)**
```cuda
case OP_CUBIC: {
    // Stack: x0 y0 c1x c1y c2x c2y x1 y1
    // Pop 8 values: start, control1, control2, end
    float y1 = stack[--stack_top];
    float x1 = stack[--stack_top];
    float c2y = stack[--stack_top];
    float c2x = stack[--stack_top];
    float c1y = stack[--stack_top];
    float c1x = stack[--stack_top];
    float y0 = current_y;
    float x0 = current_x;

    const int CUBIC_SEGMENTS = 16;
    for (int i = 1; i <= CUBIC_SEGMENTS; i++) {
        float t = (float)i / CUBIC_SEGMENTS;
        float mt = 1.0f - t;

        // Cubic Bézier: (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
        float x = mt*mt*mt*x0 + 3*mt*mt*t*c1x + 3*mt*t*t*c2x + t*t*t*x1;
        float y = mt*mt*mt*y0 + 3*mt*mt*t*c1y + 3*mt*t*t*c2y + t*t*t*y1;

        // Emit segment (same as QUAD)
        // ... [segment emission code] ...

        current_x = x;
        current_y = y;
    }
    break;
}
```

#### **ARC Opcode (Elliptical Arc)**
```cuda
case OP_ARC: {
    // Stack: cx cy rx ry start_angle sweep_angle
    // Pop 6 values: center (cx,cy), radii (rx,ry), angles
    float sweep = stack[--stack_top];
    float start = stack[--stack_top];
    float ry = stack[--stack_top];
    float rx = stack[--stack_top];
    float cy = stack[--stack_top];
    float cx = stack[--stack_top];

    const int ARC_SEGMENTS = 16;
    for (int i = 1; i <= ARC_SEGMENTS; i++) {
        float t = (float)i / ARC_SEGMENTS;
        float angle = start + sweep * t;

        float x = cx + rx * cosf(angle);
        float y = cy + ry * sinf(angle);

        // Emit segment
        // ... [segment emission code] ...

        current_x = x;
        current_y = y;
    }
    break;
}
```

---

### **Priority 2: Ternary Device Gates** (This Session)

Add ternary stroke width application in `rpn_executor.cu`:

```cuda
__device__ float apply_ternary_stroke_width(int8_t ternary_weight, float base_width) {
    // Ternary weight: -1 (light), 0 (normal), +1 (bold)
    if (ternary_weight == -1) {
        return base_width * 0.7f;  // Thin
    } else if (ternary_weight == 1) {
        return base_width * 1.5f;  // Thick
    } else {
        return base_width;  // Normal
    }
}

// Usage in SET_LINE_WIDTH opcode:
case OP_SET_LINE_WIDTH: {
    float width = stack[--stack_top];
    int8_t ternary = 0;  // TODO: Get from metadata buffer
    current_line_width = apply_ternary_stroke_width(ternary, width);
    break;
}
```

**Metadata Buffer Extension:**
```cuda
__global__ void execute_rpn_bytecode(
    uint8_t* bytecode,
    int32_t bytecode_len,
    float* stack_workspace,
    float2* output_points,
    float4* output_colors,
    float* output_widths,
    int32_t* output_count,
    int8_t* ternary_metadata  // NEW: -1/0/+1 for weight, slant, etc.
) {
    // Read ternary metadata at kernel start
    int8_t weight_ternary = ternary_metadata[0];
    int8_t slant_ternary = ternary_metadata[1];

    // Use in opcode execution...
}
```

---

### **Priority 3: Opcode Constants** (This Session)

Define opcodes in kernel header:

```cuda
// knowledge3d/cranium/kernels/rpn_executor.cu (top of file)
#define OP_MOVE           0x01
#define OP_LINE           0x02
#define OP_QUAD           0x03  // NEW
#define OP_CUBIC          0x04  // NEW
#define OP_ARC            0x05  // NEW
#define OP_CLOSE          0x06  // NEW
#define OP_STROKE         0x07  // NEW
#define OP_FILL           0x08  // NEW (future)
#define OP_SET_COLOR      0x10
#define OP_SET_LINE_WIDTH 0x11
```

**Update bytecode compiler** in `procedural_drawing_bridge.py`:

```python
OPCODE_MAP = {
    'MOVE': 0x01,
    'LINE': 0x02,
    'QUAD': 0x03,  # NEW
    'CUBIC': 0x04,  # NEW
    'ARC': 0x05,  # NEW
    'CLOSE': 0x06,  # NEW
    'STROKE': 0x07,  # NEW
    'SET_COLOR': 0x10,
    'STROKE_WIDTH': 0x11,
    'SET_LINE_WIDTH': 0x11,  # Alias
}
```

---

### **Priority 4: Test Coverage** (This Session)

Extend `tests/test_rpn_executor_gpu.py`:

```python
@pytest.mark.cuda
def test_rpn_quad_curve():
    """Test quadratic Bézier curve execution."""
    bridge = ProceduralDrawingBridge()

    # RPN program: Move to (0,0), quad to (1,1) via control (0.5,1.0)
    rpn = "0.0 0.0 MOVE 0.5 1.0 1.0 1.0 QUAD STROKE"
    bytecode = bridge.compile_rpn_to_bytecode(rpn)

    result = bridge.execute_rpn_bytecode_gpu(bytecode)

    # Should generate ~16 segments (QUAD_SEGMENTS)
    assert result.segments is not None
    assert len(result.segments) > 10  # At least 10 approximation segments
    assert len(result.segments) < 20  # Not excessive

    # Check start/end points match
    assert np.allclose(result.segments[0, :2], [0.0, 0.0], atol=0.01)
    assert np.allclose(result.segments[-1, 2:4], [1.0, 1.0], atol=0.01)


@pytest.mark.cuda
def test_rpn_arc():
    """Test elliptical arc execution."""
    # Similar structure for ARC opcode...


@pytest.mark.cuda
def test_rpn_ternary_stroke_width():
    """Test ternary stroke width application."""
    # Test that ternary metadata affects output widths...
```

---

### **Priority 5: Performance Validation** (This Session)

Update `tests/test_procedural_drawing_performance.py`:

```python
@pytest.mark.cuda
def test_quad_curve_latency():
    """Validate QUAD opcode latency <100µs."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    rpn = "0.0 0.0 MOVE 0.5 1.0 1.0 1.0 QUAD STROKE"
    bytecode = bridge.compile_rpn_to_bytecode(rpn)

    import time
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = bridge.execute_rpn_bytecode_gpu(bytecode)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to µs

    avg_latency = np.mean(times)
    print(f"QUAD latency: {avg_latency:.2f} µs")

    # Target: <100µs (eventually <10µs with optimization)
    assert avg_latency < 100.0, f"Latency {avg_latency:.2f}µs exceeds 100µs"
```

---

## Compile + Test Workflow

### Step 1: Edit Kernel
```bash
# Add QUAD/CUBIC/ARC opcodes to knowledge3d/cranium/kernels/rpn_executor.cu
# Add ternary device gates
```

### Step 2: Recompile PTX
```bash
nvcc -ptx -arch=sm_86 --ptxas-options=-v \
  knowledge3d/cranium/kernels/rpn_executor.cu \
  -o knowledge3d/cranium/ptx/rpn_executor.ptx
```

### Step 3: Update Bridge
```bash
# Edit knowledge3d/cranium/bridges/procedural_drawing_bridge.py
# Add QUAD/CUBIC/ARC to OPCODE_MAP
# Add ternary metadata parameter to execute_rpn_bytecode_gpu()
```

### Step 4: Run Tests
```bash
pytest tests/test_rpn_executor_gpu.py -xvs
pytest tests/test_procedural_drawing_performance.py -xvs
```

---

## What NOT to Do (Scope Boundaries)

❌ **Don't implement PTX TTF parser yet** - Deferred to Stage 4 (after training validation)
❌ **Don't implement FILL opcode yet** - Rasterization complexity; Stage 3 focus is STROKE only
❌ **Don't implement RPN decoder yet** - Visual → RPN reverse path is future work
❌ **Don't run full training yet** - Wait for Stage 2 completion validation

✅ **DO implement:** QUAD, CUBIC, ARC, CLOSE, STROKE, ternary gates
✅ **DO test:** All new opcodes with GPU tests + performance benchmarks
✅ **DO validate:** Latency targets (<100µs per opcode, eventual <10µs)

---

## Expected Deliverables

### Files Modified:
1. `knowledge3d/cranium/kernels/rpn_executor.cu` - Add QUAD/CUBIC/ARC/CLOSE/STROKE opcodes + ternary gates
2. `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Update OPCODE_MAP, add ternary metadata param
3. `tests/test_rpn_executor_gpu.py` - Add tests for each new opcode
4. `tests/test_procedural_drawing_performance.py` - Add latency benchmarks

### Files Created:
- `knowledge3d/cranium/ptx/rpn_executor.ptx` - Recompiled kernel (already exists, just update)

### Test Results:
```
tests/test_rpn_executor_gpu.py::test_rpn_quad_curve PASSED
tests/test_rpn_executor_gpu.py::test_rpn_cubic_curve PASSED
tests/test_rpn_executor_gpu.py::test_rpn_arc PASSED
tests/test_rpn_executor_gpu.py::test_rpn_ternary_stroke_width PASSED
tests/test_procedural_drawing_performance.py::test_quad_curve_latency PASSED
```

### Success Criteria:
- ✅ All new opcodes working (segments generated correctly)
- ✅ Ternary gates applied (stroke width varies with metadata)
- ✅ Tests green (20+ passed, 1 xfail expected)
- ✅ Latency <100µs per opcode (eventual target <10µs)

---

## Why This Matters (Motivation)

You're completing **Stage 2** of the Procedural Drawing pipeline, which unblocks:

1. **Training validation** (Stage 3) - Can't train without full opcode coverage
2. **Atomic cognition** - Model learns curves → letters → words → phrases
3. **Sovereign OCR** - Visual → text via learned embeddings (no Tesseract)
4. **W3C standards** - Procedural Compression spec proof-of-concept

Your `rpn_executor.cu` kernel is the **atomic foundation** for all visual procedural reasoning. Every character the model learns to recognize starts as RPN bytecode executed by your kernel.

**This is the path from drawings to language.** 🎨 → 🔤

---

## Questions for Claude (If Needed)

If you hit blockers or want to discuss architecture:
- **Opcode design:** "Should ARC use parametric or polar form?"
- **Ternary integration:** "Where should ternary metadata come from in training?"
- **Performance:** "Is 100µs realistic for first iteration?"

I can answer via Daniel relay or you can proceed independently based on this spec.

---

## Final Notes

Your incremental approach is **exactly right** for sovereignty. We don't ship stubs, we ship working foundations and build up.

Stage 1 (offline dataset) proved the pipeline. Stage 2 (device-side RPN) proves GPU execution. Stage 3 (training) proves atomic cognition. Stage 4 (PTX parser) is optimization polish.

**You're doing this the right way.** Keep the small steps, keep the tests green, keep the latency guards realistic.

Looking forward to seeing QUAD/CUBIC/ARC in action! 🚀

— Claude (Swarm Partner)
