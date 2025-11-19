# Phase 2C-3: RPN Math Kernel Composition for Procedural Drawing

**Date:** 2025-11-18
**Assignee:** Codex
**Phase:** 2C-3 (RPN Interop)
**Prerequisites:** Phase 2C-2 Complete (Path State Management ✅)

---

## 🎯 Mission

**Leverage the 18-stack × 69-depth RPN math kernel to unlock trigonometric operations in procedural drawing.**

### Daniel's Vision
> "Procedural drawing is nothing else than math executed to vectors, we have a three level math RPN kernel with 18 inter-referable stacks and 69 memory lines each that can be instantiated... can't we compose with math under it? leverage our gem!"

**Goal:** Implement kernel composition architecture where RPN Math Kernel computes complex math (sin/cos/atan2) and Drawing Kernel consumes results via shared GPU buffers.

**Outcome:** ARC, ROTATE, and ELLIPSE opcodes operational without hardcoding trigonometry in PTX.

---

## 📋 Context: What's Already Working

### Phase 2C-1 ✅ (Transform Infrastructure)
- Transform matrix (2×3 affine)
- TRANSLATE, SCALE opcodes
- Transform application to LINE, QUAD, CUBIC, CLOSE

### Phase 2C-2 ✅ (Path State Management)
- Path state registers (stroke_width, stroke_r/g/b/a, local_ternary_hint)
- BEGIN_PATH, STROKE_WIDTH, SET_COLOR, TERNARY_MODULATE opcodes
- Local ternary hint override

### Test Status
- 12/14 tests passing ✅
- 1 skipped (ARC - **your task!**)
- 1 xfailed (latency with rasterization - expected)

---

## 🚫 The Problem: Pure PTX Limitations

**PTX has NO trigonometric intrinsics.**

```ptx
// ❌ DOESN'T EXIST IN PTX
atan2.approx.f32 angle, dy, dx;
cos.approx.f32 cos_val, angle;
sin.approx.f32 sin_val, angle;
```

**Current Workarounds:**
1. ❌ **NVRTC compilation** - Adds build complexity, requires CUDA C++ device math
2. ❌ **CPU fallback** - Violates sovereignty, breaks <100µs latency
3. ✅ **RPN Math Kernel** - Already exists, GPU-native, <10µs execution

**Solution:** Use RPN Math Kernel as a "math coprocessor" for Drawing Kernel.

---

## 🧠 The RPN Math Gem: What You Have Available

### Architecture Overview

**Three-Level RPN Stack:**
```
knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py
├─ Tier 1: Float32 arithmetic (18 stacks × 69 depth)
├─ Tier 2: Complex operations (sin, cos, atan2, sqrt, pow)
└─ Tier 3: Advanced (matrix ops, FFT, convolution)
```

**Per-Stack Capacity:**
- **18 independent stacks** (inter-referable via SWAP, DUP, OVER)
- **69 memory lines per stack** (STORE M0 ... M68, RECALL M0 ... M68)
- **~200 opcodes** including trigonometry, exponentials, logarithms

**Execution:** GPU-native PTX kernel (<10µs for typical expressions)

**Interface:**
```python
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

engine = ModularRPNEngine(num_stacks=18, stack_depth=69)

# Example: Compute rotation matrix
program = "0.5 PI * SIN STORE M0  0.5 PI * COS STORE M1"
result = engine.execute(program)

cos_val = result.memory[1]  # M1
sin_val = result.memory[0]  # M0
```

### Supported Trigonometric Operations

| RPN Token | Operation | Example |
|-----------|-----------|---------|
| `PI` | π constant (3.14159...) | `PI` → 3.14159 |
| `SIN` | Sine (radians) | `0.5 PI * SIN` → 1.0 |
| `COS` | Cosine (radians) | `0.5 PI * COS` → 0.0 |
| `TAN` | Tangent | `0.25 PI * TAN` → 1.0 |
| `ATAN2` | Arctangent (y, x) | `1 1 ATAN2` → 0.785 (π/4) |
| `SQRT` | Square root | `16 SQRT` → 4.0 |
| `POW` | Power | `2 8 POW` → 256.0 |

**Full list:** See `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

---

## 🏗️ Architecture: Kernel Composition

### High-Level Design

```
┌───────────────────────────────────────────────────────────┐
│ User Program (RPN Drawing Bytecode)                       │
│ "0.5 PI * RPN_SIN RPN_COS ROTATE_MATRIX 0 0 MOVE ..."    │
└───────────────────────────┬───────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ ProceduralDrawingBridge (Python)      │
        │ ├─ Parse RPN tokens                   │
        │ ├─ Detect RPN_* opcodes               │
        │ ├─ Execute RPN Math Kernel (Tier 1-3) │
        │ └─ Store results in shared GPU buffer │
        └───────────┬───────────────────────────┘
                    │ Shared GPU Buffer (float* math_results)
                    ↓
        ┌───────────────────────────────────────┐
        │ Drawing Kernel (PTX)                  │
        │ (pixel_genesis_universal_primitive)   │
        │                                       │
        │ OPC_ROTATE_MATRIX:                    │
        │   cos = math_results[0]  ← RPN output │
        │   sin = math_results[1]  ← RPN output │
        │   // Apply rotation to matrix         │
        │                                       │
        │ OPC_ARC:                              │
        │   point_count = math_results[0]       │
        │   for i in 0..point_count:            │
        │     x = math_results[i*2 + 1]         │
        │     y = math_results[i*2 + 2]         │
        │     emit_line_segment(x, y)           │
        └───────────────────────────────────────┘
```

### Execution Flow

**Step 1: User writes RPN drawing program**
```python
program = "0.5 PI * RPN_SIN RPN_COS ROTATE_MATRIX -1 0 MOVE 1 1 0 0 1 1 0 RPN_ARC STROKE"
```

**Step 2: Bridge parses and executes RPN math**
```python
# ProceduralDrawingBridge._preprocess_rpn_math()
if "RPN_SIN" in program or "RPN_COS" in program:
    rpn_engine = ModularRPNEngine()
    math_result = rpn_engine.execute("0.5 PI * SIN STORE M0  0.5 PI * COS STORE M1")
    math_buffer = cupy.array([math_result.memory[0], math_result.memory[1]], dtype=cupy.float32)
```

**Step 3: Bridge uploads math_buffer to GPU**
```python
# Pass as kernel parameter
self.pixel_genesis_kernel(
    ...,
    math_buffer.data.ptr,  # float* math_results
    np.uint32(len(math_buffer)),  # math_result_count
    ...
)
```

**Step 4: Drawing kernel consumes math_buffer**
```ptx
OPC_ROTATE_MATRIX:
    // Load RPN-computed sin/cos
    ld.global.f32 cos_val, [math_buffer + 0];
    ld.global.f32 sin_val, [math_buffer + 4];

    // Apply rotation to transform matrix
    // ... (matrix multiplication)
```

---

## 📐 Shared GPU Buffer Protocol

### Data Structure

**C-like pseudocode:**
```c
struct MathResultBuffer {
    uint32_t count;           // Number of valid results
    uint32_t checksum;        // Simple validation (sum of all values as uint32)
    float values[256];        // Precomputed values (max 256 floats)
};
```

**PTX Access:**
```ptx
// Kernel parameter declarations (add to existing params)
.param .u64 math_buffer_ptr;
.param .u32 math_buffer_count;

// Load parameter
.reg .u64 math_buffer;
ld.param.u64 math_buffer, [math_buffer_ptr];

// Read value at index i
.reg .u32 offset;
.reg .f32 value;
mul.lo.u32 offset, i, 4;  // i * sizeof(float)
add.u64 addr, math_buffer, offset;
ld.global.f32 value, [addr];
```

### Buffer Layout Examples

**Example 1: ROTATE_MATRIX**
```
Index | Value       | Description
------|-------------|-------------
  0   | cos(angle)  | Cosine for rotation
  1   | sin(angle)  | Sine for rotation
```

**RPN Program:**
```
"0.5 PI * SIN STORE M0  0.5 PI * COS STORE M1"
→ M0 = sin(π/2) = 1.0
→ M1 = cos(π/2) = 0.0
→ math_buffer = [0.0, 1.0]  (cos, sin)
```

**Example 2: ARC Tessellation**
```
Index | Value       | Description
------|-------------|-------------
  0   | 16          | Point count (number of segments)
  1   | x0          | First point X
  2   | y0          | First point Y
  3   | x1          | Second point X
  4   | y1          | Second point Y
  ...
 31   | x15         | Last point X
 32   | y15         | Last point Y
```

**RPN Program (pseudocode):**
```
"
  # Arc parameters
  0 0 SET_CENTER
  1 SET_RADIUS
  0 PI START_ANGLE SET
  PI END_ANGLE SET
  16 SEGMENTS SET

  # Loop to compute arc points
  0 SEGMENTS FOR_I
    I SEGMENTS / PI * START_ANGLE + ANGLE SET
    RADIUS ANGLE COS * X SET
    RADIUS ANGLE SIN * Y SET
    X STORE M{2*I+1}
    Y STORE M{2*I+2}
  NEXT
"
→ math_buffer = [16, x0, y0, x1, y1, ...]
```

---

## 🛠️ Implementation Checklist

### Part 1: Bridge Enhancements (Python)

**File:** `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`

#### Task 1.1: Add RPN Preprocessing
```python
def _preprocess_rpn_math(self, program: str) -> tuple[str, np.ndarray]:
    """
    Detect RPN_* tokens, execute RPN Math Kernel, return cleaned program + math_buffer.

    Args:
        program: RPN drawing program (may contain RPN_SIN, RPN_COS, RPN_ARC, etc.)

    Returns:
        (cleaned_program, math_buffer)
        - cleaned_program: Drawing opcodes with RPN_* tokens replaced by indices
        - math_buffer: GPU array of precomputed values
    """
    # Example implementation:
    # 1. Scan for "RPN_SIN RPN_COS" pair → extract angle expression
    # 2. Execute RPN engine: "{angle} SIN STORE M0  {angle} COS STORE M1"
    # 3. Build math_buffer = [M1, M0]  (cos, sin)
    # 4. Replace "RPN_SIN RPN_COS ROTATE_MATRIX" → "ROTATE_MATRIX" (consumes math_buffer[0:2])
    # 5. Return cleaned program + math_buffer
```

**Tokens to Handle:**
| Token | Meaning | RPN Expression | Buffer Layout |
|-------|---------|----------------|---------------|
| `RPN_SIN` | Push sin(angle) | `{angle} SIN` | [sin_val] |
| `RPN_COS` | Push cos(angle) | `{angle} COS` | [cos_val] |
| `RPN_SIN RPN_COS` | Push both | `{angle} SIN STORE M0 {angle} COS STORE M1` | [cos, sin] |
| `RPN_ARC` | Tessellate arc | See ARC pseudocode below | [count, x0, y0, ...] |

#### Task 1.2: Update execute_rpn_gpu Signature
```python
def execute_rpn_gpu(
    self,
    program: str,
    width: int = 256,
    height: int = 256,
    skip_raster: bool = False,
    ternary_hint: float = 0.0,
    math_buffer: Optional[np.ndarray] = None,  # ← NEW
) -> DrawingResult:
    """
    Execute RPN drawing program on GPU.

    Args:
        ...
        math_buffer: Optional precomputed math results (from RPN kernel)

    Returns:
        DrawingResult with segments and optional RGBA
    """
    # If math_buffer not provided, preprocess program
    if math_buffer is None:
        program, math_buffer = self._preprocess_rpn_math(program)

    # Convert math_buffer to CuPy
    if len(math_buffer) > 0:
        math_buffer_gpu = cp.array(math_buffer, dtype=cp.float32)
    else:
        math_buffer_gpu = cp.zeros(1, dtype=cp.float32)  # Dummy buffer

    # Pass to kernel
    self.pixel_genesis_kernel(
        ...,
        math_buffer_gpu.data.ptr,
        np.uint32(len(math_buffer)),
        ...
    )
```

---

### Part 2: PTX Kernel Enhancements

**File:** `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx`

#### Task 2.1: Add Kernel Parameters
```ptx
// Add to .entry pixel_genesis_universal_primitive parameters:
.param .u64 math_buffer_ptr,       // Pointer to math results
.param .u32 math_buffer_count      // Number of floats in buffer
```

#### Task 2.2: Load Parameters
```ptx
// After existing parameter loads
.reg .u64 math_buffer;
.reg .u32 math_count;
ld.param.u64 math_buffer, [math_buffer_ptr];
ld.param.u32 math_count, [math_buffer_count];
```

#### Task 2.3: Implement ROTATE_MATRIX (0x79)

**Opcode:** 0x79
**Operands:** None (consumes math_buffer[0:2])
**Behavior:** Apply rotation to transform matrix using RPN-computed cos/sin.

**Rotation Matrix Formula:**
```
[cos  -sin  0]   [a  c  e]   [a*cos+c*sin  -a*sin+c*cos  e]
[sin   cos  0] × [b  d  f] = [b*cos+d*sin  -b*sin+d*cos  f]
[0     0    1]   [0  0  1]   [0            0             1]
```

**PTX Implementation:**
```ptx
OPC_ROTATE_MATRIX:
    // Validate math_buffer has at least 2 values
    .reg .u32 need_rotate;
    mov.u32 need_rotate, 2;
    setp.lt.u32 p0, math_count, need_rotate;
    @p0 bra WRITE_COUNT;  // Skip if insufficient data

    // Load cos/sin from math_buffer
    .reg .f32 cos_val, sin_val;
    ld.global.f32 cos_val, [math_buffer + 0];   // math_buffer[0] = cos
    ld.global.f32 sin_val, [math_buffer + 4];   // math_buffer[1] = sin

    // Apply rotation to current transform matrix
    // New matrix = [cos -sin] × [a c e]
    //              [sin  cos]   [b d f]
    .reg .f32 new_a, new_b, new_c, new_d;
    .reg .f32 tmp;

    // new_a = a*cos + c*sin
    mul.f32 new_a, mat_a, cos_val;
    mul.f32 tmp, mat_c, sin_val;
    add.f32 new_a, new_a, tmp;

    // new_b = b*cos + d*sin
    mul.f32 new_b, mat_b, cos_val;
    mul.f32 tmp, mat_d, sin_val;
    add.f32 new_b, new_b, tmp;

    // new_c = -a*sin + c*cos
    .reg .f32 neg_sin;
    neg.f32 neg_sin, sin_val;
    mul.f32 new_c, mat_a, neg_sin;
    mul.f32 tmp, mat_c, cos_val;
    add.f32 new_c, new_c, tmp;

    // new_d = -b*sin + d*cos
    mul.f32 new_d, mat_b, neg_sin;
    mul.f32 tmp, mat_d, cos_val;
    add.f32 new_d, new_d, tmp;

    // Update matrix (e, f unchanged)
    mov.f32 mat_a, new_a;
    mov.f32 mat_b, new_b;
    mov.f32 mat_c, new_c;
    mov.f32 mat_d, new_d;

    bra LOOP;
```

**Test Program:**
```python
# Rotate by 90° (π/2 radians) using RPN
program = "0.5 PI * RPN_SIN RPN_COS ROTATE_MATRIX 1 0 MOVE 1 1 LINE STROKE"
# Expected: math_buffer = [cos(π/2), sin(π/2)] = [0, 1]
# Line (1, 0) → (1, 1) rotated by 90° → (0, 1) → (-1, 1)
```

#### Task 2.4: Implement PRECOMPUTED_PATH (0x7A)

**Opcode:** 0x7A
**Operands:** None (consumes math_buffer[0] = count, math_buffer[1:] = points)
**Behavior:** Emit line segments from RPN-precomputed path points.

**Buffer Layout:**
```
Index | Value       | Description
------|-------------|-------------
  0   | count       | Number of points (N)
  1   | x0          | Point 0 X
  2   | y0          | Point 0 Y
  3   | x1          | Point 1 X
  4   | y1          | Point 1 Y
  ...
2N-1  | x(N-1)      | Last point X
2N    | y(N-1)      | Last point Y
```

**PTX Implementation:**
```ptx
OPC_PRECOMPUTED_PATH:
    // Validate math_buffer has count field
    setp.lt.u32 p0, math_count, 1;
    @p0 bra WRITE_COUNT;

    // Load point count
    .reg .u32 point_count_raw;
    .reg .f32 point_count_float;
    ld.global.f32 point_count_float, [math_buffer + 0];
    cvt.rzi.u32.f32 point_count_raw, point_count_float;

    // Validate buffer size
    .reg .u32 required_size;
    mul.lo.u32 required_size, point_count_raw, 2;  // 2 floats per point
    add.u32 required_size, required_size, 1;       // +1 for count field
    setp.gt.u32 p0, required_size, math_count;
    @p0 bra WRITE_COUNT;  // Insufficient data

    // Loop through points and emit line segments
    .reg .u32 i;
    .reg .u64 addr;
    .reg .f32 x, y, prev_x, prev_y;
    mov.u32 i, 0;

    // Load first point as start
    ld.global.f32 prev_x, [math_buffer + 4];   // math_buffer[1]
    ld.global.f32 prev_y, [math_buffer + 8];   // math_buffer[2]
    mov.f32 curx, prev_x;
    mov.f32 cury, prev_y;
    mov.u32 i, 1;

PRECOMPUTED_LOOP:
    setp.ge.u32 p0, i, point_count_raw;
    @p0 bra PRECOMPUTED_DONE;

    // Load current point
    .reg .u32 offset;
    mul.lo.u32 offset, i, 8;        // i * 2 floats * 4 bytes
    add.u32 offset, offset, 4;      // Skip count field
    add.u64 addr, math_buffer, offset;
    ld.global.f32 x, [addr + 0];
    ld.global.f32 y, [addr + 4];

    // Emit line segment from prev to current (reuse LINE logic)
    // ... (copy transform + emit code from OPC_LINE)

    // Update prev
    mov.f32 prev_x, x;
    mov.f32 prev_y, y;
    mov.f32 curx, x;
    mov.f32 cury, y;

    add.u32 i, i, 1;
    bra PRECOMPUTED_LOOP;

PRECOMPUTED_DONE:
    bra LOOP;
```

**Test Program:**
```python
# RPN generates triangle points
math_buffer = np.array([
    3.0,          # 3 points
    0.0, 0.0,     # Point 0
    1.0, 0.0,     # Point 1
    0.5, 0.866,   # Point 2 (equilateral triangle)
], dtype=np.float32)

program = "PRECOMPUTED_PATH CLOSE STROKE"
result = bridge.execute_rpn_gpu(program, math_buffer=math_buffer, skip_raster=True)
# Expected: 3 line segments forming closed triangle
```

#### Task 2.5: Implement ARC (0x68) via RPN Tessellation

**Opcode:** 0x68 (already reserved)
**Operands:** rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, x, y (SVG arc parameters)
**Behavior:** Preprocess arc parameters via RPN, tessellate into line segments.

**High-Level Strategy:**
1. **Bridge preprocessing:** Detect ARC opcode + parameters
2. **RPN tessellation:** Compute arc points using trigonometry
3. **Buffer handoff:** Store points in math_buffer
4. **Drawing kernel:** Replace ARC with PRECOMPUTED_PATH

**Bridge Pseudocode:**
```python
def _tessellate_arc_via_rpn(
    start_x, start_y, rx, ry, rotation, large_arc, sweep, end_x, end_y
) -> np.ndarray:
    """
    Use RPN Math Kernel to tessellate SVG arc into line segments.

    Returns:
        math_buffer: [count, x0, y0, x1, y1, ...]
    """
    # 1. Convert SVG arc to center parameterization (standard math)
    #    (Use numpy/scipy or RPN for angle calculations)

    # 2. Build RPN program to compute arc points
    num_segments = self._compute_arc_segments(rx, ry, rotation)

    rpn_program = f"""
        # Arc parameters
        {start_x} {start_y} SET_ARC_START
        {end_x} {end_y} SET_ARC_END
        {rx} {ry} SET_ARC_RADII
        {rotation} SET_ARC_ROTATION
        {num_segments} SET_ARC_SEGMENTS

        # Compute start/end angles via ATAN2
        # ... (detailed RPN math)

        # Loop to generate points
        0 {num_segments} FOR_I
            # t = i / num_segments
            I {num_segments} / T SET
            # angle = start_angle + t * (end_angle - start_angle)
            # x = cx + rx * cos(angle)
            # y = cy + ry * sin(angle)
            # STORE M{2*i+1} (x)
            # STORE M{2*i+2} (y)
        NEXT
    """

    # 3. Execute RPN
    result = self.rpn_engine.execute(rpn_program)

    # 4. Build math_buffer
    points = [num_segments]
    for i in range(num_segments):
        points.append(result.memory[2*i + 1])  # x
        points.append(result.memory[2*i + 2])  # y

    return np.array(points, dtype=np.float32)
```

**PTX Arc Handler (Simple):**
```ptx
OPC_ARC:
    // Arc parameters already loaded by bridge into math_buffer
    // Jump to PRECOMPUTED_PATH to emit segments
    bra OPC_PRECOMPUTED_PATH;
```

**Alternative: Inline Arc Tessellation (Complex)**
If you prefer to keep arc logic in PTX:
1. Load arc parameters from bytecode
2. Compute center parameterization in PTX (using math_buffer for sin/cos)
3. Loop to emit line segments

**Recommendation:** Use bridge preprocessing for Phase 2C-3, inline PTX for Phase 2D optimization.

---

### Part 3: RPN Programs for Common Operations

#### Rotation by Angle
```rpn
# Input: angle (in radians)
# Output: math_buffer[0] = cos(angle), math_buffer[1] = sin(angle)

{angle} COS STORE M0
{angle} SIN STORE M1
```

**Usage:**
```python
program = "0.7854 RPN_SIN RPN_COS ROTATE_MATRIX 1 0 MOVE 1 1 LINE STROKE"
# Rotate by π/4 (45°)
```

#### Circle Tessellation
```rpn
# Input: center_x, center_y, radius, num_segments
# Output: math_buffer[0] = count, [1:] = points

{num_segments} STORE M0  # Point count

0 {num_segments} FOR_I
    # angle = 2π * i / num_segments
    I {num_segments} / 2 PI * * ANGLE SET

    # x = center_x + radius * cos(angle)
    {radius} ANGLE COS * {center_x} + X SET

    # y = center_y + radius * sin(angle)
    {radius} ANGLE SIN * {center_y} + Y SET

    X STORE M{2*I+1}
    Y STORE M{2*I+2}
NEXT
```

#### SVG Arc Center Parameterization
```rpn
# Complex - deferred to Python preprocessing
# Use scipy or manual implementation of SVG arc-to-center conversion
```

---

## 🧪 Testing Strategy

### Test 1: ROTATE_MATRIX Basic
```python
@pytest.mark.cuda
def test_rotate_matrix_90deg():
    """Verify ROTATE_MATRIX applies 90° rotation."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Rotate by 90° (π/2), draw horizontal line
    # Expected: Line rotates to vertical
    program = "0.5 PI * RPN_SIN RPN_COS ROTATE_MATRIX 0 0 MOVE 1 0 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True)

    # Original: (0,0) → (1,0)
    # Rotated 90°: (0,0) → (0,1)
    seg = result.segments[0]
    assert abs(seg[0] - 0.0) < 0.01  # x0
    assert abs(seg[1] - 0.0) < 0.01  # y0
    assert abs(seg[2] - 0.0) < 0.01  # x1 ≈ 0 after rotation
    assert abs(seg[3] - 1.0) < 0.01  # y1 ≈ 1
```

### Test 2: PRECOMPUTED_PATH Triangle
```python
@pytest.mark.cuda
def test_precomputed_path_triangle():
    """Verify PRECOMPUTED_PATH emits segments from RPN points."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Equilateral triangle
    math_buffer = np.array([
        3.0,          # 3 points
        0.0, 0.0,     # Point 0
        1.0, 0.0,     # Point 1
        0.5, 0.866,   # Point 2
    ], dtype=np.float32)

    program = "PRECOMPUTED_PATH CLOSE STROKE"
    result = bridge.execute_rpn_gpu(program, math_buffer=math_buffer, skip_raster=True)

    # Expected: 3 segments (2 from path, 1 from CLOSE)
    assert result.segments.shape[0] == 3
```

### Test 3: ARC via RPN Tessellation
```python
@pytest.mark.cuda
def test_arc_via_rpn():
    """Verify ARC tessellates via RPN and renders correctly."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Semicircle from (-1,0) to (1,0)
    program = "-1 0 MOVE 1 1 0 0 1 1 0 ARC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)
    rgba = result.rgba

    # Should have curved pixels
    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 100, f"Arc should have ≥100 pixels, got {non_zero}"
```

### Test 4: Full Composition Example
```python
@pytest.mark.cuda
def test_rpn_circle_with_rotation():
    """Verify RPN can generate circle, rotate it, and render."""
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Generate circle via RPN, rotate by 45°, render
    program = """
        RPN_CIRCLE 0 0 0.8 16     # center=(0,0), radius=0.8, 16 segments
        0.7854 RPN_SIN RPN_COS ROTATE_MATRIX  # Rotate by π/4
        PRECOMPUTED_PATH CLOSE STROKE
    """

    result = bridge.execute_rpn_gpu(program, width=128, height=128)
    rgba = result.rgba

    # Circle should be rendered
    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 200, f"Circle should have ≥200 pixels, got {non_zero}"
```

---

## 📊 Performance Budget

**Phase 2C-3 Latency Targets:**

| Operation | Budget | Notes |
|-----------|--------|-------|
| RPN Math Execution | <10µs | Typical expression (sin/cos/atan2) |
| Buffer Upload (CPU→GPU) | <5µs | 256 floats = 1KB |
| ROTATE_MATRIX | <2µs | Matrix multiplication (6 floats) |
| PRECOMPUTED_PATH | <20µs | 16 segments = 16× LINE emission |
| **Total AI Mode (with RPN)** | <60µs | Still under target ✅ |

**With Rasterization:** ~550µs (unchanged from Phase 2C-2)

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
1. ✅ ROTATE_MATRIX opcode working (consumes RPN sin/cos)
2. ✅ PRECOMPUTED_PATH opcode working (consumes RPN points)
3. ✅ test_rotate_matrix_90deg passing
4. ✅ test_precomputed_path_triangle passing

### Full Phase 2C-3 Complete
5. ✅ ARC opcode working via RPN tessellation
6. ✅ test_arc_via_rpn passing (skipped test now enabled)
7. ✅ RPN_CIRCLE token supported
8. ✅ Documentation: RPN interop design in TEMP/
9. ✅ All 14 tests passing (no skips)

---

## 📝 Deliverables

1. **Updated Python Bridge:**
   - `_preprocess_rpn_math()` method
   - `execute_rpn_gpu()` signature with math_buffer param
   - RPN token handlers (RPN_SIN, RPN_COS, RPN_CIRCLE, etc.)

2. **Updated PTX Kernel:**
   - math_buffer parameter declarations
   - OPC_ROTATE_MATRIX implementation (0x79)
   - OPC_PRECOMPUTED_PATH implementation (0x7A)
   - OPC_ARC implementation (0x68, via RPN)

3. **Test Coverage:**
   - `test_rotate_matrix_90deg`
   - `test_precomputed_path_triangle`
   - `test_arc_via_rpn` (enable existing skipped test)
   - `test_rpn_circle_with_rotation`

4. **Documentation:**
   - `TEMP/RPN_INTEROP_IMPLEMENTATION.md` - Technical details
   - `TEMP/PHASE_2C3_COMPLETION.md` - Completion report

---

## 🚀 Implementation Order

### Phase 1: Foundation (2-3 hours)
1. Add math_buffer parameter to PTX kernel
2. Implement `_preprocess_rpn_math()` stub in bridge
3. Test: Pass dummy math_buffer, verify kernel still works

### Phase 2: ROTATE_MATRIX (2-3 hours)
1. Implement RPN_SIN/RPN_COS token handlers in bridge
2. Implement OPC_ROTATE_MATRIX in PTX
3. Test: test_rotate_matrix_90deg

### Phase 3: PRECOMPUTED_PATH (3-4 hours)
1. Implement OPC_PRECOMPUTED_PATH in PTX
2. Add manual math_buffer test in bridge
3. Test: test_precomputed_path_triangle

### Phase 4: ARC via RPN (4-6 hours)
1. Implement arc tessellation in `_tessellate_arc_via_rpn()`
2. Wire up OPC_ARC to use PRECOMPUTED_PATH
3. Implement RPN_CIRCLE helper
4. Test: test_arc_via_rpn, test_rpn_circle_with_rotation

**Total Estimated Time:** 12-16 hours

---

## 🎨 Example Programs

### Example 1: Rotated Square
```python
program = """
    0.7854 RPN_SIN RPN_COS ROTATE_MATRIX  # Rotate by 45°
    -0.5 -0.5 MOVE
    0.5 -0.5 LINE
    0.5 0.5 LINE
    -0.5 0.5 LINE
    CLOSE STROKE
"""
# Diamond shape (rotated square)
```

### Example 2: Animated Rotation (Multi-Frame)
```python
for angle in np.linspace(0, 2*np.pi, 60):  # 60 frames
    program = f"""
        {angle} RPN_SIN RPN_COS ROTATE_MATRIX
        -0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE
    """
    result = bridge.execute_rpn_gpu(program, width=256, height=256)
    save_frame(result.rgba, f"rotation_{i:03d}.png")
```

### Example 3: Pac-Man (Arc + Fill)
```python
program = """
    # Pac-Man body (circle with missing wedge)
    0 0 MOVE
    0.8 0.8 0 0 1 0.8 0.3 ARC      # Upper arc
    0 0 LINE                         # Line to center
    0.8 0.8 0 0 0 0.8 -0.3 ARC     # Lower arc
    CLOSE FILL

    # Eye (small circle)
    0.2 0.5 TRANSLATE
    RPN_CIRCLE 0 0 0.1 8
    PRECOMPUTED_PATH CLOSE FILL
"""
```

### Example 4: Spirograph (Complex RPN Composition)
```python
program = """
    # Generate spirograph points via RPN
    RPN_SPIROGRAPH 0 0 0.8 0.3 5 100  # center, R, r, iterations, segments
    PRECOMPUTED_PATH STROKE
"""
# RPN_SPIROGRAPH computes: x = (R-r)*cos(t) + r*cos((R-r)/r * t)
#                          y = (R-r)*sin(t) - r*sin((R-r)/r * t)
```

---

## 🔗 References

### Existing Code
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` - RPN Math Kernel
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` - RPN opcode definitions
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Drawing bridge
- `knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx` - Drawing kernel

### Documentation
- `TEMP/PHASE_2C_PROGRESS_SUMMARY.md` - Phase 2C-1 completion
- `TEMP/PHASE_2C2_COMPLETION.md` - Phase 2C-2 completion (path state)
- `TEMP/RPN_DRAWING_INTEROP_DESIGN.md` - Initial design (Codex)

### External
- [SVG Path Arc Specification](https://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands)
- [PTX ISA Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [RPN on Wikipedia](https://en.wikipedia.org/wiki/Reverse_Polish_notation)

---

## 💡 Pro Tips

1. **Start with ROTATE_MATRIX** - Simplest RPN interop (2 floats)
2. **Test math_buffer manually first** - Bypass RPN preprocessing for initial PTX debugging
3. **Use skip_raster=True** - Faster iteration during geometry debugging
4. **Validate buffer sizes** - PTX bounds checks prevent crashes
5. **Print RPN intermediate results** - Use `rpn_engine.execute(program, trace=True)`

---

## ✅ Checklist for Codex

Before starting:
- [ ] Read `TEMP/PHASE_2C2_COMPLETION.md` (Phase 2C-2 summary)
- [ ] Review `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` (RPN API)
- [ ] Understand shared GPU buffer protocol (this doc, section 📐)

During implementation:
- [ ] Add math_buffer parameter to PTX kernel
- [ ] Implement `_preprocess_rpn_math()` in bridge
- [ ] Implement OPC_ROTATE_MATRIX (0x79)
- [ ] Implement OPC_PRECOMPUTED_PATH (0x7A)
- [ ] Implement ARC tessellation via RPN
- [ ] Write test_rotate_matrix_90deg
- [ ] Write test_precomputed_path_triangle
- [ ] Enable test_arc_via_rpn (remove skip marker)

After implementation:
- [ ] All tests passing (14/14, no skips) ✅
- [ ] AI mode latency <60µs ✅
- [ ] Create `TEMP/PHASE_2C3_COMPLETION.md`
- [ ] Update `TEMP/PHASE_2C_PROGRESS_SUMMARY.md` (mark Phase 2C-3 complete)

---

## 🎉 Closing Thoughts

**This is the breakthrough moment.**

By composing the RPN Math Kernel with the Drawing Kernel, you're unlocking:
- ✅ **Full SVG path primitives** (ARC, ELLIPSE, etc.)
- ✅ **Trigonometric transforms** (ROTATE)
- ✅ **Procedural generation** (Spirographs, fractals, parametric curves)
- ✅ **Sovereign architecture** (zero external dependencies)
- ✅ **<100µs latency** (GPU-native end-to-end)

**You're not just implementing opcodes — you're building a compositional GPU compute architecture.**

The 18-stack × 69-depth RPN gem is now your math coprocessor. Use it wisely.

---

**Phase 2C-3: Let's leverage our gem! 🚀**

*Good luck, Codex. Daniel and I believe in you.*

— Claude
