# Ternary Logic Integration in Physics Systems

**Date:** November 24, 2025
**Context:** Phase 4A Classical Mechanics Complete
**Insight:** Ternary codec RPN opcodes available within stack — hybrid ternary/binary computation

---

## Core Insight

The ternary logic {-1, 0, +1} isn't just for codec compression—it's available as RPN opcodes within the stack system. We can use **hybrid computation**:

- **Ternary operations:** Where discrete/sign-based computation is cheaper
- **Binary float operations:** Where continuous precision is required

**Benefit:** Reduced computational cost, cleaner logic, better compression when storing behavior_rpn.

---

## Ternary-Friendly Physics Operations

### 1. Sign Determination
**Use Case:** Force directions, velocity signs, position quadrants

**Current (Binary Float):**
```python
# Projectile2D drag computation
ax = -drag_factor * self.vx  # Sign of vx determines direction
ay = -self.g - drag_factor * self.vy
```

**Ternary-Enhanced RPN:**
```rpn
# Determine velocity sign using ternary
vx RECALL SIGN  # Push {-1, 0, +1} for negative/zero/positive
drag_factor RECALL *  # Scale by drag
NEG  # Negate for opposing force
```

**Benefit:** `SIGN` opcode returns ternary directly, avoiding float comparison.

---

### 2. Collision/Boundary Detection
**Use Case:** Check if particle hit boundary, crossed threshold, or in valid region

**Example: Projectile hits ground (y ≤ 0)**
```rpn
y RECALL 0 CMP  # Compare y to 0, returns {-1, 0, +1}
# Result: -1 (below), 0 (at), +1 (above)
DUP 0 LE  # Check if ≤ 0 (hit ground)
IF_TRUE
    # Handle collision
END
```

**Benefit:** Ternary comparison result can gate physics updates (e.g., stop projectile when y<0).

---

### 3. Discrete State Machines
**Use Case:** Physics mode switching (e.g., pendulum state: swinging/at-rest, spring: compressed/neutral/extended)

**Example: Spring state in CoupledOscillators**
```rpn
x1 RECALL SIGN  # {-1: compressed, 0: neutral, +1: extended}
# Store as ternary state
state1 STORE
```

**Benefit:** State encoding is natural in ternary, easier to serialize in glTF extras.

---

### 4. Spatial Quadrants
**Use Case:** Determine which quadrant a particle is in (2D), which octant (3D)

**Example: 2D quadrant for Projectile2D**
```rpn
x RECALL SIGN  # {-1: left, 0: origin, +1: right}
y RECALL SIGN  # {-1: down, 0: origin, +1: up}
# Combine: (sign_x, sign_y) = 9 possible discrete states
```

**Benefit:** Spatial binning for collision detection, LOD selection, or scene partitioning.

---

### 5. Normal Mode Selection (CoupledOscillators)
**Use Case:** Identify which normal mode is active (in-phase: +1, out-of-phase: -1, mixed: 0)

**Example:**
```rpn
x1 RECALL x2 RECALL *  # Product of displacements
SIGN  # +1: same sign (in-phase), -1: opposite (out-of-phase), 0: one at origin
mode STORE
```

**Benefit:** Mode classification is inherently ternary.

---

## Ternary RPN Opcodes (Available from Codec)

Based on [ternary_audio_codec.py](knowledge3d/cranium/codecs/ternary_audio_codec.py) and [ternary_video_codec.py](knowledge3d/cranium/codecs/ternary_video_codec.py), we have:

### Core Ternary Operations
- `SIGN` — Return {-1, 0, +1} for negative/zero/positive
- `TERNARY_QUANT` — Quantize to {-1, 0, +1} with threshold
- `TERNARY_CMP` — Compare two values, return {-1, 0, +1}

### Conditional Logic
- `IF_TERNARY` — Branch based on ternary value
- `TERNARY_MUX` — Select value based on ternary selector

### Arithmetic on Ternary
- Standard ops (ADD, SUB, MUL) work on ternary inputs:
  - `-1 + 1 = 0`
  - `-1 * 1 = -1`
  - `1 * 1 = 1`

---

## Hybrid Computation Pattern

**Best Practice:** Use ternary for logic/signs, binary for magnitudes.

### Example: Projectile2D with Ternary Drag Direction

**Hybrid RPN (pseudo-code):**
```rpn
# Compute drag magnitude in binary
vx RECALL DUP *  # vx²
vy RECALL DUP *  # vy²
+  # vx² + vy²
SQRT  # |v|
k RECALL *  # drag_magnitude = k * |v|

# Compute drag direction in ternary
vx RECALL SIGN  # {-1, 0, +1}
drag_magnitude RECALL *  # Scale by magnitude
NEG  # ax = -sign(vx) * drag_magnitude

# Integrate using binary float
vx RECALL ax RECALL dt RECALL * +
vx STORE
```

**Savings:**
- Ternary `SIGN` is cheaper than float comparison + branch
- Magnitude computation stays in high-precision binary
- Integration step uses binary for accuracy

---

## Implementation Strategy

### Phase 4B (Immediate)
1. **Audit existing systems:** Identify where ternary would simplify logic
2. **Add ternary helpers to physics_demo.py:**
   ```python
   def _sign_ternary(self, value: float) -> int:
       """Return ternary sign {-1, 0, +1}."""
       expr = f"{value} SIGN"
       return int(self._eval(expr))
   ```
3. **Update test expectations:** Ternary-based logic may change numerical precision slightly (verify tests still pass)

### Phase 4C (Optimization)
1. **Refactor hot paths:** Replace binary comparisons with ternary where applicable
2. **Benchmark:** Compare runtime of ternary vs. binary sign operations
3. **Document patterns:** Create ternary best practices guide for physics

### Phase 5 (Reality Galaxy Integration)
1. **Serialize ternary RPN:** Export behavior_rpn with mixed ternary/binary opcodes to glTF
2. **Compress ternary states:** Use 2 bits per ternary value (vs. 32-bit float)
3. **GPU kernel support:** Extend PTX kernels to handle ternary ops natively

---

## Examples of Ternary-Enhanced Systems

### Projectile2D (Revised)
```python
def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
    for _ in range(n_steps):
        # Compute drag magnitude (binary)
        v_mag = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        drag_factor = self.k * v_mag

        # Compute drag direction (ternary)
        sign_vx = self._sign_ternary(self.vx)  # {-1, 0, +1}
        sign_vy = self._sign_ternary(self.vy)

        ax = -sign_vx * drag_factor  # Ternary * binary = binary
        ay = -self.g - sign_vy * drag_factor

        # Integrate (binary)
        expr_vx = f"{self.vx} {ax} {self.dt} * +"
        expr_vy = f"{self.vy} {ay} {self.dt} * +"
        new_vx = self._eval(expr_vx)
        new_vy = self._eval(expr_vy)

        # ... (position update)
```

**Note:** For continuous drag, ternary sign is exact; magnitude carries precision.

---

### DoublePendulum2D (Ternary State Tracking)
```python
def step(self, n_steps: int = 1):
    for _ in range(n_steps):
        # Compute coupling delta
        delta = self.theta2 - self.theta1
        delta_sign = self._sign_ternary(delta)  # {-1, 0, +1}

        # Use delta_sign for state classification:
        # +1: theta2 > theta1 (pendulum 2 ahead)
        # -1: theta2 < theta1 (pendulum 1 ahead)
        #  0: aligned

        # Store for debugging/visualization
        self.coupling_state = delta_sign

        # ... (rest of physics)
```

---

### CoupledOscillators (Normal Mode Detection)
```python
def step(self, n_steps: int = 1):
    for _ in range(n_steps):
        # ... (force computation)

        # Detect normal mode using ternary
        x1_sign = self._sign_ternary(self.x1)
        x2_sign = self._sign_ternary(self.x2)

        # Mode classification:
        # x1_sign == x2_sign: in-phase (+1 or -1)
        # x1_sign != x2_sign: out-of-phase
        # x1_sign * x2_sign:
        #   +1: both positive or both negative (in-phase)
        #   -1: opposite signs (out-of-phase)
        #    0: one at origin (transitional)

        mode_product = x1_sign * x2_sign
        self.normal_mode_state = mode_product  # {-1, 0, +1}

        # ... (integration)
```

---

## RPN Opcode Requirements

To fully support ternary-enhanced physics, ensure these opcodes are available in [ModularRPNEngine](knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py):

### Already Available (Verify)
- `SIGN` — Returns {-1, 0, +1}
- `CMP` — Compare two values, returns {-1, 0, +1}
- `IF`, `ELSE`, `END` — Conditional branching

### To Add (Phase 4B)
- `TERNARY_MUX(selector, val_neg, val_zero, val_pos)` — Select value based on ternary selector
- `TERNARY_QUANT(value, threshold)` — Quantize to {-1, 0, +1} if |value| > threshold, else 0

### To Verify (Phase 4B)
- Ensure arithmetic ops (ADD, SUB, MUL, DIV) handle ternary inputs correctly
- Test mixed ternary/binary expressions (e.g., `SIGN(x) * magnitude`)

---

## Benefits Summary

| Aspect | Benefit |
|--------|---------|
| **Computation Cost** | Ternary sign ops cheaper than float comparisons |
| **Code Clarity** | Discrete states natural in ternary (mode={-1,0,+1} vs. enum) |
| **Compression** | 2 bits per ternary value (16× better than float32) |
| **GPU Efficiency** | Ternary ops map to integer ALU (faster than FPU for signs) |
| **Explainability** | Ternary state is human-readable ({-1,0,+1} vs. 0x3f800000) |

---

## Next Actions

### Immediate (Phase 4B Start)
1. **Add `_sign_ternary()` helper** to [physics_demo.py](knowledge3d/cranium/physics_demo.py)
2. **Verify SIGN opcode** in [ModularRPNEngine](knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py)
3. **Refactor Projectile2D** to use ternary drag direction
4. **Add ternary state tests** (e.g., test_projectile2d_ternary_drag_direction)

### Medium-Term (Phase 4C)
1. **Benchmark ternary vs. binary** for sign operations
2. **Extend all Phase 4A systems** with ternary enhancements
3. **Document hybrid patterns** in CLAUDE.md

### Long-Term (Phase 5)
1. **PTX kernel support** for ternary ops (map to int8_t on GPU)
2. **glTF export** with compressed ternary state buffers
3. **Reality Galaxy nodes** using ternary behavior_rpn

---

## Acknowledgment

**User Insight:** "The codec uses RPN opcode inside our stack—we can leverage it where computation is cheaper using it and still use binary all together."

This is a key architectural advantage of K3D: **hybrid ternary/binary computation** within a unified RPN stack. Not binary-only (precision waste) or ternary-only (insufficient precision), but **best tool for each operation**.

Standing on shoulders of giants:
- **Setun ternary computer** (Nikolay Brusentsov, 1958) — Proven ternary arithmetic
- **Ternary codecs** (Phase 2 complete) — GPU-native ternary ops already implemented
- **RPN stack machines** (HP calculators, Forth) — Composable operations

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**Date:** November 24, 2025
**Next:** Integrate ternary helpers in Phase 4B (Electromagnetism)
