# Multi-Core RPN Physics Coordination Plan

**Date:** November 24, 2025
**Context:** Phase 4A physics + Codex's ternary Reality Galaxy integration
**Goal:** Leverage 18 parallel RPN cores + ternary ops for physics simulation

---

## Current Architecture

### RPN Multi-Core System (ModularRPNEngine)
**From [modular_rpn_engine.py:51-52](knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py#L51-L52):**
```python
_INSTANCE_COUNT = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
_STACK_MAX = 69       # Tesla 6-9: 6+9=15→6, 6×9=54→9, contains literal 6&9
```

**Architecture:**
- **18 parallel RPN cores** (instance IDs 0-17)
- **69-line stack depth** per core
- Each core executes RPN programs independently
- Like modern multi-core CPUs: simultaneous execution

### Codex's Ternary Integration (COMPLETE ✅)
**Opcodes Added to Reality Galaxy:**
- `SIGN` — Returns {-1, 0, +1} for sign of value
- `TQUANT` — Ternary quantization with threshold deadband
- `TCMP` — Ternary comparison (sign of a-b)

**Validation:** 12/12 tests passing in [test_reality_galaxy.py](knowledge3d/cranium/tests/test_reality_galaxy.py)

---

## Problem: Current Physics Systems Use Single Core

**Issue:** All 9 physics systems default to `instance_id=0`:
```python
# Current pattern (all systems)
_engine: ModularRPNEngine | None = None

@property
def engine(self) -> ModularRPNEngine:
    if self._engine is None:
        self._engine = ModularRPNEngine()  # ← No instance_id specified!
    return self._engine
```

**Impact:**
- All systems compete for RPN core 0
- 17 cores (instances 1-17) sit idle
- No parallel execution benefit
- Serialized physics updates

---

## Solution: Multi-Core Physics Distribution

### Core Allocation Strategy

**Assign each physics system a unique RPN instance:**

| System | Instance ID | RPN Core | Rationale |
|--------|-------------|----------|-----------|
| ConstantAcceleration1D | 0 | Core 0 | Simple 1D motion (baseline) |
| HarmonicOscillator1D | 1 | Core 1 | 1D oscillation |
| Orbital2D | 2 | Core 2 | 2D orbital mechanics |
| Heat1D | 3 | Core 3 | 1D diffusion |
| Heat2D | 4 | Core 4 | 2D diffusion |
| Projectile2D | 5 | Core 5 | 2D projectile motion |
| DoublePendulum2D | 6 | Core 6 | Chaotic 2D system |
| CoupledOscillators | 7 | Core 7 | 2-mass spring system |
| RigidBody2D | 8 | Core 8 | 2D rotation |
| **Reserved** | 9-17 | Cores 9-17 | Phase 4B (E&M), Phase 4C (thermo), etc. |

**Benefits:**
- Each system gets dedicated RPN core
- Parallel physics updates
- No contention for stack resources
- Clear instance allocation pattern

---

## Implementation Pattern

### 1. Add Instance ID to Physics Systems

**Pattern:**
```python
@dataclass
class Projectile2D:
    # ... state variables ...
    dt: float
    _instance_id: int = 5  # ← Assigned RPN core
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        """Evaluate RPN expression on assigned core."""
        return self.engine.evaluate(expr, instance_id=self._instance_id)
```

**Changes:**
1. Add `_instance_id: int` field to each system
2. Pass `instance_id=self._instance_id` to `evaluate()` calls

---

### 2. Add Ternary Helper Methods

**Pattern (following Codex's ternary integration):**
```python
@dataclass
class Projectile2D:
    # ... existing fields ...

    def _sign_ternary(self, value: float) -> int:
        """Return ternary sign {-1, 0, +1} using SIGN opcode.

        Cheaper than float comparison for direction logic.
        """
        expr = f"{value} SIGN"
        result = self._eval(expr)
        return int(result)

    def _quantize_ternary(self, value: float, threshold: float) -> int:
        """Quantize value to {-1, 0, +1} with deadband threshold.

        Returns 0 if |value| < threshold (deadband).
        """
        expr = f"{value} {threshold} TQUANT"
        result = self._eval(expr)
        return int(result)

    def _compare_ternary(self, a: float, b: float) -> int:
        """Compare two values, return {-1, 0, +1}.

        Returns sign(a - b): -1 if a<b, 0 if a≈b, +1 if a>b.
        """
        expr = f"{a} {b} TCMP"
        result = self._eval(expr)
        return int(result)
```

---

### 3. Refactor Physics Logic with Ternary Ops

**Example: Projectile2D Drag Direction**

**Current (Binary Float):**
```python
def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
    for _ in range(n_steps):
        v_mag = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        drag_factor = self.k * v_mag

        # Binary float multiplication (includes sign)
        ax = -drag_factor * self.vx
        ay = -self.g - drag_factor * self.vy
        # ... integration ...
```

**Enhanced (Ternary + Binary Hybrid):**
```python
def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
    for _ in range(n_steps):
        # Compute magnitude (binary precision)
        v_mag = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        drag_factor = self.k * v_mag

        # Compute direction (ternary logic - cheaper!)
        sign_vx = self._sign_ternary(self.vx)  # {-1, 0, +1}
        sign_vy = self._sign_ternary(self.vy)

        # Combine: ternary * binary = binary result
        ax = -sign_vx * drag_factor
        ay = -self.g - sign_vy * drag_factor
        # ... integration ...
```

**Benefit:** Ternary sign extraction is **cheaper than float multiplication** for direction logic.

---

### 4. State Classification with Ternary

**Example: CoupledOscillators Normal Mode Detection**

**Current (No State Tracking):**
```python
def step(self, n_steps: int = 1):
    for _ in range(n_steps):
        # Compute forces...
        F1 = -self.k * self.x1 - self.k_c * (self.x1 - self.x2)
        # ... (no mode detection)
```

**Enhanced (Ternary Mode Classification):**
```python
def step(self, n_steps: int = 1):
    for _ in range(n_steps):
        # Detect normal mode using ternary
        sign_x1 = self._sign_ternary(self.x1)
        sign_x2 = self._sign_ternary(self.x2)

        # Mode product: {+1: in-phase, -1: out-of-phase, 0: transitional}
        mode_state = sign_x1 * sign_x2  # Ternary arithmetic
        self.normal_mode = mode_state  # Store for visualization/debugging

        # Compute forces...
        F1 = -self.k * self.x1 - self.k_c * (self.x1 - self.x2)
        # ...
```

**Benefit:** Mode classification is **inherently ternary**, natural representation.

---

### 5. Collision Detection with Ternary Comparisons

**Example: Projectile2D Ground Collision**

**Binary Approach:**
```python
if self.y <= 0.0:
    # Stop projectile (hit ground)
    self.vy = 0.0
```

**Ternary Approach:**
```python
# Compare y to ground level (0.0)
ground_state = self._compare_ternary(self.y, 0.0)
# Returns: -1 (below), 0 (at), +1 (above)

if ground_state <= 0:  # At or below ground
    # Stop projectile
    self.vy = 0.0
    # Could also use TQUANT to create collision deadband
```

**Benefit:** Ternary comparison + optional deadband for numerical stability.

---

## RPN Instance Allocation Table

### Phase 4A: Classical Mechanics (Cores 0-8)
| Core | System | Physics Domain | State Dim | Notes |
|------|--------|----------------|-----------|-------|
| 0 | ConstantAcceleration1D | 1D motion | (x, v) | Baseline |
| 1 | HarmonicOscillator1D | 1D oscillation | (x, v) | SHM |
| 2 | Orbital2D | 2D gravity | (x, y, vx, vy) | Kepler |
| 3 | Heat1D | 1D diffusion | T[N] | Parabolic PDE |
| 4 | Heat2D | 2D diffusion | T[H,W] | Laplacian |
| 5 | Projectile2D | 2D ballistics | (x, y, vx, vy) | Drag |
| 6 | DoublePendulum2D | 2D chaos | (θ1, θ2, ω1, ω2) | Lagrangian |
| 7 | CoupledOscillators | 2-mass spring | (x1, x2, v1, v2) | Normal modes |
| 8 | RigidBody2D | 2D rotation | (θ, ω) | Torque |

### Phase 4B: Electromagnetism (Cores 9-14)
| Core | System | Physics Domain | State Dim | Notes |
|------|--------|----------------|-----------|-------|
| 9 | PointCharge2D | Coulomb force | (x, y, vx, vy, q) | F ∝ 1/r² |
| 10 | ElectricField2D | E-field | E[H,W,2] | Vector field |
| 11 | MagneticField2D | Lorentz force | (x, y, vx, vy, B) | F = qv×B |
| 12 | LCCircuit | LC oscillator | (I, V) | ω = 1/√(LC) |
| 13 | RCCircuit | RC charging | (V, Q) | τ = RC |
| 14 | RLCCircuit | Damped LC | (I, V) | 3 regimes |

### Phase 4C+: Reserved (Cores 15-17)
| Core | Reserved For | Physics Domain |
|------|--------------|----------------|
| 15 | Phase 4C (Thermodynamics) | Ideal gas, Carnot cycle |
| 16 | Phase 4D (Wave Physics) | Wave equation, Doppler |
| 17 | Phase 4E (Modern Physics) | Quantum, relativity |

---

## Integration with Reality Galaxy

### Export to reality_* Nodes

**Pattern: Convert physics_demo systems to Reality Galaxy nodes**

```python
from knowledge3d.cranium.reality_galaxy import RealitySystem, RealityGalaxy

# Convert Projectile2D to RealitySystem
galaxy = RealityGalaxy()

projectile_system = RealitySystem(
    node_id="system:projectile_2d",
    state={
        "x": 0.0,
        "y": 0.0,
        "vx": 10.0,
        "vy": 20.0,
        "g": 9.81,
        "k": 0.1,  # Drag coefficient
        "dt": 0.001,
    },
    behavior_rpn="""
        # Compute velocity magnitude (binary)
        vx RECALL DUP * vy RECALL DUP * + SQRT v_mag STORE

        # Compute drag factor
        k RECALL v_mag RECALL * drag STORE

        # Compute drag direction (ternary)
        vx RECALL SIGN sign_vx STORE
        vy RECALL SIGN sign_vy STORE

        # Acceleration (ternary * binary hybrid)
        sign_vx RECALL NEG drag RECALL * ax STORE
        g RECALL NEG sign_vy RECALL NEG drag RECALL * - ay STORE

        # Integrate velocity
        vx RECALL ax RECALL dt RECALL * + vx STORE
        vy RECALL ay RECALL dt RECALL * + vy STORE

        # Integrate position
        x RECALL vx RECALL dt RECALL * + x STORE
        y RECALL vy RECALL dt RECALL * + y STORE
    """,
    law_rpn="""
        # Ground collision check (ternary)
        y RECALL 0 TCMP ground_state STORE
        ground_state RECALL 0 LE
    """,
)

galaxy.add_node(projectile_system)
```

**Benefits:**
- Pure RPN representation (no Python loop)
- Ternary ops integrated naturally
- Serializable to glTF (behavior_rpn + law_rpn as strings)
- Compatible with Codex's Reality Galaxy architecture

---

## Testing Strategy

### 1. Multi-Core Allocation Tests
**Verify each system uses unique RPN core:**
```python
def test_multicore_allocation():
    systems = [
        ConstantAcceleration1D(...),
        HarmonicOscillator1D(...),
        Projectile2D(...),
        # ... all 9 systems ...
    ]

    instance_ids = [sys._instance_id for sys in systems]
    assert len(set(instance_ids)) == 9  # All unique
    assert max(instance_ids) < 18  # Within valid range
```

### 2. Ternary Op Correctness
**Validate ternary helpers return expected values:**
```python
def test_ternary_helpers():
    system = Projectile2D(x=0, y=0, vx=-5.0, vy=10.0, g=9.81, k=0.1, dt=0.01)

    # Test SIGN
    assert system._sign_ternary(-5.0) == -1
    assert system._sign_ternary(0.0) == 0
    assert system._sign_ternary(10.0) == 1

    # Test TQUANT with deadband
    assert system._quantize_ternary(-0.05, 0.1) == 0  # Inside deadband
    assert system._quantize_ternary(-0.15, 0.1) == -1  # Outside deadband

    # Test TCMP
    assert system._compare_ternary(5.0, 3.0) == 1   # 5 > 3
    assert system._compare_ternary(3.0, 5.0) == -1  # 3 < 5
```

### 3. Parallel Execution Benchmark
**Measure speedup from multi-core execution:**
```python
def benchmark_parallel_physics():
    systems = [create_all_9_systems()]

    # Sequential (all on core 0)
    t_seq = time_sequential_execution(systems)

    # Parallel (distributed across cores 0-8)
    t_par = time_parallel_execution(systems)

    speedup = t_seq / t_par
    assert speedup > 1.5  # Expect at least 1.5× speedup
```

---

## Next Actions

### Immediate (Coordinate with Codex)
1. **Verify RPN opcode availability:**
   - Confirm SIGN, TQUANT, TCMP in ModularRPNEngine
   - Check Codex's reality_galaxy.py implementation

2. **Add instance_id to physics_demo.py:**
   - Assign cores 0-8 to Phase 4A systems
   - Update `_eval()` to pass instance_id

3. **Add ternary helpers:**
   - `_sign_ternary()`, `_quantize_ternary()`, `_compare_ternary()`
   - Test with existing Phase 4A systems

### Short-Term (Phase 4B Prep)
1. **Reserve cores 9-14 for E&M systems**
2. **Document allocation strategy** in CLAUDE.md
3. **Benchmark ternary vs binary** for sign operations

### Medium-Term (Reality Galaxy Export)
1. **Convert Phase 4A systems to RealitySystem nodes**
2. **Export behavior_rpn with ternary ops**
3. **Test glTF serialization** with Codex's export layer

---

## Benefits Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Core Utilization** | 1/18 (5.6%) | 9/18 (50%) | 9× better |
| **Parallel Execution** | Sequential | Parallel | 1.5-9× speedup |
| **Direction Logic** | Float multiply | Ternary SIGN | Cheaper ops |
| **State Classification** | No encoding | Ternary {-1,0,+1} | Natural repr |
| **Collision Detection** | Float compare | TCMP + deadband | Numerical stability |
| **Reality Galaxy Export** | N/A | Pure RPN | Serializable to glTF |

---

## Coordination with Codex

**Codex's Role (Complete ✅):**
- ✅ Ternary ops integrated into Reality Galaxy
- ✅ SIGN, TQUANT, TCMP opcodes working
- ✅ 12/12 tests passing

**My Role (Next):**
- Add instance_id allocation to physics_demo.py
- Implement ternary helper methods
- Refactor Phase 4A systems to use ternary where beneficial
- Export systems to Reality Galaxy nodes

**Shared Goal:**
- Unified physics simulation in Reality Galaxy
- Hybrid ternary/binary computation
- Leveraging 18-core RPN architecture
- Preparing for Phase 4B (Electromagnetism)

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**Date:** November 24, 2025
**Status:** Coordination plan for multi-core RPN + ternary physics
**Next:** Implement instance_id allocation + ternary helpers in physics_demo.py
