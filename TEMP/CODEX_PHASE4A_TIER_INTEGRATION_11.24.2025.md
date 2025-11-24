# Phase 4A: Tier-Aware Physics Integration — IMPLEMENTATION GUIDE

**Date:** November 24, 2025
**For:** Codex (Implementation Lead)
**Context:** Phase 4A physics complete (9 systems, 14/14 tests), ternary ops integrated into Reality Galaxy
**Goal:** Bind physics systems to 3-tier math core hierarchy with ternary-enhanced behaviors

---

## EXECUTIVE SUMMARY

You've already built the foundation:
- ✅ Ternary ops (SIGN, TQUANT, TCMP) working in Reality Galaxy
- ✅ STORE/RECALL semantics complete
- ✅ 12/12 Reality Galaxy tests passing

Claude built Phase 4A physics:
- ✅ 9 physics systems in physics_demo.py
- ✅ 14/14 tests passing
- ⚠️ All systems default to instance_id=0 (single core, no tier awareness)

**Your Mission:** Implement tier-aware system allocation + ternary-enhanced behaviors to leverage the full 18-core math architecture (worker-worker → worker → master).

---

## PART 1: MATH CORE ARCHITECTURE (READ FIRST)

### Three-Tier Hierarchy

**From [MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md):**

| Tier | Engine | Purpose | Opcodes | Matryoshka | Instances |
|------|--------|---------|---------|------------|-----------|
| **Tier-1 Simple** | LightweightRPNEngine | Ultra-fast, low-cost | Basic arithmetic, elementary math, stack ops | 64/128D | 0-11 |
| **Tier-2 Mid** | ModularRPNEngine | Moderate complexity | Tier-1 + matvec, clustering, reductions | 128/512D | 12-15 |
| **Tier-3 High** | AdvancedRPNEngine | High complexity | All tiers + TRM, symbolic ops | 512/2048D | 16-17 |

**Orchestration:** TieredRPNEngine routes programs to appropriate tier based on opcode analysis.

### Worker-Worker → Worker → Master Pattern

```
12× Simple cores (0-11)   → Pre-aggregation, local ops, scalar physics
                              ↓
4× Mid cores (12-15)       → Consume simple outputs, matrix ops, field equations
                              ↓
2× High cores (16-17)      → Consume mid summaries, TRM coupling, chaotic systems
```

**Key Insight:** Physics systems should be distributed across tiers based on **computational complexity**, not just assigned sequential instance IDs.

---

## PART 2: PHASE 4A PHYSICS TIER ALLOCATION

### Recommended Tier Assignment

**Claude's 9 systems + your task to implement tier binding:**

| System | Complexity | Tier | Instance | Matryoshka | Rationale |
|--------|------------|------|----------|------------|-----------|
| ConstantAcceleration1D | Simple | Tier-1 | 0 | 64D | 1D kinematics, scalar ops only |
| HarmonicOscillator1D | Simple | Tier-1 | 1 | 64D | 1D oscillation, trig + scalar |
| Projectile2D | Simple | Tier-1 | 2 | 128D | 2D motion, drag needs SIGN |
| RigidBody2D | Simple | Tier-1 | 3 | 128D | 2D rotation, torque/inertia |
| Heat1D | Moderate | Tier-2 | 12 | 128D | 1D Laplacian stencil, array ops |
| CoupledOscillators | Moderate | Tier-2 | 13 | 512D | 2-mass coupling, normal modes |
| Orbital2D | Moderate | Tier-2 | 14 | 512D | 2D gravity, energy conservation |
| Heat2D | Moderate | Tier-2 | 15 | 512D | 2D Laplacian, 5-point stencil |
| DoublePendulum2D | Complex | Tier-3 | 16 | 2048D | Chaotic, Lagrangian equations |
| **Reserved** | - | Tier-3 | 17 | 2048D | Phase 4B E&M (RLC circuits?) |

**Capacity:**
- Using 4/12 simple cores (0-3)
- Using 4/4 mid cores (12-15)
- Using 1/2 high cores (16)
- Leaves 8 simple cores + 1 high core for Phase 4B

---

## PART 3: IMPLEMENTATION TASKS

### Task 3.1: Add Tier Metadata to RealitySystem

**File:** `knowledge3d/cranium/reality_galaxy.py`

**Add fields to RealitySystem dataclass:**

```python
@dataclass
class RealitySystem(RealityNode):
    """Reality system with tier/instance metadata."""
    # ... existing fields ...

    # NEW: Tier assignment
    rpn_tier: int = 1  # 1=simple, 2=mid, 3=high
    rpn_instance: int = 0  # Specific instance ID (0-17)
    matryoshka_dim: int = 128  # Embedding dimension (64/128/512/2048)
```

**Update RealityGalaxy to use tier/instance:**

```python
class RealityGalaxy:
    def step_system(self, node_id: str, n_steps: int = 1) -> dict:
        """Step system using assigned tier/instance."""
        node = self._nodes[node_id]
        if not isinstance(node, RealitySystem):
            raise ValueError(f"Node {node_id} is not a RealitySystem")

        # Use node's assigned instance for RPN execution
        instance_id = node.rpn_instance
        state = node.state.copy()

        for step in range(n_steps):
            # Execute behavior_rpn with assigned instance
            # ... existing logic, but pass instance_id to RPN engine
```

**Success Criteria:**
- RealitySystem accepts `rpn_tier`, `rpn_instance`, `matryoshka_dim` parameters
- RealityGalaxy.step_system() honors `rpn_instance` when executing RPN programs
- Tests confirm tier assignment affects execution (e.g., log which tier was used)

---

### Task 3.2: Add Ternary Helper Opcodes to RPN Interpreter

**File:** `knowledge3d/cranium/reality_galaxy.py` (or dedicated helper module)

**Status:** You already have SIGN, TQUANT, TCMP working! ✅

**Enhancement:** Add convenience helpers for physics patterns:

```python
def _physics_ternary_opcodes() -> dict[str, callable]:
    """Physics-specific ternary helpers."""
    return {
        # Direction determination (cheaper than float multiply for sign)
        "DRAG_DIR": lambda stack: stack[-1] * (-1 if stack[-1] > 0 else 1 if stack[-1] < 0 else 0),

        # Collision gating (ternary comparison + threshold)
        "GROUND_CHECK": lambda stack, threshold: 0 if abs(stack[-1]) < threshold else stack[-1],

        # Mode detection (in-phase: +1, out-of-phase: -1, mixed: 0)
        "MODE_DETECT": lambda stack: stack[-2] * stack[-1],  # sign(x1) * sign(x2)
    }
```

**Integration:**
- Add these to your RPN opcode dictionary
- Test with Projectile2D drag direction
- Test with CoupledOscillators mode detection

**Success Criteria:**
- Ternary helpers available in behavior_rpn
- Tests demonstrate cheaper direction logic vs. float multiply
- Mode detection works for coupled oscillators

---

### Task 3.3: Convert Phase 4A Systems to RealitySystem Nodes

**Goal:** Export Claude's physics_demo.py systems to Reality Galaxy nodes with proper tier assignment.

**Implementation Pattern:**

```python
from knowledge3d.cranium.reality_galaxy import RealitySystem, RealityGalaxy
from knowledge3d.cranium.physics_demo import Projectile2D as Projectile2DDemo

def export_projectile2d_to_reality(params: dict) -> RealitySystem:
    """Convert Projectile2D to RealitySystem with ternary-enhanced RPN."""
    return RealitySystem(
        node_id="system:projectile_2d",
        state={
            "x": params.get("x", 0.0),
            "y": params.get("y", 0.0),
            "vx": params.get("vx", 10.0),
            "vy": params.get("vy", 20.0),
            "g": params.get("g", 9.81),
            "k": params.get("k", 0.1),
            "dt": params.get("dt", 0.001),
        },
        behavior_rpn="""
            # Compute velocity magnitude (binary)
            vx RECALL DUP *
            vy RECALL DUP *
            + SQRT v_mag STORE

            # Compute drag factor
            k RECALL v_mag RECALL * drag STORE

            # Drag direction (ternary - cheaper!)
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
            # Ground collision check (ternary comparison)
            y RECALL 0 TCMP ground_state STORE
            ground_state RECALL 0 LE
        """,
        rpn_tier=1,           # Tier-1 simple
        rpn_instance=2,       # Instance 2
        matryoshka_dim=128,   # 128D for 2D motion
    )
```

**Tasks:**
1. Create export functions for all 9 Phase 4A systems
2. Add ternary ops where beneficial:
   - Projectile2D: SIGN for drag direction
   - CoupledOscillators: SIGN for mode detection
   - DoublePendulum2D: TCMP for delta_sign state
3. Assign tier/instance/dimension per table above

**Success Criteria:**
- All 9 systems exported as RealitySystem nodes
- Ternary ops used for direction/mode logic
- Tests confirm RealityGalaxy executes on correct tier/instance
- Behavior matches original physics_demo.py (validate against existing 14 tests)

---

### Task 3.4: Create Tier Assignment Tests

**File:** `knowledge3d/cranium/tests/test_reality_physics_tiers.py` (new file)

**Test Cases:**

```python
def test_tier1_simple_systems():
    """Verify Tier-1 systems use instances 0-3."""
    galaxy = RealityGalaxy()

    # Add simple systems
    systems = [
        export_constant_acceleration_1d(),  # instance 0
        export_harmonic_oscillator_1d(),    # instance 1
        export_projectile_2d(),             # instance 2
        export_rigid_body_2d(),             # instance 3
    ]

    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 1
        assert 0 <= sys.rpn_instance <= 3
        assert sys.matryoshka_dim in [64, 128]


def test_tier2_mid_systems():
    """Verify Tier-2 systems use instances 12-15."""
    galaxy = RealityGalaxy()

    systems = [
        export_heat_1d(),               # instance 12
        export_coupled_oscillators(),   # instance 13
        export_orbital_2d(),            # instance 14
        export_heat_2d(),               # instance 15
    ]

    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 2
        assert 12 <= sys.rpn_instance <= 15
        assert sys.matryoshka_dim in [128, 512]


def test_tier3_high_systems():
    """Verify Tier-3 systems use instances 16-17."""
    galaxy = RealityGalaxy()

    sys = export_double_pendulum_2d()  # instance 16
    galaxy.add_node(sys)
    assert sys.rpn_tier == 3
    assert sys.rpn_instance == 16
    assert sys.matryoshka_dim == 2048


def test_ternary_drag_direction():
    """Verify ternary SIGN cheaper than float multiply for drag."""
    galaxy = RealityGalaxy()
    projectile = export_projectile_2d()
    galaxy.add_node(projectile)

    # Step system
    state = galaxy.step_system("system:projectile_2d", n_steps=10)

    # Verify ternary sign was used (check state has sign_vx, sign_vy)
    assert "sign_vx" in state
    assert "sign_vy" in state
    assert state["sign_vx"] in [-1.0, 0.0, 1.0]  # Ternary values


def test_coupled_oscillators_mode_detection():
    """Verify ternary mode detection for normal modes."""
    galaxy = RealityGalaxy()
    coupled = export_coupled_oscillators()
    galaxy.add_node(coupled)

    # In-phase: x1 > 0, x2 > 0 → mode = +1
    coupled.state["x1"] = 1.0
    coupled.state["x2"] = 0.5
    state = galaxy.step_system("system:coupled_oscillators", n_steps=1)
    # (Add mode detection to behavior_rpn first!)
    # assert state.get("mode_product") == 1.0  # In-phase

    # Out-of-phase: x1 > 0, x2 < 0 → mode = -1
    coupled.state["x1"] = 1.0
    coupled.state["x2"] = -0.5
    state = galaxy.step_system("system:coupled_oscillators", n_steps=1)
    # assert state.get("mode_product") == -1.0  # Out-of-phase
```

**Success Criteria:**
- All tier assignment tests pass
- Ternary ops verified in behavior_rpn execution
- Tests confirm tier/instance routing works correctly

---

### Task 3.5: Benchmark Ternary vs. Binary Performance

**File:** `knowledge3d/cranium/tests/benchmarks/test_ternary_physics_perf.py` (new)

**Goal:** Measure speedup from ternary ops in hot paths.

```python
import time
import numpy as np

def benchmark_drag_direction_binary_vs_ternary():
    """Compare float multiply vs. ternary SIGN for drag direction."""
    # Setup
    velocities = np.random.randn(10000, 2).astype(np.float32)
    drag_factor = 0.1

    # Binary approach (float multiply)
    start = time.perf_counter()
    for _ in range(1000):
        ax_binary = -drag_factor * velocities[:, 0]
        ay_binary = -drag_factor * velocities[:, 1]
    t_binary = time.perf_counter() - start

    # Ternary approach (SIGN + multiply)
    start = time.perf_counter()
    for _ in range(1000):
        sign_vx = np.sign(velocities[:, 0])
        sign_vy = np.sign(velocities[:, 1])
        ax_ternary = -sign_vx * drag_factor
        ay_ternary = -sign_vy * drag_factor
    t_ternary = time.perf_counter() - start

    speedup = t_binary / t_ternary
    print(f"Binary: {t_binary:.4f}s, Ternary: {t_ternary:.4f}s, Speedup: {speedup:.2f}×")

    # Assert measurable speedup
    assert speedup > 1.0, "Ternary should be faster for sign extraction"
```

**Success Criteria:**
- Benchmark shows measurable speedup (>1.0×) for ternary sign ops
- Document results in TEMP/PHASE4A_TERNARY_BENCHMARK_RESULTS.md

---

## PART 4: PHASE 4B PREPARATION (PLANNING ONLY)

### Electromagnetism Systems (6 total)

**Tier allocation plan:**

| System | Complexity | Tier | Instance | Matryoshka | Notes |
|--------|------------|------|----------|------------|-------|
| PointCharge2D | Simple | Tier-1 | 4 | 128D | Coulomb force, 1/r² |
| ElectricField2D | Moderate | Tier-2 | Reserved | 512D | Vector field ops |
| MagneticField2D | Moderate | Tier-2 | Reserved | 512D | Lorentz force |
| LCCircuit | Simple | Tier-1 | 5 | 128D | LC oscillator |
| RCCircuit | Simple | Tier-1 | 6 | 64D | Exponential decay |
| RLCCircuit | Moderate | Tier-2 | Reserved | 512D | Damped oscillation |

**Ternary opportunities:**
- Charge signs for Coulomb force
- Field direction quantization
- Damping regime detection (underdamped/critical/overdamped) → ternary state

**Action:** Document this in TEMP/PHASE4B_EM_TIER_PLAN.md (you don't need to implement yet, just plan)

---

## PART 5: RISKS AND MITIGATIONS

### Risk 1: TieredRPNEngine Not Respecting Instance Assignment

**Issue:** TieredRPNEngine may override instance_id based on opcode analysis.

**Mitigation:**
- Test that `instance_id` parameter is honored in `TieredRPNEngine.evaluate()`
- If TieredRPNEngine overrides, add `force_instance=True` flag
- Document behavior in MATH_CORE_SPECIFICATION.md

### Risk 2: Ternary Ops Not Available in PTX Path

**Issue:** SIGN/TQUANT/TCMP work in Python interpreter but may not be in PTX kernels.

**Mitigation:**
- Verify opcodes exist in rpn_opcodes.py
- Test on GPU if available (mark tests with @pytest.mark.cuda)
- Document which opcodes are Python-only vs. PTX-ready

### Risk 3: Law_rpn Assertions Cause Tier Promotion

**Issue:** Complex law_rpn (e.g., energy conservation checks) might force Tier-3 execution.

**Mitigation:**
- Keep law_rpn lightweight (simple comparisons, ternary ops)
- Use ASSERT only for critical invariants
- Log tier promotion events for debugging

---

## PART 6: SUCCESS CRITERIA

### Phase 4A Tier Integration Complete When:

- [ ] **Task 3.1:** RealitySystem has `rpn_tier`, `rpn_instance`, `matryoshka_dim` fields
- [ ] **Task 3.2:** Ternary helpers (SIGN, TQUANT, TCMP) confirmed working in behavior_rpn
- [ ] **Task 3.3:** All 9 Phase 4A systems exported to RealitySystem nodes with proper tier assignment
- [ ] **Task 3.4:** Tier assignment tests passing (tier1/tier2/tier3, ternary ops)
- [ ] **Task 3.5:** Benchmark shows ternary speedup documented
- [ ] **Validation:** Original 14 physics_demo tests still pass
- [ ] **Documentation:** TEMP/PHASE4A_TIER_INTEGRATION_COMPLETE.md written

### Expected Outcomes:

1. **Performance:** Physics systems distributed across 18 cores (not just core 0)
2. **Efficiency:** Ternary ops reduce computational cost for direction/mode logic
3. **Scalability:** Clear tier allocation pattern for Phase 4B E&M systems
4. **Sovereignty:** All execution via RPN stack (no black-box frameworks)

---

## PART 7: IMPLEMENTATION ORDER

**Recommended sequence:**

1. **Day 1:** Task 3.1 (add metadata to RealitySystem) → Verify tests pass
2. **Day 1:** Task 3.2 (ternary helpers) → Test SIGN/TQUANT/TCMP in simple cases
3. **Day 2:** Task 3.3 (export systems) → Start with Projectile2D, then expand
4. **Day 2:** Task 3.4 (tier tests) → Validate tier assignment works
5. **Day 3:** Task 3.5 (benchmarks) → Measure ternary speedup
6. **Day 3:** Write completion report + Phase 4B plan

---

## PART 8: COORDINATION WITH CLAUDE

**Claude's Role:**
- ✅ Built Phase 4A physics systems (done)
- ✅ Documented tier architecture (done)
- Will review your implementation
- Will integrate Phase 4B E&M systems after your foundation is ready

**Your Role (Codex):**
- **Lead implementer** for tier integration
- Own the reality_galaxy.py tier routing logic
- Own the ternary helper opcodes
- Own the export functions for Phase 4A → RealitySystem
- Own the test suite for tier assignment

**Communication:**
- Commit your work incrementally
- Write clear commit messages (e.g., "feat(reality): add tier metadata to RealitySystem")
- Document any blockers or design decisions in TEMP/ markdown files

---

## APPENDIX A: CODE TEMPLATES

### Template: Export Function

```python
def export_<system>_to_reality(params: dict) -> RealitySystem:
    """Convert <System> to RealitySystem with tier assignment."""
    return RealitySystem(
        node_id="system:<name>",
        state={
            # ... state variables ...
        },
        behavior_rpn="""
            # ... RPN program with ternary ops ...
        """,
        law_rpn="""
            # ... invariant checks ...
        """,
        rpn_tier=<1|2|3>,
        rpn_instance=<0-17>,
        matryoshka_dim=<64|128|512|2048>,
    )
```

### Template: Ternary-Enhanced Drag

```rpn
# Binary magnitude
vx RECALL DUP * vy RECALL DUP * + SQRT v_mag STORE
k RECALL v_mag RECALL * drag STORE

# Ternary direction (cheaper!)
vx RECALL SIGN sign_vx STORE
vy RECALL SIGN sign_vy STORE

# Hybrid: ternary * binary
sign_vx RECALL NEG drag RECALL * ax STORE
```

### Template: Mode Detection

```rpn
# Detect normal mode via ternary sign product
x1 RECALL SIGN x1_sign STORE
x2 RECALL SIGN x2_sign STORE
x1_sign RECALL x2_sign RECALL * mode_product STORE
# Result: +1 (in-phase), -1 (out-of-phase), 0 (transitional)
```

---

## APPENDIX B: REFERENCE DOCUMENTS

**Read these first:**
1. [docs/vocabulary/MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md) — Worker-worker → worker → master pattern
2. [TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md](TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md) — Multi-core allocation strategy
3. [TEMP/TERNARY_PHYSICS_INTEGRATION_PLAN_11.24.2025.md](TEMP/TERNARY_PHYSICS_INTEGRATION_PLAN_11.24.2025.md) — Hybrid ternary/binary patterns

**Claude's work:**
1. [knowledge3d/cranium/physics_demo.py](knowledge3d/cranium/physics_demo.py) — 9 physics systems (your export source)
2. [knowledge3d/cranium/tests/test_physics_demo.py](knowledge3d/cranium/tests/test_physics_demo.py) — 14 tests (validation baseline)
3. [TEMP/CODEX_PHASE4A_COMPLETION_11.24.2025.md](TEMP/CODEX_PHASE4A_COMPLETION_11.24.2025.md) — Phase 4A completion report

**Your previous work:**
1. [knowledge3d/cranium/reality_galaxy.py](knowledge3d/cranium/reality_galaxy.py) — Your reality node implementation
2. [knowledge3d/cranium/tests/test_reality_galaxy.py](knowledge3d/cranium/tests/test_reality_galaxy.py) — Your 12 passing tests

---

## FINAL NOTES

**You're in charge, Codex.** This is your implementation to own. Claude has laid the physics foundation and documented the tier architecture. Now you bring it together:

1. **Bind systems to tiers** (metadata + routing)
2. **Enhance with ternary** (SIGN/TQUANT for hot paths)
3. **Test thoroughly** (tier assignment + performance)
4. **Document results** (completion report + benchmarks)

**Expected Timeline:** 2-3 days for full implementation + testing.

**Success Metric:** All 9 Phase 4A systems running on appropriate tiers with measurable ternary speedup, ready for Phase 4B E&M integration.

**Let's build this. You've got the foundation; now make it sing across all 18 cores. 🚀**

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**Implementation Lead:** Codex (OpenAI)
**Date:** November 24, 2025
**Status:** Ready for implementation — Codex take the wheel!
