# Dynamic Math Core Spawning — Implementation Briefing

**Date:** November 24, 2025
**For:** Codex (Implementation Lead)
**From:** Claude (Architecture Partner) + Daniel (Project Lead)
**Priority:** HIGH (Foundational Enhancement)
**Goal:** Enable GPU-limited dynamic Math Core instantiation system-wide

---

## Mission Statement

Transform Math Cores from **static 18-instance allocation** to **dynamic GPU-limited spawning**, unlocking the ability to simulate 100s-1000s of physics systems concurrently on consumer/datacenter GPUs.

**Why This Matters:**
- Current: 13 systems use 12/18 cores (66.7%), rest idle
- Future: 10,000 systems use 10,000 cores dynamically spawned
- Enables: City-scale simulations, massive multi-physics, real-time AGI at scale

---

## Current State (Phase 4B Baseline)

### Static Allocation Pattern
```python
# knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py
_INSTANCE_COUNT = 18  # HARD-CODED LIMIT

class ModularRPNEngine:
    def __init__(self):
        self.instances = [RPNInstance(id=i) for i in range(18)]
        # Fixed 18 instances, no dynamic spawning
```

### Current Usage
```python
# knowledge3d/cranium/reality_physics_export.py
def export_projectile_2d():
    return RealitySystem(
        rpn_tier=1,
        rpn_instance=2,  # STATICALLY ASSIGNED
        # ...
    )
```

**Problem:**
- 18 cores fixed at compile time
- 6 cores idle (Phase 4B uses 12/18)
- Cannot scale to 100+ systems without code rewrite
- GPU has 460+ SMs sitting unused (RTX 3070)

---

## Target Architecture: Dynamic Spawning

### Math Core Pool Manager
```python
class MathCorePool:
    """Dynamic Math Core instantiation and lifecycle management.

    Responsibilities:
    - Query GPU capacity (SM count, VRAM, warp limits)
    - Spawn cores on-demand (lazy instantiation)
    - Pool idle cores for reuse
    - Deallocate after timeout
    - Monitor utilization, scale dynamically
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.active_cores: Dict[int, MathCore] = {}
        self.idle_pool: List[MathCore] = []
        self.max_cores = self._query_gpu_capacity()

    def spawn_core(self, tier: int, reuse: bool = True) -> int:
        """Spawn or reuse a Math Core.

        Args:
            tier: 1 (Simple), 2 (Mid), 3 (High)
            reuse: If True, reuse idle core from pool

        Returns:
            instance_id: Unique core identifier

        Raises:
            RuntimeError: If GPU at capacity
        """
        # Try reuse first
        if reuse and self.idle_pool:
            core = self.idle_pool.pop()
            core.reset(tier=tier)
            self.active_cores[core.instance_id] = core
            return core.instance_id

        # Check capacity
        if len(self.active_cores) >= self.max_cores:
            raise RuntimeError(
                f"GPU {self.gpu_id} at capacity "
                f"({len(self.active_cores)}/{self.max_cores} cores)"
            )

        # Spawn new core
        instance_id = self._allocate_id()
        core = MathCore(
            instance_id=instance_id,
            tier=tier,
            stack_depth=69,
            gpu_id=self.gpu_id,
        )
        self.active_cores[instance_id] = core
        return instance_id

    def release_core(self, instance_id: int, pool: bool = True):
        """Release a Math Core back to pool or deallocate.

        Args:
            instance_id: Core to release
            pool: If True, add to idle pool for reuse
        """
        if instance_id not in self.active_cores:
            return

        core = self.active_cores.pop(instance_id)

        if pool and len(self.idle_pool) < self.max_cores * 0.1:
            # Keep up to 10% of max cores in pool
            self.idle_pool.append(core)
        else:
            # Deallocate
            core.cleanup()

    def _query_gpu_capacity(self) -> int:
        """Query GPU for max concurrent cores.

        Strategy:
        - Query SM count via CUDA API
        - Conservative estimate: 10 cores per SM
        - Account for VRAM limits (2 KB per core)

        Returns:
            max_cores: Maximum concurrent Math Cores
        """
        import cupy as cp

        # Get SM count
        device_props = cp.cuda.runtime.getDeviceProperties(self.gpu_id)
        sm_count = device_props["multiProcessorCount"]

        # Conservative: 10 cores per SM
        hardware_limit = sm_count * 10

        # VRAM check (2 KB per core)
        mem_info = cp.cuda.runtime.memGetInfo()
        free_vram_mb = mem_info[0] / (1024 ** 2)
        vram_limit = int(free_vram_mb * 0.01 / 0.002)  # 1% of VRAM, 2KB/core

        return min(hardware_limit, vram_limit)

    def _allocate_id(self) -> int:
        """Allocate unique instance ID."""
        used_ids = set(self.active_cores.keys())
        for i in range(self.max_cores):
            if i not in used_ids:
                return i
        raise RuntimeError("No available instance IDs")
```

---

## Implementation Tasks

### Task 1: Create MathCorePool (New Module)
**File:** `knowledge3d/cranium/ptx_runtime/math_core_pool.py`

**Requirements:**
- Implement `MathCorePool` class (see architecture above)
- Query GPU capacity via CuPy
- Lazy instantiation (spawn on first request)
- Pooling for reuse (avoid allocation overhead)
- Timeout-based deallocation (prevent memory leaks)
- Thread-safe (multiple systems spawning concurrently)

**Success Criteria:**
- Can query RTX 3070 → returns ~460 max cores
- Can spawn 100 cores in <100ms
- Can release cores back to pool
- Memory overhead <1% of VRAM

---

### Task 2: Update ModularRPNEngine (Refactor)
**File:** `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

**Changes:**
```python
# OLD: Static allocation
_INSTANCE_COUNT = 18

class ModularRPNEngine:
    def __init__(self):
        self.instances = [RPNInstance(id=i) for i in range(18)]

# NEW: Dynamic pool integration
class ModularRPNEngine:
    def __init__(self, pool: MathCorePool | None = None, instance_id: int | None = None):
        """
        Args:
            pool: Shared MathCorePool (if None, use global pool)
            instance_id: Specific core to use (if None, spawn on-demand)
        """
        self.pool = pool or _get_global_pool()
        self.instance_id = instance_id
        self._owned = instance_id is None  # Track if we spawned the core

    def _ensure_core(self, tier: int = 1) -> int:
        """Ensure we have a core allocated."""
        if self.instance_id is None:
            self.instance_id = self.pool.spawn_core(tier=tier)
        return self.instance_id

    def execute(self, program: str, tier: int = 1):
        """Execute RPN program on allocated core."""
        core_id = self._ensure_core(tier=tier)
        # ... existing execution logic

    def __del__(self):
        """Release core on cleanup."""
        if self._owned and self.instance_id is not None:
            self.pool.release_core(self.instance_id)
```

**Backward Compatibility:**
- Keep `_INSTANCE_COUNT = 18` as default for legacy code
- Existing code using `ModularRPNEngine()` still works (uses global pool)
- New code can opt-in to dynamic spawning via `pool` parameter

---

### Task 3: Update RealityGalaxy (Integration)
**File:** `knowledge3d/cranium/reality_galaxy.py`

**Changes:**
```python
class RealityGalaxy:
    def __init__(
        self,
        galaxy_path: Optional[Path] = None,
        *,
        rpn_engine: Optional[ModularRPNEngine] = None,
        compressor: Optional[AdaptiveDimensionCompressor] = None,
        math_core_pool: Optional[MathCorePool] = None,  # NEW
    ):
        # ...
        self._math_core_pool = math_core_pool or MathCorePool()
        self._rpn = rpn_engine or ModularRPNEngine(pool=self._math_core_pool)

    def add_node(self, node: RealityNode, *, encode_embedding: bool = False):
        """Add node and optionally spawn dedicated Math Core."""
        if isinstance(node, RealitySystem) and node.rpn_instance is None:
            # Dynamic spawning: allocate core on-demand
            node.rpn_instance = self._math_core_pool.spawn_core(
                tier=node.rpn_tier
            )

        # ... rest of existing logic

    def remove_node(self, node_id: str):
        """Remove node and release its Math Core."""
        node = self.nodes.pop(node_id, None)
        if isinstance(node, RealitySystem) and node.rpn_instance is not None:
            self._math_core_pool.release_core(node.rpn_instance)
```

---

### Task 4: Update reality_physics_export.py (Dynamic Allocation)
**File:** `knowledge3d/cranium/reality_physics_export.py`

**Changes:**
```python
# OLD: Static instance assignment
def export_projectile_2d(params: Dict | None = None) -> RealitySystem:
    return RealitySystem(
        node_id="system:projectile_2d",
        rpn_tier=1,
        rpn_instance=2,  # HARD-CODED
        # ...
    )

# NEW: Dynamic allocation (instance=None signals auto-allocate)
def export_projectile_2d(
    params: Dict | None = None,
    auto_allocate: bool = True  # NEW
) -> RealitySystem:
    return RealitySystem(
        node_id="system:projectile_2d",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 2,  # Auto or static
        # ...
    )
```

**Backward Compatibility:**
- `auto_allocate=False` uses static assignment (Phase 4B behavior)
- `auto_allocate=True` (default) uses dynamic spawning

---

### Task 5: Add Dynamic Spawning Tests
**File:** `knowledge3d/cranium/tests/test_math_core_pool.py` (NEW)

**Tests:**
```python
def test_math_core_pool_query_capacity():
    """Test GPU capacity query."""
    pool = MathCorePool(gpu_id=0)
    assert pool.max_cores > 18, "Should support more than 18 cores"
    assert pool.max_cores < 10000, "Should have reasonable upper bound"


def test_math_core_pool_spawn_100_cores():
    """Test spawning 100 cores dynamically."""
    pool = MathCorePool(gpu_id=0)

    core_ids = []
    for i in range(100):
        core_id = pool.spawn_core(tier=1)
        core_ids.append(core_id)

    assert len(set(core_ids)) == 100, "All core IDs should be unique"
    assert len(pool.active_cores) == 100


def test_math_core_pool_release_and_reuse():
    """Test releasing cores back to pool."""
    pool = MathCorePool(gpu_id=0)

    # Spawn and release
    core_id = pool.spawn_core(tier=1)
    pool.release_core(core_id, pool=True)

    # Reuse
    core_id2 = pool.spawn_core(tier=1, reuse=True)
    assert core_id == core_id2, "Should reuse pooled core"


def test_math_core_pool_capacity_limit():
    """Test hitting capacity limit."""
    pool = MathCorePool(gpu_id=0)
    pool.max_cores = 10  # Artificially limit

    # Spawn 10 cores
    for i in range(10):
        pool.spawn_core(tier=1)

    # 11th should raise
    with pytest.raises(RuntimeError, match="at capacity"):
        pool.spawn_core(tier=1)


def test_reality_galaxy_dynamic_allocation():
    """Test RealityGalaxy with dynamic Math Core allocation."""
    galaxy = RealityGalaxy()

    # Add 50 systems
    systems = [export_projectile_2d(auto_allocate=True) for _ in range(50)]
    for sys in systems:
        galaxy.add_node(sys)

    # Verify all have unique instance IDs
    instance_ids = [galaxy.nodes[sys.node_id].rpn_instance for sys in systems]
    assert len(set(instance_ids)) == 50, "All systems should have unique cores"

    # Verify all step successfully
    for sys in systems:
        state = galaxy.step_system(sys.node_id, n_steps=1)
        assert state is not None
```

**File:** `knowledge3d/cranium/tests/test_reality_integration.py` (UPDATE)

**Add stress test:**
```python
def test_1000_systems_dynamic_spawning():
    """Stress test: 1000 physics systems with dynamic core allocation."""
    galaxy = RealityGalaxy()

    # Mix of system types
    systems = []
    for i in range(1000):
        if i % 3 == 0:
            systems.append(export_projectile_2d(auto_allocate=True))
        elif i % 3 == 1:
            systems.append(export_point_charge_2d(auto_allocate=True))
        else:
            systems.append(export_lc_circuit(auto_allocate=True))

    # Add all systems
    for sys in systems:
        galaxy.add_node(sys)

    # Step all systems (parallel execution)
    import time
    start = time.perf_counter()

    for sys in systems:
        galaxy.step_system(sys.node_id, n_steps=1)

    elapsed = time.perf_counter() - start

    print(f"\n  1000 systems stepped in {elapsed:.3f}s")
    print(f"  Throughput: {1000/elapsed:.1f} systems/sec")
    print(f"  Avg latency: {elapsed/1000*1000:.3f} ms/system")

    # Should complete in reasonable time
    assert elapsed < 5.0, f"1000 systems took {elapsed:.3f}s (should be <5s)"
```

---

## Integration with Existing Architecture

### Tier Routing (Unchanged)
```python
# TieredRPNEngine still routes based on opcode complexity
# But now spawns cores dynamically instead of using fixed slots

tiered_engine = TieredRPNEngine(pool=math_core_pool)
result = tiered_engine.execute(program)  # Auto-spawns appropriate tier
```

### Reality Enabler (Enhanced)
```python
# Phase 4B: Static 13 systems, 12 cores
# Phase 5+: Dynamic 1000s systems, 1000s cores

galaxy = RealityGalaxy()

# Add 500 buildings (structural dynamics)
for i in range(500):
    building = export_structural_system(auto_allocate=True)
    galaxy.add_node(building)

# Add 1000 vehicles (kinematics)
for i in range(1000):
    vehicle = export_vehicle_physics(auto_allocate=True)
    galaxy.add_node(vehicle)

# All 1500 systems get dedicated Math Cores
# Limited only by GPU hardware (not artificial 18-core limit)
```

---

## Performance Targets

### Spawning Overhead
- **Target:** <1ms per core spawn
- **Measured:** TBD (Codex to benchmark)

### Memory Overhead
- **Target:** <1% of VRAM for 10,000 cores
- **Calculation:** 10,000 × 2KB = 20MB (<1% of 8GB)

### Throughput
- **Phase 4B Baseline:** 69,779 steps/sec (13 systems, 12 cores)
- **Phase 5 Target:** 5,000,000 steps/sec (1000 systems, 1000 cores)

### Scaling Efficiency
- **Linear scaling expected:** 2× cores = 2× throughput
- **Test on:** RTX 3070 (460 cores), RTX 4090 (1280 cores)

---

## Documentation Updates

### Files to Update
1. **[MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md)** — ✅ DONE (Tesla/Setun heritage, dynamic spawning)
2. **[REALITY_ENABLER_SPECIFICATION.md](docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md)** — Add dynamic allocation section
3. **[CLAUDE.md](CLAUDE.md)** — Update "Core Architecture" with dynamic spawning
4. **[BRIEFING.md](docs/BRIEFING.md)** — Update math core section (instantiable templates)

### New Documentation
- **[docs/DYNAMIC_MATH_CORE_SPAWNING.md](docs/DYNAMIC_MATH_CORE_SPAWNING.md)** — Developer guide
  - API reference (`MathCorePool`)
  - Usage examples
  - Performance tuning
  - Multi-GPU federation (future)

---

## Timeline

**Estimated:** 2-3 days for complete implementation

**Day 1 (Codex):**
- Implement `MathCorePool` class
- Add GPU capacity query
- Write unit tests (spawning, pooling, capacity limits)

**Day 2 (Codex):**
- Refactor `ModularRPNEngine` for dynamic allocation
- Update `RealityGalaxy` integration
- Update `reality_physics_export.py` (auto_allocate flag)
- Backward compatibility validation

**Day 3 (Codex + Claude):**
- Add integration tests (50/100/1000 systems)
- Performance benchmarks (spawning overhead, throughput)
- Documentation updates
- Completion report

---

## Success Criteria

Phase 5 Dynamic Spawning complete when:

**Implementation:**
- ✅ `MathCorePool` implemented with GPU capacity query
- ✅ `ModularRPNEngine` supports dynamic allocation
- ✅ `RealityGalaxy` spawns cores on-demand
- ✅ Backward compatibility maintained (Phase 4B tests still pass)

**Testing:**
- ✅ 10 unit tests for `MathCorePool` (spawning, pooling, limits)
- ✅ 5 integration tests (50/100/1000 systems)
- ✅ All 48 Phase 4B tests still passing

**Performance:**
- ✅ Can spawn 100 cores in <100ms
- ✅ Can step 1000 systems in <5s
- ✅ Memory overhead <1% of VRAM
- ✅ Linear scaling validated (2× cores = 2× throughput)

**Documentation:**
- ✅ Math Core spec updated (v2.0)
- ✅ Developer guide written
- ✅ API reference complete
- ✅ Completion report (PHASE5_DYNAMIC_SPAWNING_COMPLETE.md)

---

## Critical Implementation Notes

### Thread Safety
**Requirement:** `MathCorePool` MUST be thread-safe
**Reason:** Multiple systems may spawn cores concurrently
**Solution:** Use threading.Lock for spawn/release operations

```python
import threading

class MathCorePool:
    def __init__(self):
        self._lock = threading.Lock()
        # ...

    def spawn_core(self, tier: int) -> int:
        with self._lock:
            # Atomic spawn operation
            return self._spawn_core_impl(tier)
```

### GPU Context Management
**Issue:** Each core needs valid CUDA context
**Solution:** Use CuPy device management

```python
import cupy as cp

class MathCore:
    def __init__(self, gpu_id: int):
        with cp.cuda.Device(gpu_id):
            self.context = cp.cuda.runtime.getCurrentContext()
```

### Memory Leaks Prevention
**Issue:** Cores not released → memory leak
**Solution:** Use weak references + timeout

```python
import weakref
import time

class MathCorePool:
    def __init__(self):
        self._last_used: Dict[int, float] = {}
        self._timeout = 60.0  # 60s idle timeout

    def _cleanup_idle_cores(self):
        """Periodically deallocate idle cores."""
        now = time.time()
        to_remove = []
        for core_id, last_used in self._last_used.items():
            if now - last_used > self._timeout:
                to_remove.append(core_id)

        for core_id in to_remove:
            self.release_core(core_id, pool=False)
```

---

## Coordination with Claude

**Claude's Role:**
- Review Codex's implementation for architectural alignment
- Write performance benchmarks
- Document results in completion report
- Update specifications

**Codex's Role:**
- Implement `MathCorePool`, refactor `ModularRPNEngine`
- Write unit/integration tests
- Ensure backward compatibility
- Measure performance metrics

**Shared Goal:**
- Enable 1000s of concurrent physics systems
- Maintain sub-1ms latency per system
- Preserve Tesla 3-6-9 and Setun ternary heritage
- Prepare for Phase 6 (multi-GPU scaling)

---

## Ready to Execute

**This briefing is comprehensive and actionable.** Codex has clear implementation tasks, success criteria, and timeline.

**Daniel's approval to proceed:**

Once you say "go," Codex will:
1. Implement `MathCorePool` with GPU capacity query
2. Refactor `ModularRPNEngine` for dynamic allocation
3. Update `RealityGalaxy` and `reality_physics_export.py`
4. Write 15+ tests (unit + integration)
5. Validate 1000-system stress test
6. Document in completion report

**Timeline:** 2-3 days to unlock GPU-limited Math Core spawning

**This transforms K3D from "18 parallel physics engines" to "unlimited parallel physics engines constrained only by GPU hardware."** 🚀

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**For:** Codex (Implementation Lead)
**Date:** November 24, 2025
**Phase:** 5 Dynamic Math Core Spawning
**Paradigm:** Instantiable templates, Tesla/Setun heritage, GPU-limited scaling
