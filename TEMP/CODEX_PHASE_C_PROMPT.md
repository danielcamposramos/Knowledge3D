# Codex Phase C: LED Pathfinder Sovereign Migration

**Status**: Phase B complete (Morton octree + RPN sorting, 252 tests, 116MB GPU) ✅

## Momentum Context (Fresh from Phase B Win)

**What just happened**: Your RPN sorting implementation validated Daniel's multi-instance vision from inception. By using RPN for comparisons instead of CuPy Thrust, you demonstrated:
- 15.6KB solution replacing 12GB library (777,000x smaller!)
- Zero CPU↔GPU transfers (data stays where it lives)
- Modular expansion path (add opcodes = add capabilities)

**Daniel's words**: *"Apollo 11 inspirations and sci-fi aspirations"* - the sovereign architecture reflects this: minimal, modular, mission-critical.

---

## Phase C Objectives

**Goal**: Create LED (Language-Embedded Definitions) pathfinder wrapper using A* + distance kernels

**Key Innovation**: RPN-based priority queue for A* frontier (replaces CuPy heaps)

**Files to Create/Modify**:
1. `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py` (new wrapper)
2. `tests/test_led_pathfinder.py` (update imports, remove skip)

**Time Estimate**: 1.5 hours

**Existing PTX Assets**:
- `led_astar.ptx` (12KB) - A* pathfinding kernel
- `l2_dist_warp.ptx` (4.8KB) - Vectorized distance calculations

---

## Implementation Template

See `SPATIAL_KERNEL_ASSESSMENT.md` Template 3 for complete structure.

### Core Pattern

```python
# knowledge3d/cranium/spatial_sovereign/led_pathfinder.py
from knowledge3d.cranium.sovereign.loader import load_ptx_kernel
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine
from knowledge3d.cranium.spatial_sovereign.morton_octree import MortonOctreeSovereign

class LEDPathfinderSovereign:
    def __init__(self):
        # Load existing PTX kernels
        self.astar_kernel = load_ptx_kernel("led_astar.ptx", "astar_step_kernel")
        self.dist_kernel = load_ptx_kernel("l2_dist_warp.ptx", "l2_distance_kernel")

        # RPN for priority queue operations (replaces CuPy heaps!)
        self.rpn = ModularRPNEngine()

        # Morton octree for spatial hashing
        self.octree = MortonOctreeSovereign()

    def find_path(self, start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """A* pathfinding using sovereign kernels + RPN priority queue."""
        # Encode space with Morton codes
        # Use RPN for frontier priority queue
        # Execute A* kernel steps
        # Return path
        pass

    def compute_distances(self, points: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Vectorized L2 distance using warp-optimized kernel."""
        # Use l2_dist_warp.ptx (SIMD-style computation)
        pass
```

---

## The RPN Priority Queue Innovation

**A* Algorithm Needs**:
- Priority queue: pop minimum cost node
- Traditional: CuPy heap or CPU sorting (slow!)

**Your RPN Solution**:
```python
def rpn_priority_queue_pop(self, frontier_costs: np.ndarray, frontier_nodes: np.ndarray):
    """Use RPN to find and extract minimum cost node."""
    # Program RPN with min-finding opcodes
    op_codes = [
        # Vector reduction: find minimum
        # Parallel compare-and-swap
        # Extract node index
    ]
    result = self.rpn.execute_single(
        instance_id=0,
        op_codes=op_codes,
        scalars=frontier_costs
    )
    min_idx = int(result.stack[-1])
    return frontier_nodes[min_idx], min_idx
```

**Why This Matters**:
- Priority queue operations stay on GPU
- RPN opcodes = modular expansion (add heap operations later!)
- Zero library dependencies (CuPy-free)

---

## Step-by-Step Instructions

### Step 1: Create LED Pathfinder Wrapper (45 min)

1. **Create** `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py`

2. **Load PTX kernels**:
   - `led_astar.ptx` → A* step kernel
   - `l2_dist_warp.ptx` → Distance calculations

3. **Implement core methods**:
   ```python
   def find_path(self, start, goal, obstacles):
       # 1. Encode obstacles with Morton octree
       obstacle_codes = self.octree.encode(obstacles)

       # 2. Initialize A* frontier (open set)
       frontier = [start]
       g_costs = {tuple(start): 0.0}

       # 3. A* loop
       while frontier:
           # Use RPN to pop minimum f-cost node
           current, idx = self.rpn_priority_queue_pop(...)

           # Execute A* kernel step (neighbor expansion)
           neighbors = self.astar_kernel(current, obstacle_codes)

           # Update costs and frontier
           for neighbor in neighbors:
               new_cost = g_costs[current] + self.compute_distances(...)
               if new_cost < g_costs.get(neighbor, float('inf')):
                   g_costs[neighbor] = new_cost
                   frontier.append(neighbor)

       return reconstruct_path(...)

   def compute_distances(self, points, reference):
       # Vectorized L2 via l2_dist_warp.ptx
       # Warp-optimized (32 threads process 32 points simultaneously)
       pass
   ```

4. **Add RPN priority queue helper**:
   ```python
   def rpn_priority_queue_pop(self, frontier_costs, frontier_nodes):
       """Extract minimum cost node using RPN reduction."""
       # Use RPN opcodes for parallel min-finding
       # This is where modular expansion shines!
       pass
   ```

### Step 2: Legacy Import Forwarding (10 min)

```python
# knowledge3d/spatial/led_pathfinder.py (update existing)
from knowledge3d.cranium.spatial_sovereign.led_pathfinder import LEDPathfinderSovereign as LEDPathfinder

__all__ = ['LEDPathfinder']
```

### Step 3: Update Tests (20 min)

```python
# tests/test_led_pathfinder.py
from knowledge3d.cranium.spatial_sovereign.led_pathfinder import LEDPathfinderSovereign
import numpy as np
import pytest

class TestLEDPathfinder:
    def test_straight_path(self):
        """Test A* finds straight path with no obstacles."""
        pathfinder = LEDPathfinderSovereign()
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        obstacles = np.array([], dtype=np.float32).reshape(0, 3)

        path = pathfinder.find_path(start, goal, obstacles)
        assert path.shape[0] >= 2  # At least start and goal
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)

    def test_obstacle_avoidance(self):
        """Test A* navigates around obstacles."""
        pathfinder = LEDPathfinderSovereign()
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        obstacles = np.array([
            [5.0, 0.0, 0.0],  # Wall blocking direct path
            [5.0, 1.0, 0.0],
            [5.0, -1.0, 0.0]
        ], dtype=np.float32)

        path = pathfinder.find_path(start, goal, obstacles)
        # Path should detour around obstacles
        assert path.shape[0] > 2  # More than straight line

    def test_distance_computation(self):
        """Test L2 distance kernel."""
        pathfinder = LEDPathfinderSovereign()
        points = np.random.rand(100, 3).astype(np.float32)
        reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        distances = pathfinder.compute_distances(points, reference)
        assert distances.shape == (100,)
        assert np.all(distances >= 0)

    def test_rpn_priority_queue(self):
        """Test RPN-based priority queue operations."""
        pathfinder = LEDPathfinderSovereign()
        costs = np.array([5.0, 2.0, 8.0, 1.0, 6.0], dtype=np.float32)
        nodes = np.arange(5, dtype=np.int32)

        min_node, min_idx = pathfinder.rpn_priority_queue_pop(costs, nodes)
        assert min_idx == 3  # Index of minimum (1.0)
        assert min_node == 3
```

**Remove skip marker**:
```python
# DELETE: @pytest.mark.skip(reason="CuPy deprecated")
```

### Step 4: Verify and Test (15 min)

```bash
# Run LED pathfinder tests
pytest tests/test_led_pathfinder.py -xvs

# Run full spatial suite
pytest tests/test_frustum_culling.py tests/test_morton_octree.py tests/test_led_pathfinder.py -xvs

# Run full test suite
pytest tests/ -x --tb=short

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

---

## Success Criteria

✅ **All tests passing**: 260+ tests (LED tests now included)
✅ **GPU memory**: <400MB (additional kernels loaded)
✅ **RPN priority queue**: Proven viable for A* frontier management
✅ **Distance kernel**: Warp-optimized L2 working correctly
✅ **Sovereignty intact**: Zero CuPy, pure ctypes + CUDA Driver API

---

## The Modular Expansion Path (Why This Matters)

**Current RPN opcodes**: ~75 (arithmetic, vector, stack, conditional)

**Phase C adds conceptually**:
- Priority queue operations (min-finding, reduction)
- Frontier management (push/pop patterns)

**Future expansion** (just add opcodes to modular_rpn_kernel.ptx):
- Heap operations (heapify, sift-up, sift-down)
- Graph algorithms (Dijkstra, bidirectional A*)
- Spatial queries (range search, k-NN)

**Apollo 11 principle**: Each opcode is like a circuit in AGC - simple, testable, composable. The complexity emerges from composition, not individual parts.

---

## What to Report Back

When Phase C is complete, report:

1. **Test Status**:
   - Total tests passing (should be 260+)
   - LED pathfinding tests working?
   - GPU memory usage (nvidia-smi)

2. **RPN Priority Queue**:
   - Did RPN min-finding work for A* frontier?
   - Performance vs. CuPy heaps? (if measurable)
   - Any insights on opcode expansion?

3. **Distance Kernel**:
   - L2 warp kernel performance
   - Vectorization speedup? (if measurable)

4. **Integration**:
   - Morton octree used for obstacle encoding?
   - Sovereign wrapper pattern consistent?

5. **Ready for Phase D**: Yes/No (semantic navigator next - composes all above!)

---

## Reference Files

- **Existing PTX**:
  - `knowledge3d/cranium/ptx/led_astar.ptx` (12KB, A* kernel ready)
  - `knowledge3d/cranium/ptx/l2_dist_warp.ptx` (4.8KB, SIMD distances)
- **RPN Engine**: `knowledge3d/cranium/bridges/sovereign_bridges.py:986-1066`
- **Morton Octree**: `knowledge3d/cranium/spatial_sovereign/morton_octree.py` (just completed!)
- **Template**: `SPATIAL_KERNEL_ASSESSMENT.md` Template 3
- **Architecture**: `RPN_SWARM_ARCHITECTURE.md`

---

## Technical Notes

### Warp-Optimized Distance Kernel

The `l2_dist_warp.ptx` kernel uses SIMD-style parallelism:
- 1 warp (32 threads) processes 32 points simultaneously
- Coalesced memory access patterns
- Vectorized sqrt operations

**Expected speedup**: 10-30x vs. sequential CPU computation

### RPN as Priority Queue

Traditional A* uses heaps (log N push/pop). Your RPN approach:
- Small frontiers (<100 nodes): RPN parallel scan (O(N), but fast on GPU)
- Large frontiers: Can add heap opcodes to RPN (modular expansion!)

**Trade-off**: Simplicity now, scalability later (via opcode additions)

---

## Mission Context (Sci-Fi Aspirations)

**After Phase C**, your agents will have:
- **Vision** (frustum culling) - see the world
- **Spatial awareness** (Morton octree) - organize the world
- **Navigation** (LED pathfinding) - move through the world

**Phase D** (semantic navigator) composes these into **spatial intelligence**.

**Phase E** (cleanup) polishes for the **9-agent swarm** deployment.

You're building the AGC (Apollo Guidance Computer) of spatial AI:
- Minimal footprint (400MB vs. competitors' GBs)
- Modular expansion (add opcodes like Apollo added programs)
- Mission-critical reliability (sovereign = no external dependencies)

---

**Proceed with Phase C!** The momentum is strong, the context is fresh, the architecture is proven. 🚀

**"That's one small step for a pathfinder, one giant leap for swarm-kind."** 😉
