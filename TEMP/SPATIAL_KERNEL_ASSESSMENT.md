# Spatial Kernel Assessment & Integration Strategy

**Date:** 2025-10-15  
**Purpose:** Consolidate CuPy spatial modules into sovereign PTX architecture  
**Context:** STEP 13-B Phase 1 complete, 8 legacy tests skipped awaiting migration  

---

## Executive Summary

**Current State:**
- ✅ **21 sovereign PTX kernels** already compiled and ready
- ⚠️ **8 legacy CuPy spatial modules** causing 12GB memory spikes
- ✅ **Sovereign loader** (`knowledge3d.cranium.sovereign.loader`) fully functional
- 🎯 **Goal:** Migrate spatial to sovereign WITHOUT code duplication

**Key Discovery:**
- `morton_octree.ptx` ALREADY EXISTS (8.4KB, compiled)
- `led_astar.ptx` ALREADY EXISTS (12KB, compiled A* pathfinder)
- `frustum_cull_simd.ptx` ALREADY EXISTS (5.5KB, SIMD optimized by Kimi)

**Conclusion:** 
We don't need to CREATE kernels - we need to CREATE SOVEREIGN WRAPPERS for existing PTX!

---

## Existing Sovereign PTX Kernels (Analysis)

### Spatial Kernels Already Compiled:

| Kernel | Size | Function | Status |
|--------|------|----------|--------|
| `morton_octree.ptx` | 8.4KB | Morton code computation | ✅ Ready |
| `led_astar.ptx` | 12KB | LED A* navigation | ✅ Ready |
| `frustum_cull_simd.ptx` | 5.5KB | View frustum culling (Kimi) | ✅ Ready |
| `l2_dist_warp.ptx` | 3.3KB | L2 distance (for navigation) | ✅ Ready |

### Related Kernels (Already Sovereign):

| Kernel | Size | Function | Usage |
|--------|------|----------|-------|
| `modular_rpn_kernel.ptx` | 34KB | RPN operations | ModularRPNEngine |
| `dynamic_lod_tune.ptx` | 3.6KB | LOD tuning | ThinkingTagBridge |
| `generate_shape_kernel.ptx` | 3.3KB | Shape generation | Shape primitives |
| `galaxy_resonance_engine_extended.ptx` | 11KB | Semantic embeddings | Galaxy |

**Total:** 21 PTX kernels, ~220KB compiled, ready to use!

---

## CuPy Modules to Migrate (Assessment)

### 1. LED Pathfinder
**File:** `knowledge3d/spatial/led_pathfinder.py` (580 lines, CuPy-based)

**Functions:**
- A* pathfinding in dependency graphs
- L2 distance calculations
- Frontier expansion

**Existing PTX:**
- ✅ `led_astar.ptx` (12KB) - A* navigation kernel
- ✅ `l2_dist_warp.ptx` (3.3KB) - Distance calculations

**Migration Strategy:** 
Create `knowledge3d.cranium.spatial_sovereign.led_pathfinder.py` wrapping existing PTX.

**Effort:** LOW (kernels exist, just need Python wrapper)

---

### 2. Semantic Navigator
**File:** `knowledge3d/spatial/semantic_navigator.py` (450 lines, CuPy-based)

**Functions:**
- Multi-modal graph navigation
- Semantic proximity search
- Strategy pattern routing

**Dependencies:**
- Uses LED Pathfinder internally
- Uses Morton octree for spatial indexing

**Existing PTX:**
- ✅ `led_astar.ptx` (via LED wrapper)
- ✅ `morton_octree.ptx` (spatial indexing)
- ✅ `galaxy_resonance_engine_extended.ptx` (semantic embeddings)

**Migration Strategy:**
Create `knowledge3d.cranium.spatial_sovereign.semantic_navigator.py` composing existing wrappers.

**Effort:** MEDIUM (depends on LED wrapper, composition layer needed)

---

### 3. Morton Octree
**File:** `knowledge3d/spatial/morton_octree.py` (310 lines, CuPy-based)

**Functions:**
- Morton code encoding (3D → 1D)
- Radix sort via CuPy/Thrust
- Range queries

**Existing PTX:**
- ✅ `morton_octree.ptx` (8.4KB) - `compute_morton_codes` entry point

**Missing:**
- ⚠️ Radix sort (currently CuPy Thrust backend)

**Migration Strategy:**
1. Use existing `morton_octree.ptx` for encoding
2. Add thrust radix sort wrapper OR use ModularRPN for sorting

**Effort:** MEDIUM (sort wrapper needed, or leverage RPN)

---

### 4. Frustum Culler
**File:** `knowledge3d/spatial/frustum.py` (180 lines, CuPy-based)

**Functions:**
- View frustum culling (SIMD optimized by Kimi)
- 6-plane frustum tests
- Warp-level parallelism

**Existing PTX:**
- ✅ `frustum_cull_simd.ptx` (5.5KB) - `warp_frustum_cull_simd` entry

**Migration Strategy:**
Direct wrapper - kernel is already complete and sovereign!

**Effort:** LOW (trivial wrapper, kernel complete)

---

## Integration Architecture

### Proposed Structure:

```
knowledge3d/
├── cranium/
│   ├── ptx/                           # EXISTING - 21 kernels
│   │   ├── morton_octree.ptx          ✅ Ready
│   │   ├── led_astar.ptx              ✅ Ready
│   │   ├── frustum_cull_simd.ptx      ✅ Ready
│   │   ├── l2_dist_warp.ptx           ✅ Ready
│   │   └── ... (17 other kernels)
│   │
│   ├── sovereign/                     # EXISTING - Core loader
│   │   ├── loader.py                  ✅ Pure ctypes CUDA driver API
│   │   └── trm_launcher.py            ✅ Example usage
│   │
│   └── spatial_sovereign/             # NEW - Spatial wrappers
│       ├── __init__.py                
│       ├── frustum_culler.py          ⭐ Trivial wrapper
│       ├── morton_octree.py           ⭐ Wrapper + sort integration
│       ├── led_pathfinder.py          ⭐ A* wrapper
│       └── semantic_navigator.py      ⭐ Composition of above
│
└── spatial/                           # DEPRECATED - Move to Old_Attempts/
    ├── led_pathfinder.py              → Old_Attempts/cupy_spatial/
    ├── semantic_navigator.py          → Old_Attempts/cupy_spatial/
    ├── morton_octree.py               → Old_Attempts/cupy_spatial/
    └── frustum.py                     → Old_Attempts/cupy_spatial/
```

**Key Principle:** 
- **NO NEW KERNELS** - Reuse existing PTX!
- **THIN WRAPPERS** - Just sovereign.loader glue code
- **COMPOSITION** - Navigator uses Pathfinder uses Octree

---

## Kernel Reuse Opportunities

### 1. ModularRPN for Sorting
`modular_rpn_kernel.ptx` has sorting operations - can replace CuPy Thrust!

**Code Pattern:**
```python
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

rpn = ModularRPNEngine()
sorted_codes = rpn.sort(morton_codes)  # Uses PTX, not CuPy!
```

### 2. Galaxy Resonance for Semantic Search
`galaxy_resonance_engine_extended.ptx` has embedding similarity - reuse!

**Code Pattern:**
```python
from knowledge3d.cranium.ptx_runtime.galaxy_memory_updater import search_nearest

nearest = search_nearest(query_embedding, graph_nodes)  # Sovereign PTX
```

### 3. L2 Distance Warp for Navigation
`l2_dist_warp.ptx` replaces CuPy linalg.norm!

**Code Pattern:**
```python
from knowledge3d.cranium.sovereign.loader import load_ptx_file

l2_kernel = load_ptx_file("knowledge3d/cranium/ptx/l2_dist_warp.ptx", "compute_l2_dist")
# Use instead of cp.linalg.norm()
```

---

## Migration Roadmap (Modular & Incremental)

### Phase A: Frustum Culler (30 min)
**Why First:** Trivial - kernel is complete, just needs wrapper.

**Steps:**
1. Create `cranium/spatial_sovereign/frustum_culler.py`
2. Wrap `frustum_cull_simd.ptx` using sovereign.loader
3. Update `tests/test_frustum_culling.py` imports
4. Remove skip marker
5. Verify GPU memory <200MB

**Success:** 1-2 tests passing, frustum module sovereign

---

### Phase B: Morton Octree (1 hour)
**Why Second:** Standalone, used by other modules.

**Steps:**
1. Create `cranium/spatial_sovereign/morton_octree.py`
2. Wrap `morton_octree.ptx` for encoding
3. Replace CuPy radix sort with ModularRPN sort
4. Update `tests/test_morton_octree.py` imports
5. Remove skip marker
6. Verify GPU memory <300MB

**Success:** 1-2 tests passing, octree module sovereign

---

### Phase C: LED Pathfinder (1.5 hours)
**Why Third:** Depends on octree, used by navigator.

**Steps:**
1. Create `cranium/spatial_sovereign/led_pathfinder.py`
2. Wrap `led_astar.ptx` and `l2_dist_warp.ptx`
3. Use MortonOctree from Phase B
4. Update `tests/test_led_pathfinder.py` imports
5. Remove skip marker
6. Verify GPU memory <500MB

**Success:** 2-3 tests passing, pathfinder sovereign

---

### Phase D: Semantic Navigator (1 hour)
**Why Fourth:** Composes all above modules.

**Steps:**
1. Create `cranium/spatial_sovereign/semantic_navigator.py`
2. Use LEDPathfinder, MortonOctree from above
3. Integrate galaxy_resonance for semantic search
4. Update navigator tests
5. Remove all skip markers
6. Verify GPU memory <800MB

**Success:** All 8 tests passing, spatial stack sovereign!

---

### Phase E: Cleanup & Documentation (30 min)
**Why Last:** Polish and archive.

**Steps:**
1. Move `knowledge3d/spatial/*.py` → `Old_Attempts/cupy_spatial/`
2. Update imports across codebase
3. Document migration in `docs/SPATIAL_MIGRATION.md`
4. Update TEST_LOG.md
5. Regenerate baseline report

**Success:** Zero CuPy imports, complete sovereignty

---

## Code Templates for Codex

### Template 1: Frustum Culler Wrapper

```python
# knowledge3d/cranium/spatial_sovereign/frustum_culler.py
"""Sovereign frustum culler using pre-compiled PTX (Kimi's SIMD optimized)."""
import numpy as np
from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh

class FrustumCullerSovereign:
    """GPU-accelerated view frustum culling (sovereign PTX)."""
    
    def __init__(self):
        # Load Kimi's SIMD-optimized kernel
        self.kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/frustum_cull_simd.ptx",
            "warp_frustum_cull_simd"
        )
    
    def cull(self, positions, view_proj_matrix):
        """Cull nodes outside view frustum.
        
        Args:
            positions: (N, 3) numpy array of positions
            view_proj_matrix: (4, 4) view-projection matrix
        
        Returns:
            visible_mask: (N,) boolean array
        """
        n = positions.shape[0]
        
        # Allocate GPU memory
        pos_gpu = gpu_malloc(positions.nbytes)
        visible_gpu = gpu_malloc(n)
        
        # Copy to GPU
        memcpy_htod(pos_gpu, positions)
        
        # Launch kernel (32 threads per warp)
        blocks = (n + 31) // 32
        self.kernel(
            grid=(blocks, 1, 1),
            block=(32, 1, 1),
            args=[pos_gpu, np.uint32(n), visible_gpu]
        )
        
        # Copy result back
        visible = np.zeros(n, dtype=np.uint8)
        memcpy_dtoh(visible, visible_gpu)
        
        return visible.astype(bool)
```

### Template 2: Morton Octree Wrapper (with RPN sort)

```python
# knowledge3d/cranium/spatial_sovereign/morton_octree.py
"""Sovereign Morton octree using PTX + ModularRPN for sorting."""
import numpy as np
from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

class MortonOctreeSovereign:
    """GPU-accelerated spatial indexing via Morton codes (sovereign)."""
    
    def __init__(self):
        # Load Morton encoding kernel
        self.encoder = load_ptx_file(
            "knowledge3d/cranium/ptx/morton_octree.ptx",
            "compute_morton_codes"
        )
        # Use RPN for sorting (replaces CuPy Thrust)
        self.rpn = ModularRPNEngine()
    
    def build(self, positions):
        """Build octree from positions.
        
        Args:
            positions: (N, 3) numpy array
        
        Returns:
            sorted_codes: (N,) Morton codes
            sorted_indices: (N,) original indices
        """
        n = positions.shape[0]
        
        # Compute bounding box
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        
        # Allocate GPU memory
        pos_gpu = gpu_malloc(positions.nbytes)
        codes_gpu = gpu_malloc(n * 8)  # uint64
        
        # Copy positions
        memcpy_htod(pos_gpu, positions)
        
        # Encode Morton codes
        blocks = (n + 255) // 256
        self.encoder(
            grid=(blocks, 1, 1),
            block=(256, 1, 1),
            args=[pos_gpu, np.uint32(n), codes_gpu, 
                  np.float32(bbox_min[0]), np.float32(bbox_min[1]), np.float32(bbox_min[2]),
                  np.float32(bbox_max[0] - bbox_min[0])]
        )
        
        # Copy codes back
        codes = np.zeros(n, dtype=np.uint64)
        memcpy_dtoh(codes, codes_gpu)
        
        # Sort using ModularRPN (replaces CuPy Thrust!)
        sorted_indices = self.rpn.argsort(codes)
        sorted_codes = codes[sorted_indices]
        
        return sorted_codes, sorted_indices
```

### Template 3: LED Pathfinder Wrapper

```python
# knowledge3d/cranium/spatial_sovereign/led_pathfinder.py
"""Sovereign LED A* pathfinder using pre-compiled PTX."""
import numpy as np
from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh

class LEDPathfinderSovereign:
    """GPU-accelerated dependency graph navigation (sovereign PTX)."""
    
    def __init__(self):
        # Load A* kernel
        self.astar_kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/led_astar.ptx",
            "led_astar_navigate"
        )
        # Load L2 distance kernel
        self.l2_kernel = load_ptx_file(
            "knowledge3d/cranium/ptx/l2_dist_warp.ptx",
            "compute_l2_dist"
        )
    
    def find_path(self, graph_edges, start_node, goal_node, goal_embedding):
        """Find shortest path in dependency graph.
        
        Args:
            graph_edges: (E, 2) edge list
            start_node: Starting node ID
            goal_node: Goal node ID
            goal_embedding: Goal semantic vector
        
        Returns:
            path: List of node IDs
        """
        n_edges = graph_edges.shape[0]
        
        # Allocate GPU memory
        edges_gpu = gpu_malloc(graph_edges.nbytes)
        path_gpu = gpu_malloc(1024 * 4)  # Max 1024 nodes in path
        
        # Copy to GPU
        memcpy_htod(edges_gpu, graph_edges)
        
        # Launch A* kernel
        self.astar_kernel(
            grid=(1, 1, 1),  # Single-block A* for LED graphs
            block=(256, 1, 1),
            args=[edges_gpu, np.uint32(start_node), np.uint32(goal_node),
                  np.float32(goal_embedding[0]), np.float32(goal_embedding[1]),
                  path_gpu, ...]
        )
        
        # Copy path back
        path = np.zeros(1024, dtype=np.uint32)
        memcpy_dtoh(path, path_gpu)
        
        # Trim to actual path length
        path_len = np.argmax(path == 0xFFFFFFFF)
        return path[:path_len].tolist()
```

---

## Success Metrics

### Technical:
- ✅ Zero CuPy imports in sovereign codebase
- ✅ GPU memory <1GB (vs. 12GB CuPy spike)
- ✅ Performance equivalent or better (PTX pre-compiled)
- ✅ All 260+ tests passing (252 current + 8 migrated)

### Architectural:
- ✅ Single loader pattern (`sovereign.loader`)
- ✅ Kernel reuse (no duplication)
- ✅ Modular composition (Navigator → Pathfinder → Octree)
- ✅ Clear deprecation path (CuPy → Old_Attempts/)

### Documentation:
- ✅ Migration guide in `docs/SPATIAL_MIGRATION.md`
- ✅ Updated TEST_LOG.md
- ✅ Baseline report regenerated
- ✅ Architecture diagrams updated

---

## Estimated Total Effort

| Phase | Effort | Output |
|-------|--------|--------|
| A: Frustum | 30 min | 1-2 tests passing |
| B: Morton | 1 hour | 1-2 tests passing |
| C: LED | 1.5 hours | 2-3 tests passing |
| D: Navigator | 1 hour | 2-3 tests passing |
| E: Cleanup | 30 min | Documentation |
| **Total** | **4.5 hours** | **260+ tests passing** |

Spread across 2-3 sessions, very achievable!

---

## Why This Approach Wins

1. **No Code Duplication** - Reuses 4 existing PTX kernels
2. **Leverages Existing Work** - Kimi's SIMD frustum, LED A*, Morton encoding
3. **Modular** - Each phase adds value independently
4. **Testable** - Tests pass incrementally
5. **Documented** - Clear before/after migration path
6. **Sovereign** - Zero external dependencies (no CuPy, no cuda-python)

---

**Ready to execute? Codex can knock this out in 2-3 sessions!** 🚀
