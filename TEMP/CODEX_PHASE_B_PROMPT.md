# Codex Phase B: Morton Octree Sovereign Migration

**Status**: Phase A complete (frustum wrapper, 252 tests, 116MB GPU) ✅

## Strategic Updates from Daniel

### RPN Multi-Instance Capability Confirmed
- Each `ModularRPNEngine()` creates independent GPU instance (15.6KB)
- 10 instances (9 agents + 1 system) = **156KB total** (negligible!)
- Architecture supports 1000s of instances if needed
- See: `/mnt/arquivos/.../TEMP/RPN_SWARM_ARCHITECTURE.md`

### GPU Memory Target Updated
- **OLD**: 2GB conservative target
- **NEW**: 3.5GB target (GTX 970 / mobile minimum)
- Rationale: Future-proof for handheld deployment
- Update in code: `knowledge3d/cranium/sovereign/loader.py`

---

## Phase B Objectives

**Goal**: Create Morton Octree sovereign wrapper using existing PTX + ModularRPN for sorting

**Key Innovation**: Use ModularRPN to replace CuPy Thrust sorting (zero dependencies!)

**Files to Create/Modify**:
1. `knowledge3d/cranium/spatial_sovereign/morton_octree.py` (new wrapper)
2. `tests/test_morton_octree.py` (update imports, remove skip)

**Time Estimate**: 1 hour

---

## Implementation Template

See `SPATIAL_KERNEL_ASSESSMENT.md` Template 2 for complete code structure.

### Core Pattern

```python
# knowledge3d/cranium/spatial_sovereign/morton_octree.py
from knowledge3d.cranium.sovereign.loader import load_ptx_kernel
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine

class MortonOctreeSovereign:
    def __init__(self):
        self.encode_kernel = load_ptx_kernel("morton_octree.ptx", "morton_encode_kernel")
        self.decode_kernel = load_ptx_kernel("morton_octree.ptx", "morton_decode_kernel")
        self.rpn = ModularRPNEngine()  # For sorting operations

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Convert 3D points to Morton codes."""
        # Use morton_encode_kernel (existing PTX)
        pass

    def sort(self, morton_codes: np.ndarray) -> np.ndarray:
        """Sort Morton codes using RPN bitonic sort."""
        # Use ModularRPN's sorting opcodes (replaces CuPy Thrust!)
        pass

    def build_tree(self, points: np.ndarray):
        """Build octree from sorted Morton codes."""
        pass
```

### RPN Sorting Strategy

**Replace CuPy Thrust with ModularRPN**:
- Existing PTX has bitonic sort opcodes
- RPN instance slots can batch sort operations
- Memory footprint: 15.6KB vs. CuPy's 12GB!

```python
def sort_using_rpn(self, values: np.ndarray) -> np.ndarray:
    """Sort array using RPN bitonic sort kernel."""
    # Program RPN with sorting opcodes
    op_codes = [...]  # Bitonic sort sequence
    result = self.rpn.execute_single(
        instance_id=0,
        op_codes=op_codes,
        scalars=values
    )
    return result.stack[-len(values):]
```

---

## Step-by-Step Instructions

### Step 1: Update GPU Memory Target (5 min)

```python
# knowledge3d/cranium/sovereign/loader.py
# Find and update:
GPU_MEMORY_TARGET_GB = 3.5  # Updated from 2.0 for GTX 970/mobile targets
```

### Step 2: Create Morton Wrapper (30 min)

1. Create `knowledge3d/cranium/spatial_sovereign/morton_octree.py`
2. Import sovereign loader and ModularRPNEngine
3. Load `morton_octree.ptx` kernels (encode/decode)
4. Implement `encode()`, `decode()`, `sort()`, `build_tree()`
5. Use RPN for sorting (not CuPy!)

### Step 3: Update Tests (15 min)

```python
# tests/test_morton_octree.py
# Update imports:
from knowledge3d.cranium.spatial_sovereign.morton_octree import MortonOctreeSovereign

# Remove skip marker:
# @pytest.mark.skip(reason="CuPy deprecated")  # DELETE THIS LINE

class TestMortonOctree:
    def test_encoding(self):
        octree = MortonOctreeSovereign()
        points = np.random.rand(100, 3).astype(np.float32)
        codes = octree.encode(points)
        assert codes.shape == (100,)

    def test_sorting_with_rpn(self):
        octree = MortonOctreeSovereign()
        codes = np.random.randint(0, 1000, size=50, dtype=np.uint32)
        sorted_codes = octree.sort(codes)
        assert np.all(sorted_codes[:-1] <= sorted_codes[1:])  # Verify sorted
```

### Step 4: Verify and Test (10 min)

```bash
# Run morton tests
pytest tests/test_morton_octree.py -xvs

# Run full suite
pytest tests/ -x --tb=short

# Check GPU memory (should be ~200-300MB)
nvidia-smi
```

---

## Success Criteria

✅ **All tests passing**: 252+ tests (morton tests now included)
✅ **GPU memory**: <300MB (verify with nvidia-smi)
✅ **Zero CuPy**: No import cupy anywhere in spatial_sovereign/
✅ **RPN sorting works**: Bitonic sort via ModularRPN proven
✅ **Sovereignty intact**: Pure ctypes + CUDA Driver API only

---

## What to Report Back

When Phase B is complete, report:

1. **Test Status**:
   - Total tests passing
   - Any new failures/skips
   - GPU memory usage (nvidia-smi output)

2. **RPN Sorting**:
   - Did ModularRPN sorting work?
   - Performance vs. CuPy Thrust? (if measured)
   - Any issues with RPN instance allocation?

3. **Files Modified**:
   - List of new/changed files
   - Line counts (git diff --stat)

4. **Ready for Phase C**: Yes/No (LED pathfinder next)

---

## Reference Files

- **Existing PTX**: `knowledge3d/cranium/ptx/morton_octree.ptx` (8.4KB, ready to use)
- **RPN Engine**: `knowledge3d/cranium/bridges/sovereign_bridges.py:986-1066`
- **Template**: `SPATIAL_KERNEL_ASSESSMENT.md` Template 2
- **Architecture**: `RPN_SWARM_ARCHITECTURE.md`

---

## Notes

- Morton octree PTX already exists (8.4KB compiled kernel)
- Only need thin Python wrapper + RPN sorting
- This proves RPN can replace heavyweight libraries (CuPy Thrust)
- Sets pattern for Phase C (LED) and Phase D (semantic navigator)

**Proceed when ready!** 🚀
