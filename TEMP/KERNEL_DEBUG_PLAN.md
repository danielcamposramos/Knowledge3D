# RPN Kernel Debug & Fix Plan
## Critical Issue: Opcode Execution Failures

### 🔴 Problem Statement

The extended RPN kernel rejects **standard opcodes** (0x30 DOT, 0x31 NORM) with `kErrorUnknownOpcode`, causing:
- `compute_similarity_matrix_rpn()` → all zeros
- Consolidation cohesion metrics → 0.0
- New clustering opcodes (0xC0-0xC5) → untestable

**Root Cause**: Kernel execution path broken after extension or PTX compilation issue.

---

## Phase 1: Diagnose Kernel State

### Step 1.1: Verify PTX Compilation

**Check current kernel**:
```bash
cd knowledge3d/cranium/ptx/
ls -lh modular_rpn_kernel*.ptx

# Should see:
# modular_rpn_kernel.ptx           (original)
# modular_rpn_kernel_extended.ptx  (new)
```

**Questions**:
1. Which PTX is being loaded by `sovereign_rpn_executor.py`?
2. Does the executor load the **extended** version?
3. Are opcodes 0x30/0x31 present in BOTH kernels?

**Action**:
```python
# In sovereign_rpn_executor.py __init__:
ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel.ptx"
# vs
ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel_extended.ptx"
```

**Verify**: Ensure executor loads the **correct** PTX file.

---

### Step 1.2: Inspect Kernel Opcodes

**Extract opcode list from kernel source**:
```bash
cd knowledge3d/cranium/kernels/

# Original kernel
grep -n "case 0x" modular_rpn_kernel.cu | head -20

# Extended kernel
grep -n "case 0x" modular_rpn_kernel_extended.cu | head -30
```

**Expected in BOTH kernels**:
```cuda
case 0x30:  // DOT product
case 0x31:  // NORM
case 0x32:  // COSINE (if exists)
```

**Expected ONLY in extended kernel**:
```cuda
case 0xC0:  // VEC_L2_NORM
case 0xC1:  // VEC_NORMALIZE
case 0xC2:  // VEC_ARGMAX
case 0xC3:  // VEC_BLEND
case 0xC4:  // COSINE_SIM_BATCH
case 0xC5:  // CLUSTER_ASSIGN
```

**Issue Check**: Did we accidentally break the switch statement when adding new cases?

---

### Step 1.3: Test Original Kernel First

**Hypothesis**: The original `modular_rpn_kernel.ptx` works, but extended version is broken.

**Test script** (`test_original_kernel.py`):
```python
import numpy as np
from pathlib import Path
from knowledge3d.cranium.sovereign import loader

# Load ORIGINAL kernel (not extended)
ptx_path = Path("knowledge3d/cranium/ptx/modular_rpn_kernel.ptx")
module = loader.load_module_from_file(str(ptx_path))
kernel = loader.get_function(module, "modular_rpn_geometric_kernel")

print(f"✓ Loaded original kernel from {ptx_path}")

# Test simple DOT product program
# Opcode: 0x30 (DOT)
# Expected: dot([1,2,3], [4,5,6]) = 32.0

op_codes = np.array([0x30], dtype=np.uint16)  # DOT opcode
scalars = np.array([], dtype=np.float32)
vectors = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)  # Two 3D vectors

# Allocate GPU memory
state_size = 15 * 1040
state_buffer = loader.gpu_malloc(state_size)
zeros = np.zeros(state_size, dtype=np.uint8)
loader.memcpy_htod(state_buffer, zeros.ctypes.data, state_size)

op_codes_gpu = loader.gpu_malloc(op_codes.nbytes)
scalars_gpu = loader.gpu_malloc(max(4, scalars.nbytes))  # At least 4 bytes
vectors_gpu = loader.gpu_malloc(vectors.nbytes)

loader.memcpy_htod(op_codes_gpu, op_codes.ctypes.data, op_codes.nbytes)
loader.memcpy_htod(vectors_gpu, vectors.ctypes.data, vectors.nbytes)

# Launch kernel
import ctypes
loader.launch(
    kernel,
    grid=(1, 1, 1),
    block=(1, 1, 1),
    params=[
        ctypes.c_uint32(0),  # instance_id
        ctypes.c_uint64(op_codes_gpu.value),
        ctypes.c_uint64(scalars_gpu.value),
        ctypes.c_uint64(vectors_gpu.value),
        ctypes.c_uint64(state_buffer.value),
        ctypes.c_uint32(len(op_codes)),
    ],
)
loader.synchronize()

# Read error code from state buffer
# State layout: head(4) + size(4) + error(4) + reserved(4) + stack[...]
error_bytes = np.empty(4, dtype=np.uint8)
error_ptr = loader.CUdeviceptr(state_buffer.value + 8)  # Offset 8 = error field
loader.memcpy_dtoh(error_bytes.ctypes.data, error_ptr, 4)
error_code = np.frombuffer(error_bytes, dtype=np.uint32)[0]

# Read result from stack
result_bytes = np.empty(4, dtype=np.uint8)
result_ptr = loader.CUdeviceptr(state_buffer.value + 16)  # Offset 16 = stack[0]
loader.memcpy_dtoh(result_bytes.ctypes.data, result_ptr, 4)
result = np.frombuffer(result_bytes, dtype=np.float32)[0]

print(f"Error code: {error_code} (0=success, 9001=unknown opcode)")
print(f"Result: {result} (expected: 32.0)")

# Cleanup
loader.gpu_free(op_codes_gpu)
loader.gpu_free(scalars_gpu)
loader.gpu_free(vectors_gpu)
loader.gpu_free(state_buffer)

if error_code == 0 and abs(result - 32.0) < 0.01:
    print("✅ Original kernel WORKS")
else:
    print("❌ Original kernel BROKEN")
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_original_kernel.py
```

**Expected**:
- ✅ Error code: 0
- ✅ Result: 32.0

**If this fails**: Original kernel is also broken → check loader API.

---

## Phase 2: Fix Extended Kernel

### Step 2.1: Verify Opcode Constants

**Check** `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`:

```cuda
// At top of file (in anonymous namespace)
constexpr uint16_t kOpDot = 0x30;          // Must exist
constexpr uint16_t kOpNorm = 0x31;         // Must exist
constexpr uint16_t kOpVecL2Norm = 0xC0;    // New
constexpr uint16_t kOpVecNormalize = 0xC1; // New
constexpr uint16_t kOpVecArgmax = 0xC2;    // New
constexpr uint16_t kOpVecBlend = 0xC3;     // New
constexpr uint16_t kOpCosineSimilarityBatch = 0xC4; // New
constexpr uint16_t kOpClusterAssign = 0xC5; // New
```

**Issue**: Are constants defined but never referenced in switch statement?

---

### Step 2.2: Locate Switch Statement

**Find the main opcode switch**:
```bash
grep -n "switch.*opcode" knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu
```

**Should see**:
```
Line XXX:     switch (opcode) {
```

**Check**: Are ALL opcodes present as cases?

---

### Step 2.3: Common Syntax Errors

**Check for**:

1. **Missing break statements**:
```cuda
case 0xC0: {
    // ... code ...
    break;  // ← MUST HAVE THIS
}
```

2. **Fall-through to default**:
```cuda
case 0xC5: {
    // ... code ...
    break;
}
default:
    error_code = kErrorUnknownOpcode;  // ← This gets triggered if no break
    break;
```

3. **Scope issues**:
```cuda
case 0xC0: {  // ← Need braces for local variables
    TensorRef vec{};
    // ...
    break;
}
```

4. **Duplicate case labels**:
```bash
# Check for duplicates
grep "case 0x" knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu | sort | uniq -d
```

---

### Step 2.4: Recompile Extended Kernel

**Clean rebuild**:
```bash
cd knowledge3d/cranium/kernels/

# Backup old PTX
cp ../ptx/modular_rpn_kernel_extended.ptx ../ptx/modular_rpn_kernel_extended.ptx.backup

# Recompile with verbose errors
nvcc -ptx -arch=sm_86 \
  modular_rpn_kernel_extended.cu \
  -o ../ptx/modular_rpn_kernel_extended.ptx \
  2>&1 | tee compile.log

# Check for errors
grep -i error compile.log
grep -i warning compile.log
```

**Expected**: Zero errors, warnings OK.

---

### Step 2.5: Test Extended Kernel with Simple Opcode

**Test script** (`test_extended_kernel_basic.py`):
```python
import numpy as np
from pathlib import Path
from knowledge3d.cranium.sovereign import loader
import ctypes

# Load EXTENDED kernel
ptx_path = Path("knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx")
module = loader.load_module_from_file(str(ptx_path))
kernel = loader.get_function(module, "modular_rpn_geometric_kernel")

print(f"✓ Loaded extended kernel from {ptx_path}")

# Test 1: Old opcode (0x30 DOT) - should still work
print("\nTest 1: DOT product (0x30)")
# ... (same as original kernel test)

# Test 2: New opcode (0xC0 VEC_L2_NORM)
print("\nTest 2: VEC_L2_NORM (0xC0)")
# Program: Push vector [3,4] → VEC_L2_NORM → expect 5.0
op_codes = np.array([0xC0], dtype=np.uint16)
vectors = np.array([3.0, 4.0], dtype=np.float32)

# ... (similar GPU memory setup)
# ... (launch kernel)
# ... (read result)

print(f"Result: {result} (expected: 5.0)")
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_extended_kernel_basic.py
```

---

## Phase 3: Fix Sovereign RPN Executor

### Step 3.1: Correct PTX Path

**In** `knowledge3d/cranium/sovereign_rpn_executor.py`:

```python
def __init__(self, ptx_path: Optional[Path] = None):
    if ptx_path is None:
        # BEFORE (wrong):
        ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel.ptx"

        # AFTER (correct):
        ptx_path = Path(__file__).parent / "ptx" / "modular_rpn_kernel_extended.ptx"
```

**Reason**: Executor must load extended kernel to access 0xC0-0xC5 opcodes.

---

### Step 3.2: Fix Result Reading

**Current issue** (from Codex report):
```python
# BROKEN (causes segfault):
result_bytes = np.empty(4, dtype=np.uint8)
loader.memcpy_dtoh(result_bytes.ctypes.data, result_ptr, 4)
result = np.frombuffer(result_bytes, dtype=np.float32)[0]
```

**Fixed version**:
```python
# Use ctypes for proper alignment
result_host = ctypes.c_float()
loader.memcpy_dtoh(ctypes.byref(result_host), result_ptr, ctypes.sizeof(result_host))
result = result_host.value
```

**Apply this fix** throughout `execute_single()`.

---

### Step 3.3: Add Debugging Output

**Enhanced executor**:
```python
def execute_single(self, instance_id, op_codes, scalars, vectors):
    # ... (setup code)

    # Launch kernel
    loader.launch(self.kernel, grid=(1,1,1), block=(1,1,1), params=params)
    loader.synchronize()

    # Read error code FIRST
    error_offset = instance_id * self.INSTANCE_STRIDE + 8
    error_host = ctypes.c_uint32()
    error_ptr = loader.CUdeviceptr(self.state_buffer.value + error_offset)
    loader.memcpy_dtoh(ctypes.byref(error_host), error_ptr, ctypes.sizeof(error_host))

    if error_host.value != 0:
        print(f"⚠️  RPN Error: {error_host.value}")
        print(f"   Opcodes: {op_codes.tolist()}")
        if error_host.value == 9001:
            print(f"   → kErrorUnknownOpcode")

    # Then read result
    result_offset = instance_id * self.INSTANCE_STRIDE + 16
    result_host = ctypes.c_float()
    result_ptr = loader.CUdeviceptr(self.state_buffer.value + result_offset)
    loader.memcpy_dtoh(ctypes.byref(result_host), result_ptr, ctypes.sizeof(result_host))

    # ... (cleanup)

    return float(result_host.value)
```

---

## Phase 4: Validate Clustering Operations

### Step 4.1: Test Individual Clustering Opcodes

**Create** `test_clustering_opcodes.py`:
```python
from knowledge3d.cranium.sovereign_rpn_executor import get_sovereign_rpn_executor
import numpy as np

executor = get_sovereign_rpn_executor()

# Test 1: VEC_L2_NORM (0xC0)
print("Test 1: VEC_L2_NORM")
op_codes = np.array([0xC0], dtype=np.uint16)
scalars = np.array([], dtype=np.float32)
vectors = np.array([3.0, 4.0, 0.0], dtype=np.float32)
result = executor.execute_single(0, op_codes, scalars, vectors)
print(f"  Result: {result} (expected: 5.0)")

# Test 2: VEC_ARGMAX (0xC2)
print("\nTest 2: VEC_ARGMAX")
op_codes = np.array([0xC2], dtype=np.uint16)
vectors = np.array([1.0, 5.0, 3.0, 2.0], dtype=np.float32)
result = executor.execute_single(0, op_codes, scalars, vectors)
print(f"  Result: {result} (expected: 1.0)")

# Test 3: VEC_BLEND (0xC3)
print("\nTest 3: VEC_BLEND")
op_codes = np.array([0xC3], dtype=np.uint16)
scalars = np.array([0.5], dtype=np.float32)  # alpha
vectors = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)  # a=[0,0], b=[10,10]
result = executor.execute_single(0, op_codes, scalars, vectors)
print(f"  Result: {result} (expected: blend)")

# Add tests for 0xC4, 0xC5...
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_clustering_opcodes.py
```

---

### Step 4.2: Test Clustering Ops Wrapper

**Once opcodes work**, test high-level API:
```python
from knowledge3d.cranium.sovereign_clustering_ops import SovereignClusteringOps
import numpy as np

ops = SovereignClusteringOps()

# Test normalize_vectors_batch
vectors = np.random.randn(10, 128).astype(np.float32)
normalized = ops.normalize_vectors_batch(vectors)
norms = np.linalg.norm(normalized, axis=1)
print(f"Norms after normalization: {norms}")
assert np.allclose(norms, 1.0), "Vectors not normalized!"

# Test cosine_similarity_matrix
centroids = np.random.randn(5, 128).astype(np.float32)
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
similarities = ops.cosine_similarity_matrix(vectors, centroids)
print(f"Similarity matrix shape: {similarities.shape}")
assert similarities.shape == (10, 5), "Wrong similarity shape!"

# Test assign_to_clusters
assignments = ops.assign_to_clusters(similarities)
print(f"Assignments: {assignments}")
assert len(assignments) == 10, "Wrong assignments length!"
```

---

## Phase 5: Integration Testing

### Step 5.1: Test Sleep Consolidation

**Create** `test_consolidation_sovereign.py`:
```python
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
import numpy as np

# Create test engine with embeddings
engine = RPNEmbeddingEngine(embedding_dim=128)
for i in range(100):
    vec = np.random.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    engine.embeddings[i] = vec

print(f"Created engine with {len(engine.embeddings)} embeddings")

# Create consolidator
consolidator = SleepTimeConsolidator(engine, cluster_count=10)

# Run consolidation
print("\nRunning consolidation...")
metrics = consolidator.consolidate()

print("\nMetrics:")
print(f"  Cluster refinement: {metrics['cluster_refinement']}")
print(f"  Redundancy pruning: {metrics['redundancy_pruning']}")
print(f"  Cohesion before: {metrics['cluster_refinement']['cohesion_before']}")
print(f"  Cohesion after: {metrics['cluster_refinement']['cohesion_after']}")

# Verify cohesion is non-zero
cohesion_before = metrics['cluster_refinement']['cohesion_before']
cohesion_after = metrics['cluster_refinement']['cohesion_after']

assert cohesion_before > 0, "❌ Cohesion before is ZERO - RPN kernel broken!"
assert cohesion_after > 0, "❌ Cohesion after is ZERO - RPN kernel broken!"
print("\n✅ Consolidation produced non-zero cohesion metrics!")
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_consolidation_sovereign.py
```

---

## Phase 6: Phase G Training Integration

### Step 6.1: Update Training Script

**In** `scripts/phase_g_gpu_training_session.py`:

**Remove** any `--skip-consolidation` hacks:
```python
# BEFORE:
if not args.skip_consolidation:
    consolidator = SleepTimeConsolidator(engine)
    metrics = consolidator.consolidate()

# AFTER (always consolidate):
consolidator = SleepTimeConsolidator(engine)
metrics = consolidator.consolidate()
```

---

### Step 6.2: End-to-End Test

**Run full training loop**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/phase_g_gpu_training_session.py \
  --specialists speech \
  --cooldown-seconds 60 \
  --epochs 1
```

**Monitor**:
- ✅ Speech specialist trains
- ✅ Cooldown waits
- ✅ Consolidation runs without segfault
- ✅ Cohesion metrics > 0
- ✅ GPU memory stable

---

## Fallback Strategy

If sovereign executor remains broken after debugging:

### Option A: Hybrid Approach

**Keep CuPy executor for RPN programs**, use sovereign only for new opcodes:

```python
# In sleep_time_consolidator.py
def __post_init__(self):
    # Use CuPy executor for proven ops
    from knowledge3d.cranium.rpn_executor import get_rpn_executor
    self._rpn_executor = get_rpn_executor()  # CuPy-based

    # Use sovereign only for new clustering ops
    from knowledge3d.cranium.sovereign_clustering_ops import SovereignClusteringOps
    self._clustering_ops = SovereignClusteringOps()
```

### Option B: CPU Fallback

**Temporarily disable GPU clustering**:
```python
# In sovereign_clustering_ops.py
def __init__(self, use_gpu=False):  # ← Force CPU for now
    if use_gpu:
        try:
            self.module = loader.load_module_from_file(ptx_path)
        except Exception:
            print("⚠️  GPU clustering unavailable, using CPU")
            use_gpu = False
    self.use_gpu = use_gpu
```

---

## Success Criteria Checklist

### Immediate (Kernel Fix)
- [ ] Original kernel (modular_rpn_kernel.ptx) executes DOT (0x30) successfully
- [ ] Extended kernel (modular_rpn_kernel_extended.ptx) executes DOT (0x30) successfully
- [ ] Extended kernel executes VEC_L2_NORM (0xC0) successfully
- [ ] Extended kernel executes all clustering opcodes (0xC0-0xC5)

### Short-term (Executor)
- [ ] `SovereignRPNExecutor` initializes without segfault
- [ ] Can execute single RPN program and get correct result
- [ ] Can execute batch programs across multiple instances
- [ ] Error codes properly reported

### Integration
- [ ] `compute_similarity_matrix_rpn()` produces non-zero similarities
- [ ] Consolidation cohesion metrics > 0
- [ ] Full Phase G training loop works with consolidation
- [ ] GPU memory stable across multiple consolidation cycles

---

## Debug Workflow Summary

```
1. Test original kernel (0x30 DOT)
   ✅ Works → Problem is in extended kernel
   ❌ Fails → Problem is in loader/infrastructure

2. Fix extended kernel
   - Check switch statement completeness
   - Verify all opcodes have cases
   - Recompile PTX
   - Test basic opcodes

3. Fix sovereign executor
   - Load correct PTX (extended version)
   - Fix ctypes pointer handling
   - Add error code debugging

4. Test clustering opcodes individually
   - 0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5
   - Verify each opcode works in isolation

5. Test high-level clustering ops
   - normalize_vectors_batch
   - cosine_similarity_matrix
   - assign_to_clusters

6. Test full consolidation pipeline
   - Non-zero cohesion metrics
   - Embeddings actually updated

7. Integrate with Phase G training
   - End-to-end test
   - Memory stability
```

---

**Priority Order**:
1. Test original kernel → establishes baseline
2. Fix extended kernel compilation/opcodes → unlocks new functionality
3. Fix sovereign executor → enables CuPy-free execution
4. Integration testing → proves full pipeline

**Estimated Time**: 2-4 hours if kernel compilation is clean, 8+ hours if deep debugging needed.

---

**Next Command**:
```bash
# Start with baseline test
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  test_original_kernel.py
```
