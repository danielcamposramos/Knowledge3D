# Phase 2 Purge Plan — Top 5 Sovereign Violations (Hot Path)

**Date**: 2026-04-18  
**Scope**: Identify and map concrete PTX/RPN replacements for highest-density violations in sovereign hot paths  
**Files**: 5 critical offenders (3,881 violations / 127 files total, see bulk_lib_audit_04.18.2026.md)

---

## File 1: procedural_compiler.py (198 violations, CRITICAL)

**Path**: `knowledge3d/cranium/procedural_compiler.py`  
**Primary Lib**: numpy (198 refs)  
**Secondary Libs**: cupy  
**Severity**: CRITICAL — RPN opcode compilation, embedding initialization  

### Violations Map

| Line Range | Code Pattern | Banned Lib | Replacement Strategy |
|-----------|--------------|-----------|----------------------|
| ~42 | `np.random.randn(...)` opcode embedding init | numpy | Use `matryoshka_prefix_dot(opcode, dim)` deterministic seeding OR pre-computed constant table (store in Galaxy as static entry) |
| ~167 | `np.random.randn(256, dim)` opcode table | numpy | Same as above; load from PTX kernel `rng_pool.ptx` at kernel init (see `knowledge3d/cranium/utils/rng_pool.py`) |
| ~200–250 | `np.array(...)`, `np.zeros(...)` dtype construction | numpy | Replace with ctypes.Structure (scalar uint32/float32 native types); no ndarray wrapper |
| ~300+ | `np.linspace(...)` numeric table generation | numpy | Use PTX grid generation kernel (OP_LINSPACE equivalent in RPN opcode registry) OR pre-computed lookup table |

### Concrete Replacements

**Option A (Deterministic Seed via Matryoshka):**
```python
# BEFORE (FORBIDDEN)
opcode_embeddings = np.random.randn(256, self.matryoshka_dim).astype(np.float32) * 0.01

# AFTER (SOVEREIGN)
# Pre-compute seed table from opcode ID via matryoshka_prefix_dot (deterministic, GPU-friendly)
seed_table = ptx_loader.launch_kernel(
    'compute_opcode_seeds',
    grid=(256,),
    block=(32,),
    args=(self.matryoshka_dim,)
)
# Load into Galaxy as static entry: opcode_embedding_table@rng_pool
self.opcode_embeddings = seed_table  # ctypes.c_float array, GPU-resident
```

**Option B (PTX RNG Pool at Load Time):**
```python
# Use existing rng_pool.ptx kernel
kernel = self.ptx_loader.load('rng_pool.ptx')
self.opcode_embeddings = kernel.launch(
    grid=(256,),
    block=(32,),
    args=(self.matryoshka_dim, seed_from_opcode_id)
)
```

### PTX Kernels to Implement / Verify

- `rng_pool.ptx` — existing, verify it supports deterministic seeding
- `compute_opcode_seeds.cu` — new, small kernel (128 regs max) computing matryoshka hash for each opcode
- `linspace_kernel.cu` — existing or new, replaces np.linspace in table generation

### Success Criteria

- Zero numpy imports in procedural_compiler.py
- Opcode embedding table deterministic across runs (same seed → same table)
- No malloc/memcpy per compile; all tables pre-loaded at boot
- Frame latency regression: < 0.1ms (measure via perf_event_open)

---

## File 2: procedural_drawing_specialist.py (127 violations, CRITICAL)

**Path**: `knowledge3d/cranium/specialists/procedural_drawing_specialist.py`  
**Primary Lib**: numpy (127 refs)  
**Secondary Libs**: cupy  
**Severity**: CRITICAL — Matryoshka embedding, semantic encoding  

### Violations Map

| Line | Code Pattern | Banned Lib | Replacement Strategy |
|------|--------------|-----------|----------------------|
| 167 | `np.random.randn(256, matryoshka_dim)` | numpy | Pre-compute via Matryoshka + Galaxy static entry OR PTX RNG kernel |
| 214–220 | `np.zeros(n, dtype=...)`, `np.array(codes)` | numpy | Replace with ctypes.Structure + byte-packed struct, bit-field extraction in PTX |
| 95–110 | `np.einsum(...)` opcode embedding projection | numpy | Use PTX `matryoshka_prefix_dot` kernel (existing) or `OP_OUTER_PRODUCT` RPN opcode |
| 250+ | `np.norm(...)`, `np.dot(...)` normalization | numpy | Replace with `OP_VEC_L2_NORM`, `OP_DOT_PRODUCT` from RPN opcode registry; execute via PTX |

### Concrete Replacements

**Matryoshka Embedding Table:**
```python
# BEFORE
self.matryoshka_embeddings = np.random.randn(256, matryoshka_dim).astype(np.float32) * 0.01

# AFTER
# Pre-computed via matryoshka_prefix_dot (deterministic)
self.matryoshka_embeddings = ptx_loader.launch_kernel(
    'matryoshka_seed_table',
    grid=(256,),
    block=(32,),
    args=(matryoshka_dim,)
)
```

**Semantic Encoding (np.zeros + np.array):**
```python
# BEFORE
codes = np.zeros(n_codes, dtype=np.uint32)
for i, code in enumerate(codes_list):
    codes[i] = code

# AFTER
# Use ctypes.Structure with native uint32 array
import ctypes
CodesBuffer = (ctypes.c_uint32 * n_codes)
codes = CodesBuffer()
for i, code in enumerate(codes_list):
    codes[i] = code
# Bit-field extraction in PTX kernel
encoded = ptx_loader.launch_kernel(
    'encode_semantic_bits',
    args=(codes, n_codes, matryoshka_dim)
)
```

**Vector Normalization (np.norm + np.dot):**
```python
# BEFORE
norm = np.linalg.norm(vector)
normalized = vector / (norm + 1e-8)

# AFTER
# Use RPN opcode execution via PTX
norm = rpn_interpreter.execute(
    [vector, 'OP_VEC_L2_NORM'],
    device='gpu'
)
normalized = rpn_interpreter.execute(
    [vector, norm, 1e-8, 'OP_ADD', 'OP_VEC_DIVIDE'],
    device='gpu'
)
```

### PTX Kernels to Implement / Verify

- `matryoshka_seed_table.cu` — deterministic seeding for 256-dim embedding table
- `encode_semantic_bits.cu` — bit-field extraction from byte-packed codes
- Verify existing `rpn_interpreter.py` / `rpn_opcodes.py` includes `OP_VEC_L2_NORM`, `OP_DOT_PRODUCT`, `OP_VEC_DIVIDE`

### Success Criteria

- Zero numpy imports in procedural_drawing_specialist.py
- Embedding projection deterministic (same seed → same table)
- Bit-field encoding matches original np.array semantics
- GPU kernel latency: < 1ms per call
- Regression test: drawing specialist produces identical output (within float32 epsilon)

---

## File 3: knowledgeverse.py (136 violations, CRITICAL)

**Path**: `knowledge3d/knowledgeverse/knowledgeverse.py`  
**Primary Lib**: numpy (136 refs), torch (17 refs)  
**Severity**: CRITICAL — VRAM substrate initialization, Galaxy state  

### Violations Map

| Line Range | Code Pattern | Banned Lib | Replacement Strategy |
|-----------|--------------|-----------|----------------------|
| 50–80 | `np.zeros(n_nodes, dtype=...)`, `np.array(...)` galaxy state init | numpy | Migrate to GPU-resident ctypes.Structure + CUDA malloc; eliminate CPU-GPU memcpy |
| 100+ | `np.concatenate(...)`, `np.stack(...)` state merging | numpy | Use PTX kernel for GPU-side concatenation (avoid memcpy) |
| 150+ | `torch.as_tensor(...)` model weight loading | torch | Replace with ctypes array + GLB loader (already in `glb_weights.py`); defer model init to PTX bridge |
| 200+ | `np.where(...)`, `np.masked_array(...)` conditional logic | numpy | Use PTX conditional kernels OR RPN `OP_BRANCH` opcode executed via GPU interpreter |

### Concrete Replacements

**Galaxy State Initialization (np.zeros → ctypes + CUDA malloc):**
```python
# BEFORE (FORBIDDEN)
self.galaxy_nodes = np.zeros(n_nodes, dtype=np.float32)
self.galaxy_edges = np.zeros((n_nodes, n_neighbors), dtype=np.int32)

# AFTER (SOVEREIGN)
import ctypes

class GalaxyNodeState(ctypes.Structure):
    _fields_ = [
        ('node_id', ctypes.c_uint32),
        ('embedding', (ctypes.c_float * embedding_dim)),
        ('confidence', ctypes.c_float),
        ('polarity', ctypes.c_int8),
    ]

# Allocate on GPU
n_nodes_bytes = n_nodes * ctypes.sizeof(GalaxyNodeState)
self.galaxy_nodes_gpu = cuda.malloc(n_nodes_bytes)  # cupy fallback only if cupy_env is enabled

# Load initial state via PTX kernel (GPU-side init, zero memcpy)
kernel = ptx_loader.load('galaxy_state_init.cu')
kernel.launch(
    grid=(n_nodes // 256,),
    block=(256,),
    args=(self.galaxy_nodes_gpu, n_nodes, embedding_dim)
)
```

**State Merging (np.concatenate → GPU kernel):**
```python
# BEFORE
merged_state = np.concatenate([state_a, state_b], axis=0)

# AFTER
# GPU-side concatenation kernel
kernel = ptx_loader.load('gpu_concatenate.cu')
merged_state_gpu = kernel.launch(
    grid=(...),
    args=(state_a_gpu, state_b_gpu, merged_state_gpu, size_a, size_b)
)
```

**Conditional Logic (np.where → PTX or RPN):**
```python
# BEFORE
result = np.where(confidence > threshold, galaxy_nodes, fallback_value)

# AFTER (Option A: PTX kernel)
result_gpu = ptx_loader.launch_kernel(
    'conditional_select',
    args=(confidence_gpu, galaxy_nodes_gpu, fallback_gpu, threshold)
)

# AFTER (Option B: RPN opcode)
result_gpu = rpn_interpreter.execute(
    [confidence_gpu, threshold, 'OP_GT', galaxy_nodes_gpu, fallback_gpu, 'OP_SELECT'],
    device='gpu'
)
```

### PTX Kernels to Implement / Verify

- `galaxy_state_init.cu` — initialize GPU-resident GalaxyNodeState structures
- `gpu_concatenate.cu` — GPU-side array concatenation (no CPU memcpy)
- `conditional_select.cu` — mask-based selection (replace np.where)
- Verify RPN opcode registry has `OP_GT`, `OP_SELECT`, `OP_BRANCH`

### Success Criteria

- Zero numpy.zeros / np.concatenate calls in knowledgeverse.py
- All galaxy state initialized on GPU (no CPU memcpy during frame)
- ctypes.Structure matches PTX kernel layout (byte alignment verified)
- Frame latency: < 5ms per trm_step (measure before/after)
- Memory regression: no leaks (nvidia-smi memory tracking)

---

## File 4: semantic_csr_graph.py (39 violations, HIGH)

**Path**: `knowledge3d/knowledgeverse/semantic_csr_graph.py`  
**Primary Lib**: numpy (39 refs)  
**Severity**: HIGH — CSR sparse matrix in query graph navigation  

### Violations Map

| Line | Code Pattern | Banned Lib | Replacement Strategy |
|------|--------------|-----------|----------------------|
| ~20–40 | `np.zeros(...)`, `np.array(...)` CSR row/col/data | numpy | Pre-compute CSR format during ingestion; load as static binary blob (no runtime allocation) |
| ~60–80 | `csr_matrix.dot(...)` sparse matrix multiply | numpy | Use PTX SpMV kernel (sparse matrix-vector multiply) OR RPN opcode `OP_SPARSE_MATVEC` |
| ~100+ | `np.argsort(...)`, `np.cumsum(...)` sorting | numpy | Use PTX sorting kernel (CUB-based) for edge sorting |

### Concrete Replacements

**CSR Matrix Format (Pre-compute → Binary Blob):**
```python
# BEFORE (runtime numpy)
self.csr_data = np.array(edges, dtype=np.float32)
self.csr_indices = np.array(indices, dtype=np.int32)
self.csr_indptr = np.array(indptr, dtype=np.int32)

# AFTER (pre-computed + binary load)
# Pre-compute during ingestion (build_galaxy.py):
# csr_data.bin, csr_indices.bin, csr_indptr.bin written to Galaxy
# Load at boot:
self.csr_data_gpu = ptx_loader.load_binary('csr_data.bin', dtype=np.float32)
self.csr_indices_gpu = ptx_loader.load_binary('csr_indices.bin', dtype=np.int32)
self.csr_indptr_gpu = ptx_loader.load_binary('csr_indptr.bin', dtype=np.int32)
```

**Sparse Matrix-Vector Multiply (SpMV):**
```python
# BEFORE
result = self.csr_matrix.dot(vector)

# AFTER (PTX SpMV kernel)
kernel = ptx_loader.load('sparse_matvec.cu')
result_gpu = kernel.launch(
    grid=(...),
    args=(self.csr_data_gpu, self.csr_indices_gpu, self.csr_indptr_gpu,
          vector_gpu, n_rows)
)

# OR (RPN opcode if available)
result_gpu = rpn_interpreter.execute(
    [csr_matrix_gpu, vector_gpu, 'OP_SPARSE_MATVEC'],
    device='gpu'
)
```

**Sorting (np.argsort → PTX CUB-based):**
```python
# BEFORE
sorted_indices = np.argsort(edge_weights)

# AFTER
kernel = ptx_loader.load('sort_indices.cu')  # CUB-based
sorted_indices_gpu = kernel.launch(
    grid=(...),
    args=(edge_weights_gpu, n_edges)
)
```

### PTX Kernels to Implement / Verify

- `sparse_matvec.cu` — CSR SpMV (may already exist in CUTLASS or CUB)
- `sort_indices.cu` — GPU sorting wrapper around CUB or custom radix sort
- Pre-compute CSR format offline in `build_galaxy.py`

### Success Criteria

- Zero runtime np.array() for CSR construction
- CSR matrix pre-loaded from binary at boot
- SpMV latency: < 10ms per query (measure on 1M-edge graph)
- Regression test: query results identical to numpy baseline (within float32 epsilon)

---

## File 5: confidence_propagation.py (Torch in decision branching, CRITICAL)

**Path**: `knowledge3d/cranium/actions/confidence_propagation.py`  
**Primary Lib**: torch  
**Severity**: CRITICAL — Inference-time decision branching  

### Violations Map

| Line Range | Code Pattern | Banned Lib | Replacement Strategy |
|-----------|--------------|-----------|----------------------|
| 56–90 | `torch.as_tensor(...device="cuda")` confidence tensor fusion | torch | Replace with ctypes array + PTX fuse kernel OR pre-computed confidence lookup table |
| 120+ | `torch.nn.functional.softmax(...)` decision normalization | torch | Use RPN opcode `OP_SOFTMAX` OR custom PTX kernel for stable softmax |
| 180+ | `torch.dot(...)`, `torch.sum(...)` aggregation | torch | Use RPN opcodes `OP_DOT_PRODUCT`, `OP_SUM` executed via GPU interpreter |

### Concrete Replacements

**Confidence Tensor Fusion (torch.as_tensor → ctypes + PTX):**
```python
# BEFORE (FORBIDDEN — inference-time torch on GPU)
confidence = torch.as_tensor(confidence_array, device='cuda:0').float()
fused = torch.nn.functional.softmax(confidence, dim=0)

# AFTER (SOVEREIGN)
import ctypes

# Load confidence as ctypes float array
class ConfidenceBuffer(ctypes.Structure):
    _fields_ = [
        ('values', (ctypes.c_float * n_nodes)),
        ('length', ctypes.c_int),
    ]

confidence_gpu = ConfidenceBuffer()
confidence_gpu.values[:] = confidence_array
confidence_gpu.length = n_nodes

# Fuse via PTX kernel
kernel = ptx_loader.load('confidence_softmax_fuse.cu')
fused_gpu = kernel.launch(
    grid=(256,),
    block=(256,),
    args=(confidence_gpu, n_nodes)
)
```

**Softmax (torch.nn.functional.softmax → PTX or RPN):**
```python
# AFTER (PTX kernel for numerical stability)
kernel = ptx_loader.load('stable_softmax.cu')
fused = kernel.launch(
    grid=(n_nodes // 256,),
    block=(256,),
    args=(confidence_gpu, n_nodes)
)

# OR (RPN opcode if available)
fused = rpn_interpreter.execute(
    [confidence_array_gpu, 'OP_SOFTMAX'],
    device='gpu'
)
```

**Aggregation (torch.dot, torch.sum → RPN or PTX):**
```python
# BEFORE
total_confidence = torch.sum(confidence * weights)

# AFTER (RPN)
total_confidence = rpn_interpreter.execute(
    [confidence_gpu, weights_gpu, 'OP_MULTIPLY',  # element-wise
     'OP_SUM'],  # reduction
    device='gpu'
)

# OR (PTX kernel)
kernel = ptx_loader.load('weighted_sum_reduce.cu')
total_confidence = kernel.launch(
    grid=(256,),
    block=(256,),
    args=(confidence_gpu, weights_gpu, n_nodes)
)
```

### PTX Kernels to Implement / Verify

- `confidence_softmax_fuse.cu` — stable softmax (e.g., log-sum-exp trick)
- `stable_softmax.cu` — numerically stable softmax for large arrays
- `weighted_sum_reduce.cu` — reduction kernel for confidence aggregation
- Verify RPN opcode registry has `OP_SOFTMAX`, `OP_MULTIPLY`, `OP_SUM`

### Success Criteria

- Zero torch imports in confidence_propagation.py
- Softmax numerically identical to PyTorch (within float32 epsilon: 1e-6)
- Decision branching latency: < 2ms per query
- Regression test: halting gate convergence unchanged (same decision distribution)

---

## Summary: Implementation Roadmap

### Phase 2A (P0 — Week 1)
1. **procedural_compiler.py**: Implement matryoshka seeding kernel + constant table loading
2. **knowledgeverse.py**: Migrate to GPU-resident ctypes + CUDA malloc

### Phase 2B (P0 — Week 2)
3. **procedural_drawing_specialist.py**: Implement opcode embedding + semantic bit encoding
4. **confidence_propagation.py**: Replace torch with ctypes + stable softmax kernel

### Phase 2C (P1 — Week 3)
5. **semantic_csr_graph.py**: Pre-compute CSR format + load binary blobs at boot

### Dependencies
- PTX RNG pool + matryoshka kernel (check if `rng_pool.py` exists)
- RPN opcode registry verification (`docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`)
- CUB integration for SpMV + sorting (may already be in cranium/ptx/)

### Validation
- Sovereignty audit re-run: target < 100 violations remaining (vs. 3,881)
- Regression benchmarks: ARC 10/10, Math 20/20 unchanged
- Frame latency: < 20ms per trm_step (current baseline)
- Memory: no VRAM leaks (5× 1-hour runs)

---

**Deliverable**: Implementation specs for Codex per file. Ready for lane dispatch.

**Ready for Implementation**: YES (all 5 files have concrete replacement mappings)

**Codex Handoff Status**: READY (cite this plan in implementation tickets)

---

*Generated by: Execution Lane (Agent)*  
*Approval Status*: Ready for Daniel review before Codex implementation dispatch
