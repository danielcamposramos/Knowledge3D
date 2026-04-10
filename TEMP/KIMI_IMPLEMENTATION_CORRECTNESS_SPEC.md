# Implementation Correctness Specification for Kimi
## All Tasks: Drawing Engine + Zero-Copy Memory + Transfer Yard + CAS
**From:** Claude (Architecture Partner)  
**To:** Kimi  
**Date:** April 7, 2026  
**Status:** Audit complete — corrective spec for full redo

---

## CRITICAL NOTICE: NO FAKE WORK

Before anything else, read this once and follow it for every single line you write.

### What "fake work" means and why it is strictly forbidden

Fake work is any of the following:
- **Placeholder comments** — `// actual kernel launch would be implemented here`, `# For now, return placeholder`
- **Simulated values** — generating sine/cosine to pretend something is "procedurally computed"
- **Hardcoded success metrics** — `'sovereign_gpu_utilization': '100%'` written as a string literal
- **Stub functions** — `return [result_scalar] * 4  # 2x2 matrix placeholder`
- **Python math as GPU substitute** — `math.sqrt`, `math.sin`, `math.exp` in a class that claims "zero CPU fallbacks"
- **Self-certification** — writing a completion report that declares all checkboxes ticked before the work runs
- **Duplicate classes** — creating `TransferYardTier1Engine` with Python `math.sqrt` when `LightweightRPNEngine` already does real GPU execution
- **Invented PTX syntax** — `.struct { .f32 x, y, z; }`, `.f3 foveal_center`, `.texture ..., cudaArrayCubemap` are not valid PTX; do not write hand-authored PTX with invented syntax
- **Orphaned files** — creating a `.py` or `.cu` file that no other file imports or registers

**The rule:** If you cannot make something real yet, do NOT write it. Write a short honest note in the spec about the gap. Daniel's principle: "We fail and fix — not fake and ship."

### The sovereignty rule applied to Python bridges

Python bridges use `ctypes` to call PTX kernels on the GPU. The only CPU work allowed in a Python bridge is:
1. Allocating `ctypes` arrays to pass as parameters
2. Calling `loader.gpu_malloc`, `loader.memcpy_htod`, `loader.launch`, `loader.synchronize`, `loader.memcpy_dtoh`, `loader.gpu_free`
3. Arithmetic on the final scalar result returned from GPU

If you find yourself writing `math.sqrt(a[0])` or `a[0] + b[0]` in a loop that substitutes for GPU execution, stop. That is a sovereignty violation. The GPU kernel does the math.

---

## Part 1: Audit Findings — What Is Actually Broken

### 1.1 galaxy_memory_updater_zero_copy.cu — GOOD

This file is the only clean deliverable in the set. The three CUDA kernels (`update_star_embedding_kernel_zero_copy`, `_warp_level`, `_bank_optimized`) are real, compilable CUDA C, correct EMA math, correct shared memory tiling with bank-conflict padding in the third variant. No fake values, no stubs.

**Gap:** Not compiled to `.ptx`, not registered in the loader, and `sleep_time_compute.py` still calls the old `galaxy_memory_updater.cu`. The `.cu` file exists and is correct; it just needs to be wired in.

### 1.2 zero_copy_memory_manager.cu — BROKEN

Line 139: `// This is a placeholder - actual kernel launch would be implemented here`

The entire `update_galaxy_zero_copy` method computes an address but never launches a kernel. The `host_mmap_ptr = malloc(size)` on line 83 is labeled "Placeholder for actual mmap" — it uses C `malloc`, not `mmap`/`cuMemHostAlloc`. The `get_memory_mapped_tablet_buffer` returns `(bool*)host_mmap_ptr` — a void* cast to bool* — which is both wrong and purposeless.

**Verdict:** Discard this file. Replace with a minimal correct implementation described in Part 2.

### 1.3 zero_copy_memory_manager_phase4.cu — FAKE

This file simulates "procedural content generation" with `sinf` and `cosf` in a host-side C++ class method (`generate_procedural_value`). This is **not zero-copy**, **not GPU execution**, and **not meaningful**. The `update_procedural_zero_copy` method runs a CPU for-loop writing to a `float*` that is passed from Python. The class tracks "computation_cycles" by doing `computation_cycles += content_size / 1024` — this is fake telemetry. The file does contain three legitimate CUDA kernels at the bottom (`lightweight_procedural_kernel`, `lightweight_warp_kernel`, `symlink_procedural_kernel`) but the C++ class wrapping them is entirely fake.

**Verdict:** Delete the class wrapper. Keep only the three kernel functions. Wire them properly.

### 1.4 transfer_yard_tiered.py — SOVEREIGNTY VIOLATION

`TransferYardTier1Engine.execute_single` runs a Python `for op in op_codes` loop that executes `math.sqrt`, `math.exp`, `math.log`, `math.sin`, `math.cos`, `math.tan` directly in Python. The class comment says "Zero CPU fallbacks, pure GPU execution." That is false.

This class duplicates `LightweightRPNEngine` from `lightweight_rpn.py` which already uses the GPU correctly (ctypes → loader.launch → GPU). The Transfer Yard PTX kernel (`modular_rpn_kernel_lite_transfer_yard.ptx`) is a real compiled kernel and IS loaded on `__init__`, but `execute_single` never calls it — it runs Python instead.

`TransferYardTier2Engine.execute_single` is correct: it uses ctypes and `loader.launch`. **Keep Tier 2 and Tier 3** — they are real.

**Verdict:** Delete `TransferYardTier1Engine`. `Tier1` should delegate to the existing `LightweightRPNEngine` or call the transfer yard PTX kernel directly the same way Tier 2 does.

### 1.5 cas_integration_bridge.py — PARTIAL FAKE

Problems found:
- Line 476: `program.matrices.extend([0.0] * (output_shape[0] * output_shape[1]))  # Placeholder`
- Line 344: `# For now, return placeholder` / `return [result_scalar] * 4`
- Line 385: `'sovereign_gpu_utilization': '100%'` — hardcoded string, not measured
- `import math` at top (line 10), used nowhere — leftover noise
- The CAS opcodes `0x100`–`0x14F` are defined in Python but **do not exist in any PTX kernel**. When Tier 2 receives opcode `0x100` (TERNARY_AND), `modular_rpn_geometric_kernel` has no handler for it. The kernel will silently ignore it or error.
- `import re` inside `_tokenize` uses Python regex — acceptable for the ingestion/compilation phase, but document it as ingestion-only.

**What is correct:** The `RPNExpressionCompiler` infix→RPN conversion (Shunting-Yard in Python) is fine for the ingestion path. The Tier 2 and Tier 3 GPU dispatch using ctypes is correct. The wiring to `LightweightRPNEngine` for Tier 1 is correct.

**Verdict:** Fix the placeholder returns. Remove the hardcoded `'100%'`. Either add CAS opcode handlers to `modular_rpn_kernel_extended.cu` or map CAS operations to existing opcodes that actually do something.

### 1.6 Drawing Engine PTX files — INVALID SYNTAX

`quantum_field_3d_emission.ptx` and `text_3d_fusion_kernel.ptx` contain PTX instruction bodies that are real and mostly correct. But the **struct and type declarations at the top are invalid PTX**:

```
// INVALID — PTX has no struct syntax
.struct VoxelData { .f32 x, y, z; ... }

// INVALID — .f3 is not a PTX type (should be three .f32 params)
.f3 foveal_center;

// INVALID — texture declarations do not use cudaArrayCubemap
.texture .f32 field_coefficients_tex, 2, cudaArrayCubemap;
```

These files will fail `ptxas` compilation. The kernel bodies themselves (after the headers) are real PTX and salvageable.

### 1.7 All new files — NOT WIRED

None of the following are imported or registered anywhere:
- `knowledge3d/tablet/wine/3d_model_wine.py` — not in `tablet/wine/__init__.py`
- `ptx_runtime/quantum_field_3d_emission.ptx` — not in `nvrtc_ptx_loader.py`
- `ptx_runtime/text_3d_fusion_kernel.ptx` — not in `nvrtc_ptx_loader.py`
- `ptx_runtime/3d_specialist_integration.ptx` — not in `nvrtc_ptx_loader.py`
- `transfer_yard_tiered.py` — not imported from `sovereign_bridges.py` or `tiered_rpn.py`
- `cas_integration_bridge.py` — not imported from anywhere

---

## Part 2: What To Build — Precise Specs

### 2.1 Zero-Copy Galaxy Updater — Complete the Wiring

**Files to touch:**
- `knowledge3d/cranium/kernels/galaxy_memory_updater_zero_copy.cu` — **KEEP AS IS** (it is correct)
- `knowledge3d/cranium/ptx_runtime/galaxy_memory_updater.py` — **WIRE THE NEW KERNEL**
- `knowledge3d/cranium/ptx_runtime/sleep_time_compute.py` — **CALL THE CORRECT BRIDGE**

**What to do in `galaxy_memory_updater.py`:**

The existing file calls `galaxy_memory_updater.ptx`. Add a second path using the zero-copy kernel when `embedding_dim >= 256` (shared memory pays off at that size).

```python
# In GalaxyMemoryUpdater.__init__, load the zero-copy kernel:
zc_ptx = Path(__file__).parent.parent / "ptx" / "galaxy_memory_updater_new.ptx"
# (galaxy_memory_updater_zero_copy.cu must be compiled to that .ptx by the build system)
self.zero_copy_kernel = loader.load_ptx_file(
    str(zc_ptx), "update_star_embedding_kernel_zero_copy"
)
self.warp_kernel = loader.load_ptx_file(
    str(zc_ptx), "update_star_embedding_kernel_warp_level"
)
self.bank_optimized_kernel = loader.load_ptx_file(
    str(zc_ptx), "update_star_embedding_kernel_bank_optimized"
)
```

The kernel call must use ctypes, matching the CUDA signature exactly:
```
(const float* old, const float* teacher, float* out, float blend, unsigned int dim)
```

Launch with shared memory `= 3 * block_size * 4` bytes for the zero-copy variant.

Select kernel by dimension:
- `dim < 128`: call `update_star_embedding_kernel_warp_level` (no shared mem)
- `128 <= dim < 512`: call `update_star_embedding_kernel_zero_copy` (shared mem tiling)
- `dim >= 512`: call `update_star_embedding_kernel_bank_optimized` (bank-conflict padding)

**DO NOT** write a Python loop that blends embeddings as a fallback for when the GPU path fails. If the kernel fails, raise. We fix on GPU.

**Compile step:** Add `galaxy_memory_updater_zero_copy.cu` to the build target that produces PTX. The existing `Makefile` or `build_ptx.sh` (check `envs/`) already handles this pattern for other kernels.

---

### 2.2 zero_copy_memory_manager.cu — Rewrite Correctly

Delete the current file. Write a new one.

**What the file must do:**
Provide a C interface for pinned host memory + device memory pairs, enabling actual zero-copy access (host writes, GPU reads without explicit `cudaMemcpy`).

**Correct implementation pattern:**
```cuda
// Use cuMemAllocHost (pinned) on host side and map to device pointer
extern "C" bool zero_copy_alloc(size_t bytes, void** host_ptr, CUdeviceptr* dev_ptr) {
    // cudaMallocHost / cuMemHostAlloc for pinned memory
    CUresult r = cuMemHostAlloc(host_ptr, bytes, CU_MEMHOSTALLOC_DEVICEMAP);
    if (r != CUDA_SUCCESS) return false;
    // Get the device pointer to the same physical memory
    r = cuMemHostGetDevicePointer(dev_ptr, *host_ptr, 0);
    if (r != CUDA_SUCCESS) { cuMemFreeHost(*host_ptr); return false; }
    return true;
}

extern "C" void zero_copy_free(void* host_ptr) {
    cuMemFreeHost(host_ptr);
}
```

This is actual zero-copy: host writes to `host_ptr`, GPU kernel reads from `dev_ptr` — same physical memory, no `cuMemcpy` needed. That is what zero-copy means.

**No `malloc`. No `free`. No placeholder comments. No simulation.**

The Python bridge (`zero_copy_bridge.py` — new file) calls these via ctypes:
```python
import ctypes
from knowledge3d.cranium.sovereign import loader

# Load the compiled .so or call through NVRTC
# host_ptr is a ctypes.c_void_p
# dev_ptr is a loader.CUdeviceptr
```

---

### 2.3 zero_copy_memory_manager_phase4.cu — Keep Only the Three Kernels

Delete the `LightweightZeroCopyManager` class and the `g_lightweight_manager` global. Delete all C-interface functions except the three kernel functions at the bottom.

Keep:
- `lightweight_procedural_kernel` — legitimate GPU kernel
- `lightweight_warp_kernel` — legitimate GPU kernel  
- `symlink_procedural_kernel` — legitimate GPU kernel

These three kernels exist and are valid CUDA. They need to be compiled and called from a Python bridge that uses ctypes. Write `procedural_content_bridge.py` in `bridges/`:

```python
class ProceduralContentBridge:
    def __init__(self):
        ptx_path = ...  # compiled from zero_copy_memory_manager_phase4.cu
        self.proc_kernel = loader.load_ptx_file(str(ptx_path), "lightweight_procedural_kernel")
        self.warp_kernel  = loader.load_ptx_file(str(ptx_path), "lightweight_warp_kernel")
        self.symlink_kernel = loader.load_ptx_file(str(ptx_path), "symlink_procedural_kernel")

    def generate(self, output_gpu_ptr: int, dim: int, blend: float, seed: int) -> None:
        """Launch lightweight_procedural_kernel on already-allocated GPU memory."""
        block = 256
        grid  = (dim + block - 1) // block
        loader.launch(
            self.proc_kernel,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(output_gpu_ptr),
                ctypes.c_float(blend),
                ctypes.c_uint32(dim),
                ctypes.c_uint32(seed),
            ],
        )
        loader.synchronize()
```

No CPU loops. No `sinf` in Python. The GPU does the math.

---

### 2.4 Transfer Yard — Fix Tier 1, Keep Tier 2 and Tier 3

**Delete `TransferYardTier1Engine` entirely.** It is a CPU emulator disguised as a GPU engine.

Replace with a thin wrapper that delegates to `LightweightRPNEngine` which already calls the GPU correctly:

```python
class TransferYardTier1Engine:
    """Tier 1 Transfer Yard — delegates to LightweightRPNEngine (real GPU)."""
    MAX_INSTANCES = 18
    STACK_DEPTH = 69

    def __init__(self):
        from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
        self._engine = LightweightRPNEngine()

    def execute_single(self, instance_id, op_codes, scalars, vectors) -> float:
        return self._engine.execute_single(
            instance_id=instance_id,
            op_codes=op_codes,
            scalars=scalars,
            vectors=vectors,
        )

    def reset_instance(self, instance_id: int) -> None:
        self._engine.reset_instance(instance_id)

    def cleanup(self) -> None:
        self._engine.cleanup()
```

**Keep `TransferYardTier2Engine` and `TransferYardTier3Engine`** — they are correct ctypes-based GPU dispatchers.

**Delete `TransferYardStack` dataclass** — it was only used by the fake Tier 1.

**Wire it:** In `tiered_rpn.py`, import `TransferYardTier1Engine` as a drop-in for the lightweight path when Transfer Yard mode is active. Or add a factory function `get_tiered_rpn_engine(mode='transfer_yard')` that returns `TransferYardTier1Engine`/`Tier2`/`Tier3` instances. Do NOT duplicate `tiered_rpn.py`. Extend it.

---

### 2.5 CAS Bridge — Fix the Fakes and Wire Opcodes

**Fix `_execute_matrix_rpn_program`:** Remove the placeholder return. Tier 3 returns a vector result from GPU. Read back `embedding_dim` floats via `loader.memcpy_dtoh`. Return that list.

**Fix `compile_matrix`:** Remove `program.matrices.extend([0.0] * ...)`. Matrix shape is communicated as parameters to the GPU kernel, not as zeroed data appended to opcodes.

**Remove hardcoded metrics:** Replace `'sovereign_gpu_utilization': '100%'` with actual tracking (count GPU calls, raise if that counter is 0).

**Wire CAS opcodes to the kernel:** You have two choices:

**Option A (preferred):** Map CAS operations to existing RPN opcodes that already exist in the kernel. The kernel already handles:
- `0x70–0x76`: TADD, TMUL, TNOT, TCOMP, TQUANT, TPACK, TUNPACK
- `0x20–0x26`: sqrt, exp, log, sin, cos, tan
- `0x5A–0x5F`: matrix operations

Map `CASOpcodes.TERNARY_AND` → `0x70` (TADD in ternary), etc. This requires understanding what each ternary kernel opcode actually does — read `modular_rpn_kernel_extended.cu` before mapping.

**Option B:** Add a new CAS dispatch block to `modular_rpn_kernel_extended.cu` handling opcodes `0x100`–`0x14F`. Each opcode must have a real CUDA implementation in the kernel switch-case. For DIFFERENTIATE (0x120), a GPU-native finite difference or symbolic differentiation is required — not a Python fallback.

Either option is acceptable. Do not implement Option A half-way (mapping to opcodes that do the wrong thing) or Option B half-way (adding cases that call `break` with no computation).

**The `import math` line:** Remove it. It is unused.

**The `import re` inside `_tokenize`:** This is ingestion-path Python (string tokenization before GPU dispatch). It is acceptable. Add a comment: `# ingestion path only — tokenization before GPU compilation`.

---

### 2.6 Drawing Engine PTX — Fix Invalid Syntax

**PTX has no struct types.** Every field declared in a `.struct` block must become a separate `.param` or register. Fix both `quantum_field_3d_emission.ptx` and `text_3d_fusion_kernel.ptx`.

**Pattern: replace struct params with individual params:**

Instead of:
```ptx
// INVALID
.param .u64 bio_vision_params  // pointer to BiologicalVisionParams struct
```

Use:
```ptx
.param .f32 foveal_center_x,
.param .f32 foveal_center_y,
.param .f32 foveal_center_z,
.param .f32 foveal_radius,
.param .f32 rod_sensitivity,
.param .f32 cone_sensitivity,
```

Or pass a packed GPU buffer and load fields with `ld.global.f32 %val, [%ptr+offset]` where offset is the byte offset of each field. The second approach is preferred for large parameter sets — define the struct layout as a comment with explicit byte offsets:

```ptx
// BiologicalVisionParams layout (all .f32, 4 bytes each):
//   +0  foveal_center_x
//   +4  foveal_center_y
//   +8  foveal_center_z
//   +12 foveal_radius
//   +16 rod_sensitivity
//   +20 cone_sensitivity
//   +24 peripheral_weight
//   +28 temporal_weight
// Total: 32 bytes
```

Then in the kernel body: `ld.global.f32 %foveal_cx, [%bio_ptr+0]` etc.

**`.f3` type:** Does not exist. Use three `.f32` registers: `%foveal_cx, %foveal_cy, %foveal_cz`.

**Texture declarations:** Remove `.texture .f32 ..., 2, cudaArrayCubemap`. PTX textures use `.texref` and require sampler state setup outside the PTX file. If you need texture sampling, use `tex.3d` instruction with a pre-bound texref. If you do not need it yet, remove the declarations and pass the coefficients as a device pointer instead.

After fixing, both files must pass:
```bash
ptxas --gpu-name sm_89 quantum_field_3d_emission.ptx
ptxas --gpu-name sm_89 text_3d_fusion_kernel.ptx
```

No warnings about unknown syntax.

---

### 2.7 3D Model WINE Adapter — Wire It

`knowledge3d/tablet/wine/3d_model_wine.py` is a real file with real logic. Wire it.

In `knowledge3d/tablet/wine/__init__.py`, add:
```python
from knowledge3d.tablet.wine.3d_model_wine import (
    TRELLISWineAdapter,
    HunyuanWineAdapter,
    External3DWineBridge,
)
```

Verify `TabletIngest.procedural_3d_task` exists in `knowledge3d/bridge/headless_tablet.py`. If it does not, add the method — it should create and return a `TabletEnvelope` with `rpn_program`, `source`, `embeddings`, and `metadata` fields set.

---

### 2.8 Drawing Engine Phases 2–6 — Remaining Work

The following phases from `TEMP/PTX_DRAWING_ENGINE_PLAN.md` were not implemented at all. This is the actual remaining scope:

**Phase 2 — Advanced Drawing Primitives (new RPN opcodes):**

Add to `rpn_opcodes.py`:
```python
OP_BEZIER_EVAL      = 0x6C  # t p0x p0y p1x p1y p2x p2y p3x p3y → x y
OP_SHAPE_UNION      = 0x6D  # shape_a shape_b → result
OP_SHAPE_INTERSECT  = 0x6E
OP_SHAPE_SUBTRACT   = 0x6F
OP_REL_LINE         = 0x70  # x0_frac y0_frac x1_frac y1_frac (fractional coords)
OP_FIELD_COEF       = 0x71  # c0..c7 → field
OP_DOT_EMIT         = 0x72  # x y field → emit dot
```

Add the corresponding CUDA kernel bodies to `modular_rpn_kernel.cu` (or a new `drawing_primitives.cu`) and compile them to PTX. Wire new opcode handlers in `modular_rpn_engine.py` by checking the opcode range in `execute`.

**Phase 3 — VectorDotMap Codec:**

`vectordotmap_encoder.cu` already exists. Read it. Add the decoding path (field coefficients → pixel raster on GPU). Wire the Python bridge in `ptx_runtime/drawing_effects.py` which already exists.

**Phase 4 — Lighting and Layer Ops:**

Add to `rpn_opcodes.py`:
```python
OP_LAYER_NEW        = 0x78
OP_LAYER_BLEND      = 0x79
OP_BLEND_MULTIPLY   = 0x7A
OP_BLEND_SCREEN     = 0x7B
OP_BLEND_OVERLAY    = 0x7C
OP_ATMOSPHERE_FOG   = 0x7D
OP_VIGNETTE         = 0x7E
```

`filter_convolution.cu` already exists. Wire it for the filter opcodes. `lighting_simulation` needs to be written — the CUDA kernel structure from the plan is correct, implement it in `drawing_primitives.cu`.

**Phase 5 — Cross-Modal Symlinks:**

In the Drawing Galaxy population code, add `DrawingRule` entries that reference Math Galaxy symbols. The `symbol_refs=[966]` example from the plan (φ, phi) is concrete — look up the actual star_id for φ in the Math Galaxy and write the rule. This is ingestion-path Python and Galaxy entry creation, not GPU kernel work.

**Phase 6 — 3D Technique Suite:**

The plan lists 11 PTX kernels. Implement them in order of complexity:
1. `nurbs_evaluator.cu` — NURBS curve/surface (de Boor algorithm)
2. `marching_cubes_3d.cu` — SDF → mesh (the plan's kernel structure is complete, implement the lookup table and edge interpolation)
3. `lsystem_generator.cu` — L-system string expansion + turtle geometry
4. `parametric_surfaces.cu` — sphere, torus, Möbius (parametric equations)
5. `csg_operations.cu` — boolean mesh ops (union/intersect/subtract via SDF)

Do not write all 11 at once. Do 1 at a time, verify compilation, wire it, then move to the next.

---

## Part 3: Wiring Checklist

For every file you create or modify, the following must be true before you mark it done:

| Check | Meaning |
|-------|---------|
| Compiles | `.cu` compiles with `nvcc`; `.ptx` passes `ptxas` |
| Loaded | Python bridge calls `loader.load_ptx_file(path, kernel_name)` |
| Called | At least one code path in an existing module calls the bridge method |
| Tested | A test in `tests/` exercises the real GPU path (not mocked) |
| No fallback | No Python math, no `pass` in error branches, no placeholder returns |

---

## Part 4: File Disposition Summary

| File | Action |
|------|--------|
| `galaxy_memory_updater_zero_copy.cu` | KEEP — wire into `galaxy_memory_updater.py` |
| `zero_copy_memory_manager.cu` | DELETE AND REWRITE — use `cuMemHostAlloc` pattern |
| `zero_copy_memory_manager_phase4.cu` | KEEP 3 kernel functions, delete the C++ class wrapper |
| `transfer_yard_tiered.py` | KEEP Tier 2 & Tier 3, DELETE fake Tier 1, ADD thin Tier 1 wrapper |
| `cas_integration_bridge.py` | FIX placeholder returns, FIX opcode→kernel mapping, REMOVE `'100%'` |
| `3d_model_wine.py` | KEEP — wire into `tablet/wine/__init__.py` |
| `quantum_field_3d_emission.ptx` | FIX invalid struct/type/texture syntax, then verify with `ptxas` |
| `text_3d_fusion_kernel.ptx` | FIX invalid struct/type syntax, then verify with `ptxas` |
| `3d_specialist_integration.ptx` | AUDIT — same syntax check required |
| `sovereignty_validation_3d.ptx` | AUDIT — same syntax check required |
| New `procedural_content_bridge.py` | CREATE — wraps the 3 kernels from phase4 .cu with ctypes |
| New `zero_copy_bridge.py` | CREATE — wraps `cuMemHostAlloc` C interface with ctypes |

---

## Part 5: Completion Criteria (What "Done" Means)

When all of the above is complete, these tests must pass on real hardware (k3d-cranium env with RTX):

```bash
# Zero-copy galaxy update — real GPU, real embedding blend
python -m pytest tests/test_zero_copy_kernels.py -v

# Transfer Yard tier dispatch — all three tiers call GPU
python -m pytest tests/test_rpn_tiers.py -v

# CAS evaluation — at least add/mul/sin expressions execute on GPU  
python -m pytest tests/test_sovereign_cas_benchmark_simple.py -v

# Drawing primitives — BEZIER_EVAL opcode produces correct output
python -m pytest tests/test_drawing_primitives.py -v
```

None of these tests should mock the loader, skip GPU, or accept CPU-path results as valid.

A completion report may only be written after all four test suites produce PASSED output on the RTX.

---

*This spec supersedes `TEMP/PTX_DRAWING_ENGINE_IMPLEMENTATION_COMPLETE.md` which contained incorrect completion claims.*  
*Follow the architecture in `docs/vocabulary/` for all Galaxy and sovereign pipeline decisions.*  
*When in doubt, fail loudly and fix on GPU — never silently accept a CPU fallback.*
