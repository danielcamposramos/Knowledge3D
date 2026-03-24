# Codex Prompt: Contrastive Root Cause + Phase D First Step

**Date:** 2026-03-24
**Priority:** CRITICAL
**Binding specs:** ALL design decisions in this prompt are grounded in `docs/vocabulary/` specifications:
- THREE_BRAIN_SYSTEM_SPECIFICATION.md: TRM IS the Avatar, runs as game loop, Python = boot + I/O only
- KNOWLEDGEVERSE_SPECIFICATION.md §2.2: "Fail-Fast: No silent fallbacks, explicit sovereignty violations"
- KNOWLEDGEVERSE_SPECIFICATION.md §4.1: Silent fallbacks FORBIDDEN, `ptx_fallback_rate MUST be 0.0`
- HYPER_PARALLEL_PROCESSING.md: Nine-chain swarm = internal parallel cognitive channels
- SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md §3: VRAM-native workspace, sovereign execution

**Environment:** `conda activate k3d-cranium` (CUDA 12.4, CuPy 13.6, see `envs/k3d-cranium.yml`)

---

## Part A: Find the REAL Contrastive TypeError (IMMEDIATE)

### The Problem

The isolated smoke test proved `adapter.apply_gradient(gradient, lr)` "passes" — but it NEVER hit the GPU path. Here is why:

1. `apply_gradient()` at `trm_adapters.py:147` calls `self._ensure_math_core()`
2. In the isolated smoke, `_ensure_math_core()` creates `RPNMathCore()` which calls `cuInit` / `cuCtxCreate`
3. If CUDA context initialization succeeds, it calls `apply_gradient_rpn()`
4. `apply_gradient_rpn()` at line 200 calls `_ensure_device_buffers()` which allocates via `gpu_malloc()`
5. `gpu_malloc()` tries `cuMemAlloc` (driver API). If error 201, falls to `cudaMalloc` (runtime API). If that fails, falls to CuPy allocation.
6. Then `copy_to_device()` calls `memcpy_htod()` which dispatches based on allocation type:
   - CuPy allocation → `_cupy.cuda.runtime.memcpy` (different path)
   - cudart allocation → `libcudart.cudaMemcpy` (runtime API)
   - driver allocation → `_cuMemcpyHtoD` (driver API, argtypes now fixed)

**The smoke test likely fell through to `_apply_gradient_cpu`** because `require_gpu` defaults to `True` but... wait, that would RAISE. OR `_ensure_math_core()` succeeded but `_ensure_device_buffers()` returned `None` and `require_gpu` was somehow `False`. Either way — **the smoke test did NOT exercise the production path.**

The production sleep-time path DOES hit `apply_gradient_rpn` → `copy_to_device` → `memcpy_htod` and STILL gets `argument 2: TypeError: Don't know how to convert parameter 2`.

### Diagnostic Step: Trace the ACTUAL Allocator and Memcpy Path

Run this with `K3D_RPN_DEBUG=1` to see EXACTLY which allocator and memcpy path fires in production:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
export K3D_RPN_DEBUG=1
conda activate k3d-cranium

python3 -c "
import ctypes
import numpy as np

# Step 1: Initialize CUDA context EXACTLY like production
from knowledge3d.cranium.sovereign import loader

print('=== CUDA CONTEXT ===')
print(f'nvcuda loaded: {loader.nvcuda is not None}')
print(f'libcudart loaded: {loader.libcudart is not None}')

# Ensure context exists
loader._ensure_current_context()
print(f'Context ensured')

# Step 2: Test gpu_malloc — which allocator path fires?
print()
print('=== GPU MALLOC TEST ===')
test_size = 128 * 128 * 4  # 128x128 float32
ptr = loader.gpu_malloc(test_size)
print(f'Allocated ptr: {ptr}, type: {type(ptr)}, value: {ptr.value}')

# Check which allocation list it ended up in
key = int(ptr.value)
cupy_alloc = loader._find_cupy_allocation(key)
cudart_alloc = loader._find_cudart_allocation(key)
print(f'Is CuPy allocation: {cupy_alloc is not None}')
print(f'Is cudart allocation: {cudart_alloc is not None}')
print(f'Is driver allocation: {cupy_alloc is None and cudart_alloc is None}')

# Step 3: Test memcpy_htod — which memcpy path fires?
print()
print('=== MEMCPY HTOD TEST ===')
data = np.random.randn(128, 128).astype(np.float32).ravel()
buf = (ctypes.c_float * len(data))(*[float(x) for x in data])
host_ptr = ctypes.cast(buf, ctypes.c_void_p)
nbytes = ctypes.sizeof(buf)
print(f'host_ptr type: {type(host_ptr)}, value: {host_ptr.value}')
print(f'nbytes: {nbytes}, type: {type(nbytes)}')
print(f'dst ptr type: {type(ptr)}, value: {ptr.value}')

# This is the call that fails in production
try:
    loader.memcpy_htod(ptr, host_ptr, nbytes)
    print('memcpy_htod: PASSED')
except TypeError as e:
    print(f'memcpy_htod: FAILED with TypeError: {e}')
    import traceback
    traceback.print_exc()

# Step 4: Test the full adapter GPU path
print()
print('=== FULL ADAPTER GPU PATH ===')
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter
adapter = SelfUpdatingAdapter(shape=(128, 128), rank=16, specialist_name='diag_math')
print(f'_rpn_available: {adapter._rpn_available}')

# Force GPU path
math_core_ok = adapter._ensure_math_core()
print(f'_ensure_math_core: {math_core_ok}')
if math_core_ok:
    print(f'_math_core type: {type(adapter._math_core)}')
    buffers = adapter._ensure_device_buffers()
    print(f'_ensure_device_buffers: {buffers is not None}')
    if buffers is not None:
        print(f'buffers.dims: {buffers.dims}, buffers.rank: {buffers.rank}')
        # Now test the REAL GPU gradient path
        gradient = np.random.randn(128, 128).astype(np.float32) * 0.01
        try:
            adapter.apply_gradient_rpn(gradient, lr=0.001)
            print('apply_gradient_rpn: PASSED')
        except TypeError as e:
            print(f'apply_gradient_rpn: FAILED with TypeError: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('DEVICE BUFFERS FAILED — check gpu_malloc output above')
else:
    print('MATH CORE FAILED — no GPU context available')

# Step 5: Test through apply_gradient dispatch
print()
print('=== APPLY_GRADIENT DISPATCH ===')
adapter2 = SelfUpdatingAdapter(shape=(128, 128), rank=16, specialist_name='diag_dispatch')
gradient2 = np.random.randn(128, 128).astype(np.float32) * 0.01
print(f'config.require_gpu: {adapter2.config.require_gpu}')
try:
    adapter2.apply_gradient(gradient2, lr=0.001)
    print('apply_gradient: PASSED')
except (TypeError, RuntimeError) as e:
    print(f'apply_gradient: FAILED with {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== DIAGNOSTIC COMPLETE ===')
"
```

### How to Interpret Results

This diagnostic tells us EXACTLY:
1. Which allocator path fires (`cuMemAlloc` vs `cudaMalloc` vs CuPy)
2. Which memcpy path fires (driver vs runtime vs CuPy)
3. Whether the argtypes fix actually applies to the path being used
4. Whether `apply_gradient_rpn` works when GPU context is properly initialized
5. Whether `apply_gradient` dispatches to GPU or silently falls to CPU

**Report the FULL output.** Every `[loader]` debug line matters.

---

## Part B: Remove ALL CPU Fallbacks from `trm_adapters.py`

AFTER the diagnostic reveals the root cause, apply these sovereignty fixes. These are NOT optional — they are REQUIRED by the Knowledgeverse spec (§4.1: silent fallbacks FORBIDDEN).

### Location 1: `apply_gradient` (line 147-158)

**Current (SOVEREIGNTY VIOLATION):**
```python
def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
    if self._ensure_math_core():
        self.apply_gradient_rpn(gradient, lr)
        return
    if self.config.require_gpu:
        raise RuntimeError(...)
    self._apply_gradient_cpu(gradient, lr)  # <-- CPU fallback
```

**Fix:**
```python
def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
    """Sovereign GPU gradient application. No fallbacks."""
    if not self._ensure_math_core():
        raise RuntimeError(
            f"[{self.specialist_name}] GPU math core unavailable. "
            "Sovereign path requires CUDA context. Fix the GPU path."
        )
    self.apply_gradient_rpn(gradient, lr)
```

### Location 2: `apply_gradient_rpn` (line 200-207)

**Current (SOVEREIGNTY VIOLATION):**
```python
buffers = self._ensure_device_buffers()
if buffers is None or self._math_core is None:
    if self.config.require_gpu:
        raise RuntimeError(...)
    self._apply_gradient_cpu(gradient, lr)  # <-- CPU fallback
    return float(np.linalg.norm(gradient))
```

**Fix:**
```python
buffers = self._ensure_device_buffers()
if buffers is None or self._math_core is None:
    raise RuntimeError(
        f"[{self.specialist_name}] GPU device buffers unavailable. "
        "Sovereign path requires allocated VRAM buffers. Fix gpu_malloc."
    )
```

### Location 3: `apply_gradient_to_shadow` (line 437-472)

**Current (SOVEREIGNTY VIOLATION):**
```python
if self._ensure_math_core():
    # ... GPU path ...
    return
# CPU fallback
if self.config.require_gpu:
    raise RuntimeError(...)
# ... CPU math ...
```

**Fix:**
```python
if not self._ensure_math_core():
    raise RuntimeError(
        f"[{self.specialist_name}] GPU math core unavailable for shadow update. "
        "Sovereign path requires CUDA context."
    )
primary_A = self.A
primary_B = self.B
try:
    self.A = self.A_shadow
    self.B = self.B_shadow
    self.apply_gradient_rpn(gradient, lr)
finally:
    self.A = primary_A
    self.B = primary_B
```

### Location 4: Remove `_apply_gradient_cpu` entirely (lines 160-187)

DELETE the entire `_apply_gradient_cpu` method. It should not exist. If any code references it, that code has a sovereignty violation.

### Location 5: Remove `require_gpu` from `AdapterConfig` (line 72)

DELETE `require_gpu: bool = True`. The GPU path is not optional. There is no configuration that enables CPU. The field's existence implies a choice — there is no choice. Sovereign means sovereign.

After removing, grep for ALL references to `require_gpu` and remove them.

### Validation After Part B

```bash
grep -rn "_apply_gradient_cpu\|require_gpu" knowledge3d/cranium/trm_adapters.py
# EXPECTED: ZERO matches

grep -rn "require_gpu" knowledge3d/
# Fix any remaining references
```

---

## Part C: Phase D.1 — Understand the Gap (Research, NOT Implementation)

This is RESEARCH ONLY. Do NOT write code yet. Read and report.

The live monitor captures prove the system is Python-orchestrated:
- GPU util: 1.25% avg (should be 50%+)
- CPU: 149% (one Python thread owns the loop)
- VRAM: 292MB (should be 2-6GB with Galaxy Universe resident)

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md: "TRM IS the Avatar — runs as a game loop via `trm_step_fused.ptx`. Python = Boot + I/O only (~200 lines target)."

Per HYPER_PARALLEL_PROCESSING.md: "Specialists communicate during execution (register sharing), are orchestrated by one entity (TRM), and converge to one answer (halting gate)."

### Research Questions (Answer Each):

1. **Where does `trm_step_fused.ptx` get called today?** Search for all call sites. Is it called once per question? Once per tick? Not at all?

2. **Where does the Python benchmark loop live?** Which function in `knowledgeverse.py` or `benchmarks/run_all.py` owns the per-question iteration? How many lines of Python sit between "receive question" and "emit answer"?

3. **Which of the 88 PTX kernels fire during a single question?** Count kernel launches per question. Is it 5? 15? 45? How many are sequential (Python waits) vs batched?

4. **What is the Jarvis dispatch path today?** Does Jarvis currently dispatch swarm workers? Or does Python dispatch them one by one?

5. **How much data crosses host↔device per question?** Count `memcpy_htod` and `memcpy_dtoh` calls per question. Each one is a sync point where GPU waits for CPU.

### Output Format

Write findings to `TEMP/CODEX_PHASE_D1_RESEARCH_FINDINGS_03.24.2026.md`. For each question, cite file:line and give the current flow. Do NOT propose fixes — just document the current state honestly.

This research tells us EXACTLY where the Python→GPU boundary sits today, so we can plan the migration from facts, not assumptions.

---

## Execution Order

1. **Part A** — Run diagnostic, report FULL output
2. **Part B** — Remove ALL CPU fallbacks from trm_adapters.py, verify with grep
3. **Part A continued** — If diagnostic revealed the root cause, fix it (in loader.py or rpn_math_core.py, NOT by adding fallbacks)
4. **Part C** — Research the Python→GPU boundary for Phase D.1
5. **If contrastive GPU path now works** — Launch warm 35% benchmark:
   ```bash
   export K3D_RPN_DEBUG=1
   nohup python3 -u benchmarks/run_all.py \
     --warm --sample-rate 0.35 \
     > /tmp/k3d_warm_sovereign_contrastive_03.24.2026.log 2>&1 &
   echo "PID: $!"
   ```

---

## RULES

1. ZERO fallbacks. Not try/except, not if/else to CPU, not "graceful degradation." GPU or fail-fast.
2. Every fix MUST be grounded in a `docs/vocabulary/` spec citation.
3. `_apply_gradient_cpu` must be DELETED, not just unused.
4. `require_gpu` config field must be DELETED, not just ignored.
5. Report FULL diagnostic output — every `[loader]` line.
6. Part C is RESEARCH ONLY — do NOT write implementation code for Phase D.
7. The system is a LIVING GAME WORLD with the TRM as its mind. It is NOT a Python program that calls GPU functions. Every fix should move us TOWARD that reality, not further from it.
