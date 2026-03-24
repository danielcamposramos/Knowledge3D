# Codex Prompt: Unblock Contrastive Learning + Knowledge Expansion Path

**Date:** 2026-03-23
**Priority:** CRITICAL — Contrastive learning has been blocked for 3 runs
**Context:** Warm 35% at 18.58%. The `.ravel()` fix was correct (prevents float-on-row error) but the REAL TypeError is deeper — in the ctypes CUDA driver call `nvcuda.cuMemcpyHtoD` at `loader.py:438`, which has NO `argtypes` set. The CuPy test passed because the test uses a different allocator path (CuPy-managed) than the production runtime (driver API).

---

## Fix 1: Unblock Contrastive Training (IMMEDIATE)

### Problem

Contrastive training fails for ALL 4 specialists with:
```
argument 2: TypeError: Don't know how to convert parameter 2
```

The `.ravel()` fix in `copy_to_device` was necessary but insufficient. The error comes from **`nvcuda.cuMemcpyHtoD(dst_device, src_host, size_bytes)`** at `loader.py:438`. This ctypes call to the CUDA Driver API has **no `argtypes` set**, so ctypes auto-converts arguments. The `src_host` parameter (`ctypes.c_void_p`) or `size_bytes` (Python `int`) fails auto-conversion depending on the driver version and symbol resolution (`cuMemcpyHtoD` v1 vs `cuMemcpyHtoD_v2`).

The `.ravel()` fix passed in tests because the test allocates via CuPy (which goes through `_cupy.cuda.runtime.memcpy`, a different path). Production allocates via `cuMemAlloc` (driver API), which hits the bare `nvcuda.cuMemcpyHtoD` path.

### Two-Part Fix

#### Fix 1a: Set `argtypes` for `cuMemcpyHtoD` and `cuMemcpyDtoH`

File: `knowledge3d/cranium/sovereign/loader.py`

After line 34 (where `libcudart` argtypes are set), add argtypes for the driver API memcpy functions:

```python
# Set argtypes for CUDA driver memcpy to prevent ctypes auto-conversion errors.
# cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
# cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
try:
    nvcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    nvcuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
    _cuMemcpyHtoD = nvcuda.cuMemcpyHtoD_v2
except AttributeError:
    nvcuda.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    nvcuda.cuMemcpyHtoD.restype = ctypes.c_int
    _cuMemcpyHtoD = nvcuda.cuMemcpyHtoD

try:
    nvcuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    nvcuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    _cuMemcpyDtoH = nvcuda.cuMemcpyDtoH_v2
except AttributeError:
    nvcuda.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    nvcuda.cuMemcpyDtoH.restype = ctypes.c_int
    _cuMemcpyDtoH = nvcuda.cuMemcpyDtoH
```

Then update the call sites:

In `memcpy_htod` at line 438, replace:
```python
res = nvcuda.cuMemcpyHtoD(dst_device, src_host, size_bytes)
```
with:
```python
res = _cuMemcpyHtoD(dst_device, src_host, ctypes.c_size_t(size_bytes))
```

In `memcpy_dtoh` (similar location), replace:
```python
res = nvcuda.cuMemcpyDtoH(dst_host, src_device, size_bytes)
```
with:
```python
res = _cuMemcpyDtoH(dst_host, src_device, ctypes.c_size_t(size_bytes))
```

This is the same pattern already used for `cuMemGetInfo_v2` at line 39 and `_cuMemcpyDtoD` at line 71.

#### Fix 1b: Add try/except fallback in `_apply_adapter_gradient`

File: `knowledge3d/cranium/adaptive_swarm.py:567-581`

Even with Fix 1a, add resilience. Contrastive training is SLEEP-TIME (Python orchestration), NOT hot-path inference. A CPU fallback is acceptable here:

```python
@staticmethod
def _apply_adapter_gradient(adapter: Any, gradient: np.ndarray, lr: float) -> None:
    if (
        hasattr(adapter, 'config')
        and bool(getattr(adapter.config, 'require_gpu', True)) is False
        and hasattr(adapter, '_apply_gradient_cpu')
    ):
        adapter._apply_gradient_cpu(gradient, lr)
        return
    if hasattr(adapter, 'apply_gradient'):
        try:
            adapter.apply_gradient(gradient, lr=lr)
            return
        except (TypeError, RuntimeError):
            # GPU contrastive path failed — fall through to CPU
            pass
    if hasattr(adapter, '_apply_gradient_cpu'):
        adapter._apply_gradient_cpu(gradient, lr)
    elif hasattr(adapter, 'A') and hasattr(adapter, 'B'):
        grad_A = gradient @ adapter.B.T
        grad_B = adapter.A.T @ gradient
        adapter.A -= lr * grad_A
        adapter.B -= lr * grad_B
```

Key change: wrap `adapter.apply_gradient()` in try/except. On GPU failure, fall through to `_apply_gradient_cpu` or direct A/B update. This ensures contrastive training ALWAYS completes, even if the GPU RPN path has issues.

**Sovereignty note:** This fallback is in sleep-time only, not inference. The hot-path `apply_gradient_rpn` in the TRM game loop remains unchanged. Sleep-time contrastive updates happening on CPU vs GPU is an optimization choice, not a sovereignty violation.

### Files to Modify

- `knowledge3d/cranium/sovereign/loader.py` — Add `argtypes` for `cuMemcpyHtoD`/`cuMemcpyDtoH`, use versioned symbols
- `knowledge3d/cranium/adaptive_swarm.py:567-581` — Add try/except CPU fallback in `_apply_adapter_gradient`

### Validation

1. `nvcuda.cuMemcpyHtoD` (or `_v2`) has explicit `argtypes` — no more auto-conversion
2. Contrastive training completes for all 4 specialists (either GPU or CPU path)
3. `checkpoint` dict is non-empty in sleeptime journal
4. Zero "Don't know how to convert parameter 2" errors in logs
5. All existing tests still pass

---

## Fix 2: Galaxy Count Regression (19 vs 24)

### Problem

Previous warm run loaded: `247974 entries across 24 galaxies`
This run loaded: `247889 entries across 19 galaxies`

5 galaxies and 85 entries disappeared between runs. This needs investigation.

### Action

Check which 5 galaxies were lost. Compare the Galaxy loading code and warm-boot persistence to understand why. Likely causes:
- Galaxy state file corruption or incomplete save
- Bootstrap code change that dropped galaxies
- Filtering threshold change

Find and fix. All 24 galaxies (or more — we added LaTeX as the 11th default) must load on warm boot.

### Files to Check

- Galaxy state persistence code (save/load)
- `knowledgeverse.py` DEFAULT_GALAXIES (should have 11 with LaTeX)
- Warm boot loading path

---

## Architecture Note: More Knowledge IS the Path

Daniel is right. Looking at the scores:
- MMLU 22% with 247k entries = reasonable for Galaxy-navigation (no external LLM)
- ARC/Math/GSM8K/LHE all <6% = the Galaxy doesn't have enough procedural knowledge for these tasks

Contrastive learning will help the TRM navigate BETTER, but only if there's knowledge TO navigate. The bottleneck is Galaxy content, not navigation quality:

- **Math** (0.80%): Needs more procedural math programs in Math Galaxy. 500 competition problems need ~2000+ operation patterns.
- **GSM8K** (1.52%): Needs word-problem decomposition patterns in Grammar Galaxy. Step-by-step extraction rules.
- **ARC** (4.76%): Needs more visual transformation patterns in Drawing Galaxy. Grid manipulation programs.
- **LHE** (5.71%): Needs multi-hop reasoning chains in Grammar Galaxy. Logical inference programs.
- **MMLU** (22%): Broadest coverage — benefits most from general knowledge quantity.

Contrastive learning is the mechanism that makes knowledge STICK. But first we need the knowledge. Fix the contrastive blocker so learning works, then focus on knowledge expansion.

---

## Execution Order

1. **Fix 1a** (loader.py argtypes) — fixes the root ctypes issue
2. **Fix 1b** (adaptive_swarm.py fallback) — ensures contrastive training always completes
3. **Fix 2** (galaxy count investigation) — understand the 24→19 regression
4. **Quick smoke test** — run a small subset (e.g., 5% or 10 questions per suite) to verify contrastive training saves a checkpoint
5. **If smoke passes** — run full warm 35% and write report to `TEMP/CLAUDE_WARM_CONTRASTIVE_WORKING_REPORT_03.23.2026.md`

---

## Test Criteria

1. **Contrastive completes:** All 4 specialists show `trained: true`
2. **Checkpoint saved:** Non-empty `checkpoint` dict
3. **Adapter weights change:** Pre/post training A/B matrices differ
4. **No TypeError:** Zero "Don't know how to convert" in logs
5. **Galaxy count:** >= 24 galaxies loaded on warm boot (investigate if <24)
6. **Score:** >= 18.58% (should not regress from contrastive fix)
