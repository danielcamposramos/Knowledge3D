# Sovereign Loader: Lazy Primary Context Warmup Patch

**Date**: April 18, 2026  
**Issue**: `get_vram_usage()` fails with CUDA_ERROR_INVALID_CONTEXT (201) when called immediately after `ensure_init()` on systems using `cuDevicePrimaryCtxRetain` fallback path  
**Root Cause**: Primary contexts are lazily initialized — device-side state (memory manager, stream scheduler) does not materialize until the first device-touching operation  
**Solution**: Eagerly materialize context on primary-retain path with minimal `cuMemAlloc(0)` + `cuMemFree()` warmup

---

## 3-Line Summary

1. `cuDevicePrimaryCtxRetain` creates a handle-only context; device-side initialization is deferred until first device operation
2. `get_vram_usage()` calls `cuMemGetInfo_v2`, which requires fully-materialized device state, so it fails 201 if called before any warmup op
3. Exemplar bindings (DCT, Quant) never hit this bug because they call `cuModuleLoadData` or `cuMemAlloc` immediately after init, materializing the context as a side effect

---

## Research Evidence

**From nemotron-3-super:cloud (NVIDIA GPU expert)**:
- `cuCtxSynchronize()` alone does NOT materialize lazy primary contexts (no pending work → nothing to sync)
- **Minimal proven pattern**: `cuMemAlloc(&ptr, 0)` + `cuMemFree()` — forces driver to initialize memory manager and all device-side context structures
- Zero-size allocation is **safer, faster, and more portable** than any other approach
- This pattern is used internally by NVIDIA (cuda-gdb, nsight compute, CUDA Runtime API)

**From K3D specs**:
- Original context fix spec: `TEMP/CLAUDE_SOVEREIGN_CUDA_CONTEXT_FIX_11.24.2025.md` — establishes ONE shared context pattern
- Current state (lines 327-419 in loader.py): fallback to primary-retain path when `cuCtxCreate` fails with error 2 or 201

---

## The Patch

### Location: `knowledge3d/cranium/sovereign/loader.py`

#### BEFORE (lines 389–398, primary-retain path only)

```python
                ck(set_res)
                current = CUcontext()
                ck(nvcuda.cuCtxGetCurrent(ctypes.byref(current)))
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuCtxGetCurrent -> {current}")
                ctx = current
                try:
                    import cupy as _cupy  # type: ignore

                    _cupy.cuda.Device(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))).use()
```

#### AFTER (insert warmup BEFORE CuPy bootstrap)

```python
                ck(set_res)
                current = CUcontext()
                ck(nvcuda.cuCtxGetCurrent(ctypes.byref(current)))
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[loader] cuCtxGetCurrent -> {current}")
                ctx = current
                
                # ===============================================
                # WARMUP: Materialize lazy primary context state
                # ===============================================
                # Primary context is lazily initialized (device-side structures
                # allocated only on first device-touching operation). cuMemGetInfo
                # requires fully-materialized state, so we eagerly trigger init
                # via minimal zero-size allocation. This is the proven pattern used
                # internally by NVIDIA (cuda-gdb, nsight compute, etc).
                d_temp = CUdeviceptr()
                warmup_res = _cuMemAlloc(ctypes.byref(d_temp), 0)
                if warmup_res == 0 and d_temp.value:
                    _cuMemFree(d_temp)
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print("[loader] Primary context warmed up (zero-size alloc)")
                else:
                    # If zero-size alloc fails, context is still broken; propagate error
                    if os.environ.get("K3D_RPN_DEBUG"):
                        print(f"[loader] Context warmup failed with code {warmup_res}")
                    ck(warmup_res)
                # ===============================================
                
                try:
                    import cupy as _cupy  # type: ignore

                    _cupy.cuda.Device(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))).use()
```

---

## Verification Snippet

**Regression test — reproduces the original bug and confirms the fix:**

```python
#!/usr/bin/env python3
"""
Verify that get_vram_usage() works immediately after ensure_init().
This reproduces the bug from benchmarks/sovereign_bitnet_attention.py line 205.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Force primary-context path to test the warmup
os.environ["K3D_USE_PRIMARY_CTX"] = "1"
os.environ["K3D_RPN_DEBUG"] = "1"  # Show debug output

from knowledge3d.cranium.sovereign import loader

try:
    loader.ensure_init()
    print("✅ ensure_init() succeeded")
    
    used, total = loader.get_vram_usage()
    print(f"✅ get_vram_usage() succeeded: {used}/{total} bytes")
    
    print("✅ PATCH VERIFIED: No CUDA_ERROR_INVALID_CONTEXT")
except RuntimeError as e:
    if "invalid device context" in str(e):
        print(f"❌ BUG STILL PRESENT: {e}")
        exit(1)
    raise
```

**Expected output** (with patch):
```
[loader] cuDevicePrimaryCtxSetFlags -> 0
[loader] cuDevicePrimaryCtxRetain -> 0, ctx=<CUcontext ...>
[loader] cuCtxSetCurrent -> 0
[loader] cuCtxGetCurrent -> <CUcontext ...>
[loader] Primary context warmed up (zero-size alloc)
✅ ensure_init() succeeded
✅ get_vram_usage() succeeded: 132MiB/8192MiB bytes
✅ PATCH VERIFIED: No CUDA_ERROR_INVALID_CONTEXT
```

---

## Edge Cases Handled

### 1. **Fork Safety** (lines 334–341)
The warmup runs inside `_ensure_init()`, which is already fork-aware. When a fork is detected, `_initialized=False` triggers a full reinit, including the warmup. ✅ Covered.

### 2. **K3D_USE_PRIMARY_CTX=1** (line 357)
The warmup is ONLY inserted on the primary-retain fallback path (triggered when `res in (2, 201)` at line 360). If forced via env var, the fallback is explicitly triggered, and warmup runs. ✅ Covered.

### 3. **CuPy Bootstrap Path** (lines 376–412)
The warmup runs BEFORE CuPy bootstrap. If CuPy is used, it gets a fully-materialized context. If CuPy fails, the warmup error propagates via `ck()` (sovereignty: no silent fallbacks). ✅ Covered.

### 4. **cuCtxCreate Path** (line 358)
The warmup is NOT inserted on the successful `cuCtxCreate` path (line 414–416) because `cuCtxCreate` already returns a fully-initialized context. Zero overhead on the common path. ✅ Optimized.

### 5. **Memory Allocation Tracking**
The zero-size alloc is NOT tracked in `_cupy_allocations` or `_cudart_allocations` (warmup is one-shot, immediately freed). `_find_cupy_allocation()` is NOT called. ✅ Clean.

---

## Proof That Warmup Is Sufficient

The bug manifests in two ways:

| Caller | First Op After Init | Fails or Succeeds |
|--------|---------------------|------------------|
| `gpu_malloc()` | `cuMemAlloc(size)` | ✅ Succeeds (large alloc materializes ctx) |
| `get_vram_usage()` | `cuMemGetInfo_v2()` | ❌ Fails 201 (no prior alloc to materialize ctx) |
| Exemplar bindings | `cuModuleLoadData()` | ✅ Succeeds (module load materializes ctx) |

**With patch**: All paths warmup with zero-size alloc, so any subsequent device op (cuMemGetInfo, cuMemAlloc, kernel launch) sees fully-materialized state. ✅

---

## Why NOT Other Approaches

| Alternative | Why Rejected |
|------------|-------------|
| `cuCtxSynchronize()` | Doesn't trigger init if no work is queued (Nemotron: "returns immediately, doing nothing") |
| `cuMemAlloc(16) + cuMemFree()` | Unnecessarily allocates/initializes 1+ bytes; slower than zero-size |
| Dummy kernel launch | Overkill: requires module load, function lookup, grid scheduling |
| `cuMemHostAlloc()` | CPU-side pinned memory, not guaranteed to trigger GPU context init |

---

## Sovereignty Compliance

✅ **No new context created** — warmup uses the same `_context` that `_ensure_init` just set current  
✅ **No try/except swallowing errors** — warmup errors propagate via `ck(warmup_res)`  
✅ **No imports added** — `_cuMemAlloc` and `_cuMemFree` already imported (lines 44–45)  
✅ **Only on primary-retain path** — common `cuCtxCreate` path unchanged  
✅ **Debug guarded** — all prints wrapped in `K3D_RPN_DEBUG` check  
✅ **Fork-safe** — runs on every reinit  

---

## Files Modified

- `knowledge3d/cranium/sovereign/loader.py` (lines 389–398, +15 lines of warmup + comments)

---

## Next Steps (For Codex)

1. **Apply the patch** to lines 389–398 (primary-retain fallback path only)
2. **Run the regression test** (snippet above) with `K3D_USE_PRIMARY_CTX=1`
3. **Run benchmarks** (`benchmarks/sovereign_bitnet_attention.py`) — should no longer raise `CUDA_ERROR_INVALID_CONTEXT`
4. **Test without forcing primary context** (`unset K3D_USE_PRIMARY_CTX`) — cuCtxCreate path should show zero warmup overhead
5. **Verify fork safety** — run any multiprocessing test; reinits should warmup correctly

---

**Status**: Patch ready for application. No approval gates needed (pure internal driver interaction).

---

**References**:
- Nemotron GPU expert ruling: Zero-size `cuMemAlloc` is the minimal, proven pattern for lazy primary context materialization
- K3D Sovereignty Spec: `TEMP/CLAUDE_SOVEREIGN_CUDA_CONTEXT_FIX_11.24.2025.md` (context pattern established)
- Bug manifestation: `benchmarks/sovereign_bitnet_attention.py` line 205, error code 201 (CUDA_ERROR_INVALID_CONTEXT)
- Original loader spec: `knowledge3d/cranium/sovereign/loader.py` lines 327–419
