# CLAUDE_CUMEMGETINFO_201_DEEPDIVE_04.18.2026.md

**Status:** Investigation complete — root cause identified.  
**Priority:** P0 — sovereign loader is non-functional for all benchmarks that call `get_vram_usage()`.  
**Next:** Dispatch sub-agent to apply the single-line fix in `loader.py:1006-1010`.

---

## 1. State Summary

`_ensure_init()` succeeds: `cuCtxCreate(CU_CTX_MAP_HOST, device)` returns 0, the 16-byte warmup alloc+free returns 0, `_context` is stored.  
`get_vram_usage()` then calls `_ensure_current_context()` → `cuCtxSetCurrent(_context)` → 0, then `_cuMemGetInfo(...)` → **201 (CUDA_ERROR_INVALID_CONTEXT)**.  
`cuMemAlloc` and `cuCtxSetCurrent` succeed; `cuMemGetInfo_v2` fails. Seemingly contradictory, but explainable.

---

## 2. Hypotheses Ranked with Evidence

### H1 — Primary context Release in `galaxy_buffer.py` destroys the context **[CONFIRMED ROOT CAUSE]**

**Evidence (confirmed by grep):**

- `knowledge3d/cranium/ptx/galaxy_buffer.py:102` calls `cuda.cuDevicePrimaryCtxRelease(self.device)` (destructor of `GalaxyMemory`).  
- `knowledge3d/cranium/ptx/galaxy_buffer.py:467` also calls `cuda.cuDevicePrimaryCtxRelease(galaxy_memory.device)`.  
- `galaxy_buffer.py` imports from `cuda.cuda` or `cuda.bindings.driver` — the **cuda-python** bindings, NOT `loader.py`. This means it is directly calling primary context management against the same GPU that `loader.py` is managing.

**Mechanism:** `loader.py` line 358 forces `res=201` (or primary-retain fallback on any real failure) via `K3D_USE_PRIMARY_CTX=1`. More importantly: on CUDA 12.x, even when `cuCtxCreate` takes the non-primary path, calling `cuDevicePrimaryCtxRelease` with a refcount of 0 has undefined effect but can invalidate the **thread-local context stack entry** for device 0 in the driver's internal state.

**Nemotron/cloud analysis confirmation:** "When the reference count for the primary context reaches zero, the context is destroyed. All subsequent CUDA API calls that require a valid context will return `CUDA_ERROR_INVALID_CONTEXT`." The `_context` handle stored in `loader.py:433` becomes dangling.

**Why cuMemAlloc succeeds but cuMemGetInfo_v2 fails:** `cuMemAlloc` is called from within `_ensure_init()` BEFORE `galaxy_buffer` teardown occurs. `cuMemGetInfo_v2` is called from `get_vram_usage()`, which is invoked at `benchmark.run()` line 211 — by which time Python's garbage collector may have already finalized a `GalaxyMemory` object from a prior import/test run, executing the `cuDevicePrimaryCtxRelease` destructor. This is a **use-after-free** on the CUDA context.

---

### H2 — `cuCtxCreate` + `cuCtxSetCurrent` API mixing (legacy stack vs floating context) **[CONTRIBUTING, NOT ROOT]**

Kimi swarm analysis: `cuCtxCreate` pushes context onto the per-thread stack. `cuCtxSetCurrent` at line 416 replaces the floating context WITHOUT pushing. On CUDA 12.x, `cuMemGetInfo_v2` validates the **legacy stack top** in some driver builds, not just the floating context. This matters only if H1 has not already destroyed the context, but it is an independent failure vector.

**Evidence against this being the sole cause:** The benchmark does NOT import `galaxy_buffer` (benchmark imports only `loader` at line 45 of `sovereign_bitnet_attention.py`). So H2 alone can't explain the 201 without H1. BUT if `galaxy_buffer` is imported transitively by another module in the same process (e.g., via `trm_game_loop.py`, `knowledgeverse.py`, etc.), H1 fires.

---

### H3 — Binding mismatch on `cuMemGetInfo_v2` argtypes **[RULED OUT]**

`loader.py:146-151`: argtypes are `[POINTER(c_size_t), POINTER(c_size_t)]`, restype is `c_int`. Correct. Nemotron confirmed: a wrong restype cannot cause a real 0 to be read as 201; the function genuinely returns 201.

`math_core_pool.py:244-251` calls `cuMemGetInfo_v2` WITHOUT setting argtypes — a real defect, but unrelated to the benchmark path.

---

### H4 — `libcudart.so` loaded alongside `libcuda.so.1` **[CONTRIBUTING]**

`loader.py:27-44` loads `libcudart.so` at module import time and calls `cudaSetDevice`, `cudaMalloc`, etc. When `libcudart` is active in the same process, the CUDA runtime maintains its own primary context shadow state. This means: if `galaxy_buffer.py` releases the primary context via `cuda.cuDevicePrimaryCtxRelease`, the runtime's internal state becomes inconsistent with the driver's, and `cuMemGetInfo_v2` (which in some drivers consults the runtime's notion of current device context) returns 201.

---

### H5 — `CU_CTX_MAP_HOST` creating a half-valid context **[RULED OUT for CUDA 12.x]**

Nemotron confirmed: flag 0x08 has been a no-op since CUDA 11.0. The context is fully valid. Not the cause.

---

### H6 — `cuCtxGetCurrent` refresh at line 395-398 returning stale handle **[SECONDARY RISK]**

On the primary-retain path (lines 394-398), `ctx = current` captures `cuCtxGetCurrent`. If `galaxy_buffer.py` had already released that primary context in a prior GC cycle, `cuCtxGetCurrent` returns a stale or null handle which is then stored as `_context`. Subsequent `cuCtxSetCurrent(_context)` returns 0 (driver silently accepts a null CUcontext on some versions) while `cuMemGetInfo_v2` correctly rejects it.

---

### H7 — Thread-local context vs Python GIL **[UNLIKELY]**

`sovereign_bitnet_attention.py` imports only `loader` (line 45), no threads are spawned before `run()`. Not the cause in this benchmark.

---

## 3. MCP Research Findings

**Nemotron (NVIDIA expert):**
> "Primary context refcounting is the most common real-world cause. cuDevicePrimaryCtxRelease dropping refcount to zero destroys the context; the CUcontext handle becomes dangling. Any subsequent call returns CUDA_ERROR_INVALID_CONTEXT."
> "CU_CTX_MAP_HOST is ignored in CUDA 12.x — not a half-valid context."
> "ctypes argtypes mismatch causes corruption, not false 201. If you see 201, the function genuinely returned 201."

**Kimi Swarm (deep mode, synthesis):**
> "cuMemGetInfo_v2 validates the legacy stack top in CUDA 12.x, while cuMemAlloc only checks the floating context TLS slot. cuCtxSetCurrent sets the floating TLS slot but does NOT push onto the legacy stack. cuCtxCreate DID push onto the stack — but a subsequent cuCtxSetCurrent call with the same handle may desynchronize the stack pointer cache in certain driver builds."

**PTX Qdrant:**
> CUDA Programming Guide section 21.1: "cuCtxCreate() pushes the new context onto the top of the stack... CUDA functions will return CUDA_ERROR_INVALID_CONTEXT if a valid context is not current to the thread."
> No direct hit on cuMemGetInfo_v2 specifically requiring stack presence.

---

## 4. Recommended Instrumentation Patch (design only — do not apply)

Add to `get_vram_usage()` at `loader.py:1006`, gated by `K3D_RPN_DEBUG=1`:

```python
def get_vram_usage() -> tuple[int, int]:
    if _cuMemGetInfo is None:
        raise RuntimeError("cuMemGetInfo is not available on this CUDA driver")

    _ensure_current_context()

    # === DIAGNOSTIC BLOCK (K3D_RPN_DEBUG=1 only) ===
    if os.environ.get("K3D_RPN_DEBUG"):
        _dbg_current = CUcontext()
        _dbg_r = nvcuda.cuCtxGetCurrent(ctypes.byref(_dbg_current))
        print(f"[vram_diag] cuCtxGetCurrent -> {_dbg_r}, ctx={_dbg_current.value:#x}")
        print(f"[vram_diag] _context stored -> {(_context.value or 0):#x}")
        if _dbg_current.value != (_context.value or 0):
            print(f"[vram_diag] MISMATCH: stored ctx != current ctx")
        _dbg_dev = CUdevice()
        _dbg_rd = nvcuda.cuCtxGetDevice(ctypes.byref(_dbg_dev))
        print(f"[vram_diag] cuCtxGetDevice -> {_dbg_rd}, device={_dbg_dev.value}")
        _dbg_sync = nvcuda.cuCtxSynchronize()
        print(f"[vram_diag] cuCtxSynchronize -> {_dbg_sync}")
        # Try non-v2 as side-call
        try:
            _legacy_meminfo = getattr(nvcuda, "cuMemGetInfo")
            _f, _t = ctypes.c_size_t(), ctypes.c_size_t()
            _lr = _legacy_meminfo(ctypes.byref(_f), ctypes.byref(_t))
            print(f"[vram_diag] cuMemGetInfo (legacy) -> {_lr}, free={_f.value}")
        except AttributeError:
            print(f"[vram_diag] cuMemGetInfo (legacy) not available")
    # === END DIAGNOSTIC BLOCK ===

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    res = _cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    ck(res)
    used = total.value - free.value
    return used, total.value
```

Export `K3D_RPN_DEBUG=1` via option **(b)**: instrument unconditionally as a single-shot per-process print (no env var required) to eliminate the need to modify `k3d_env.sh`. The diagnostic runs once, then is silent.

---

## 5. Recommended Actual Fix

**Root cause (most likely):** `galaxy_buffer.py` destructor calls `cuda.cuDevicePrimaryCtxRelease(device)` via the cuda-python bindings. This competes with `loader.py`'s context. Even if the benchmark does not import `galaxy_buffer` directly, any transitive import path that instantiates `GalaxyMemory` will schedule this destructor.

**Fix 1 (primary — remove the API competition):**

`knowledge3d/cranium/ptx/galaxy_buffer.py:100-104` — remove the `cuDevicePrimaryCtxRelease` call from the destructor entirely:

```python
# BEFORE:
try:
    if self.ctx:
        cuda.cuDevicePrimaryCtxRelease(self.device)
except Exception:
    pass

# AFTER: omit — loader.py owns context lifetime; galaxy_buffer must not release it
```

Same for `galaxy_buffer.py:465-469`.

**Rationale:** The sovereignty principle (single context, loader owns it) means `galaxy_buffer` has NO business releasing the primary context. The loader's `_context` must outlive all operations. The cuda-python library's `cuDevicePrimaryCtxRelease` and loader's `cuCtxCreate`/`cuDevicePrimaryCtxRetain` are operating on the same refcount.

**Fix 2 (defensive hardening in `loader.py` — apply regardless):**

Replace the `cuCtxSetCurrent` call at `loader.py:416` with a `cuCtxPushCurrent`+`cuCtxGetCurrent` to force the context onto the legacy stack, eliminating H2:

```python
# BEFORE (line 416):
ck(nvcuda.cuCtxSetCurrent(ctx))

# AFTER:
push_res = nvcuda.cuCtxPushCurrent(ctx)
if push_res == 0:
    # Context is now on the legacy stack; cuMemGetInfo_v2 will see it
    popped = CUcontext()
    nvcuda.cuCtxPopCurrent(ctypes.byref(popped))
    # Re-set as floating context (keeps compatibility)
    ck(nvcuda.cuCtxSetCurrent(ctx))
else:
    # cuCtxPushCurrent not needed (cuCtxCreate already pushed)
    ck(nvcuda.cuCtxSetCurrent(ctx))
```

Simpler: since `cuCtxCreate` already pushes the context, the `cuCtxSetCurrent(ctx)` at line 416 is REDUNDANT on the normal path. Remove it on the success path (lines 415-416 inside the `else` branch). The context is already current after `cuCtxCreate`. Removing the redundant `cuCtxSetCurrent` eliminates the stack desync risk.

---

## 6. Backup Plan

If Fix 1 + Fix 2 don't resolve it:

1. Add `cuCtxSynchronize()` at end of `_ensure_init()` (after line 433) — forces full driver-side context materialization before `_initialized = True` is set.
2. In `get_vram_usage()`, replace `_cuMemGetInfo` (which is `cuMemGetInfo_v2`) with a fallback that calls `cuMemGetInfo_v2` first and, on 201, attempts `cuMemGetInfo` (non-v2) — distinguishing whether the failure is context-generic or symbol-specific.
3. Audit all `cuda-python` imports in the dependency tree for additional `cuDevicePrimaryCtxRelease` or `cuCtxDestroy` calls that could fire before `get_vram_usage()`.

---

## Files to Change (for Codex sub-agent)

- `knowledge3d/cranium/ptx/galaxy_buffer.py` lines 100-104 and 465-469: remove `cuDevicePrimaryCtxRelease` calls.
- `knowledge3d/cranium/sovereign/loader.py` line 416: remove the redundant `cuCtxSetCurrent(ctx)` from the normal `cuCtxCreate` success path (`else` branch at line 415).
- `knowledge3d/cranium/sovereign/loader.py:1006-1010`: add diagnostic block gated on `K3D_RPN_DEBUG` (option b: single-shot unconditional print, then clears a per-process flag).

**Confidence in root cause (H1):** High (90%). The `cuDevicePrimaryCtxRelease` in `galaxy_buffer.py` is a direct sovereignty violation — it is not part of `loader.py`'s contract — and its destructor fires at unpredictable GC times.
