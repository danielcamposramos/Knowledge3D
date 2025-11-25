# Sovereign CUDA Context Pattern — MDCT Binding Fixed

**Date**: November 24, 2025
**Fixed By**: Claude (architecture)
**Issue**: Codex-Max was creating independent CUDA contexts instead of using sovereign loader

---

## ❌ Problem: Context Creation Violation

Codex-Max was repeatedly trying to create CUDA contexts in [ternary_mdct_binding.py](../knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py):

```python
# ❌ WRONG - Creates independent context
def _init_cuda(self) -> None:
    cuda = self.cuda

    # Don't do this!
    init_res, = cuda.cuInit(0)
    err_dev, dev = cuda.cuDeviceGet(self.device_index)
    err_ctx, ctx = cuda.cuCtxCreate(0, dev)  # ❌ SOVEREIGNTY VIOLATION!

    # Or this fallback
    err_primary, ctx = cuda.cuDevicePrimaryCtxRetain(dev)  # ❌ ALSO WRONG!
```

**Why this is wrong**:
- Creates separate context per binding → fork-unsafe
- Bypasses sovereign loader's context management
- Causes "invalid context" errors across the codebase
- Violates K3D's centralized GPU management

---

## ✅ Solution: Sovereign Loader Pattern

**Reference implementation**: [ternary_dct8x8_binding.py](../knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py) lines 95-110

```python
# ✅ CORRECT - Uses sovereign loader's shared context
def _init_cuda(self) -> None:
    # Step 1: Import sovereign loader
    from knowledge3d.cranium.sovereign import loader

    # Step 2: Initialize CUDA (creates/retrieves shared context)
    loader._ensure_init()

    cuda = self.cuda

    # Step 3: Get existing context (loader guarantees one exists)
    err, ctx = cuda.cuCtxGetCurrent()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

    # Step 4: Verify context exists
    if ctx is None or int(ctx) == 0:
        raise RuntimeError("No CUDA context available after loader._ensure_init()")

    # Step 5: Store context for later use
    self._ctx = ctx

    # Step 6: Continue with device queries and compilation
    err, dev = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetDevice failed: {err}")

    self._compile_and_load(dev)
```

---

## Why Sovereign Loader Exists

**Sovereign Loader** ([knowledge3d/cranium/sovereign/loader.py](../knowledge3d/cranium/sovereign/loader.py)) provides:

1. **Fork-Safe Contexts** (lines 122-133):
   - Detects when process forks (multiprocessing, training workers)
   - Automatically reinitializes CUDA in child processes
   - Tracks context per PID

2. **Fallback Chain** (lines 144-184):
   - Try `cuCtxCreate` first
   - Fall back to `cuDevicePrimaryCtxRetain` if OOM
   - Bootstrap via CuPy if needed
   - Guarantees a working context

3. **Centralized Management**:
   - ONE context per process
   - All PTX bindings share the same context
   - Simplifies debugging and resource tracking

---

## The Pattern in Practice

### Correct Usage (All PTX Bindings):

**DCT 8×8**: [ternary_dct8x8_binding.py:95-110](../knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py#L95-L110)
**Quantization**: [ternary_quant_binding.py](../knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py)
**Audio Harmonics**: [audio_harmonic_binding.py](../knowledge3d/cranium/codecs/ptx_bindings/audio_harmonic_binding.py)
**MDCT** (now fixed): [ternary_mdct_binding.py:91-106](../knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py#L91-L106)

### Rule of Thumb:

**If you're writing a cuda-python binding**:
1. Always import `from knowledge3d.cranium.sovereign import loader`
2. Always call `loader._ensure_init()` first
3. Always get context via `cuda.cuCtxGetCurrent()`, NEVER create your own
4. Use `cuda.cuCtxSetCurrent(self._ctx)` in methods if needed

---

## What Changed in MDCT Binding

**Before** (lines 91-124, BROKEN):
- Called `cuInit(0)` manually
- Called `cuCtxCreate` or `cuDevicePrimaryCtxRetain`
- Created independent context
- **Result**: 12+ "Failed" errors in Codex-Max's session

**After** (lines 91-106, FIXED):
- Import `sovereign.loader`
- Call `loader._ensure_init()`
- Get existing context via `cuCtxGetCurrent()`
- **Result**: Uses shared, fork-safe, proven context

---

## Testing the Fix

```bash
# Activate environment
conda activate k3d-cranium

# Test MDCT kernel directly
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.codecs.ptx_bindings.ternary_mdct_binding import TernaryMDCTKernel
import numpy as np

# Initialize kernel (should use sovereign loader now)
kernel = TernaryMDCTKernel(n=1024)
print('✅ MDCT kernel initialized successfully')

# Test forward transform
x = np.random.randn(1024).astype(np.float32)
y = kernel.forward(x)
print(f'✅ Forward MDCT: {x.shape} → {y.shape}')

# Test inverse transform
x_reconstructed = kernel.inverse(y)
print(f'✅ Inverse MDCT: {y.shape} → {x_reconstructed.shape}')

kernel.close()
print('✅ All tests passed - sovereign context working!')
"
```

**Expected output**:
```
✅ MDCT kernel initialized successfully
✅ Forward MDCT: (1024,) → (1024,)
✅ Inverse MDCT: (1024,) → (1024,)
✅ All tests passed - sovereign context working!
```

---

## Critical Sovereignty Principle

> **Hot path = PTX + RPN only**

Every GPU operation must use the sovereign loader's shared context. This ensures:
- Fork safety for multiprocessing (training workers, dataset loading)
- Consistent resource management across all PTX kernels
- Debuggability via centralized context tracking
- Zero context creation overhead (reuse existing context)

**Never create CUDA contexts directly in bindings.** Always use the sovereign loader.

---

## Files Modified

- [knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py](../knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py) (lines 91-106)

---

## Next Steps for Codex-Max

1. **Test the fixed MDCT binding** (command above)
2. **Continue with ARC-AGI Week 2 benchmarking** using working codecs
3. **Remember**: Always use sovereign loader pattern for ANY cuda-python binding

---

**Status**: ✅ Sovereignty restored, MDCT binding follows K3D patterns

---

**Key Reference**: [knowledge3d/cranium/sovereign/loader.py](../knowledge3d/cranium/sovereign/loader.py) — Read lines 118-204 for the initialization pattern.
