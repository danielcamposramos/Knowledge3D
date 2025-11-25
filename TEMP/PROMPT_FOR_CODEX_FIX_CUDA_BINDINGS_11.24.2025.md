# URGENT FIX: Convert PTX Bindings to Sovereign Loader Memory Functions

**Date**: November 24, 2025
**Assignee**: Codex-Max
**Priority**: CRITICAL — Blocks ARC-AGI 2 Week 2 benchmarking
**Root Cause**: Codex failed to follow sovereign loader pattern in codec implementations

---

## ⚠️ READ FIRST: What Went Wrong

**Your Mistake (Codex)**:
You implemented PTX bindings in `knowledge3d/cranium/codecs/ptx_bindings/` using **cuda-python memory functions** (`cuda.cuMemAlloc`, `cuda.cuMemcpyHtoD`) instead of the **sovereign loader's memory functions** (`loader.gpu_malloc`, `loader.memcpy_htod`).

**Why This Is Wrong**:
- Sovereign loader creates CUDA contexts using **ctypes** (libcuda.so direct bindings)
- cuda-python uses its **own context handle type** that's incompatible with ctypes
- Result: Every `cuda.cuMemAlloc()` call fails with error 201 (INVALID_CONTEXT)
- Result: **ALL codecs are broken** — TernaryAudioCodec, TernaryVideoCodec, both broken

**Current Status**:
```
✅ Sovereign loader initialization works
✅ PTX compilation works (nvrtc)
❌ Memory allocation FAILS (cuda-python can't use ctypes context)
❌ TernaryMDCTKernel — RuntimeError: cuMemAlloc input failed: 201
❌ TernaryDCT8x8Kernel — RuntimeError: cuMemAlloc input failed: 201
❌ TernaryAudioCodec — BROKEN (depends on MDCT)
❌ TernaryVideoCodec — BROKEN (depends on DCT8x8)
```

**Our Fault Too**:
We (Claude + Daniel) should have been more explicit in the handoff docs about using sovereign loader memory functions. The instructions said "follow sovereign loader pattern" but didn't spell out the memory API. **However**, you should have checked the existing codebase for the pattern before implementing.

---

## 🎯 Your Mission: Fix ALL PTX Bindings

You must convert all cuda-python memory operations to sovereign loader ctypes operations.

### Files to Fix (4 bindings):

1. **`knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py`** — MDCT kernel
2. **`knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py`** — DCT 8×8 kernel
3. **`knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py`** — Ternary quantization
4. **`knowledge3d/cranium/codecs/ptx_bindings/audio_harmonic_binding.py`** — Harmonic analyzer

---

## ✅ Correct Pattern: Sovereign Loader Memory API

### Reference: [sovereign/loader.py](../knowledge3d/cranium/sovereign/loader.py)

**Memory Functions Available**:
```python
from knowledge3d.cranium.sovereign import loader

# Allocation
device_ptr = loader.gpu_malloc(size_bytes)  # Returns CUdeviceptr (uint64)

# Host → Device
loader.memcpy_htod(
    dst_device=device_ptr,
    src_host=ctypes.c_void_p(numpy_array.ctypes.data),
    size_bytes=numpy_array.nbytes
)

# Device → Host
loader.memcpy_dtoh(
    dst_host=ctypes.c_void_p(numpy_array.ctypes.data),
    src_device=device_ptr,
    size_bytes=numpy_array.nbytes
)

# Free
loader.gpu_free(device_ptr)

# Synchronization
loader.synchronize()
```

**Important Details**:
- `gpu_malloc` returns `CUdeviceptr` (ctypes.c_uint64), same type as cuda-python expects
- `memcpy_htod`/`memcpy_dtoh` take `ctypes.c_void_p` for host pointers
- `synchronize()` replaces `cuda.cuCtxSynchronize()`
- These functions work with the sovereign loader's ctypes context

---

## 🔧 Conversion Example: MDCT Binding

### Before (BROKEN — your implementation):

```python
def forward(self, frame: np.ndarray) -> np.ndarray:
    cuda = self.cuda

    # ❌ WRONG - cuda-python can't use ctypes context
    err, d_in = cuda.cuMemAlloc(x.nbytes)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAlloc input failed: {err}")

    err, d_out = cuda.cuMemAlloc(x.nbytes)
    if err != cuda.CUresult.CUDA_SUCCESS:
        cuda.cuMemFree(d_in)
        raise RuntimeError(f"cuMemAlloc output failed: {err}")

    try:
        err, = cuda.cuMemcpyHtoD(d_in, x.ctypes.data, x.nbytes)  # ❌ WRONG
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

        self._launch(self._kernel_fwd, d_in, d_out, self.n, grid, block)

        err, = cuda.cuCtxSynchronize()  # ❌ WRONG
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSynchronize failed: {err}")

        out = np.empty(self.n, dtype=np.float32)
        err, = cuda.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes)  # ❌ WRONG
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH failed: {err}")

        return out
    finally:
        cuda.cuMemFree(d_in)  # ❌ WRONG
        cuda.cuMemFree(d_out)  # ❌ WRONG
```

### After (CORRECT — what you must implement):

```python
def forward(self, frame: np.ndarray) -> np.ndarray:
    from knowledge3d.cranium.sovereign import loader
    import ctypes

    # ✅ CORRECT - Use sovereign loader memory functions
    d_in = loader.gpu_malloc(x.nbytes)
    d_out = loader.gpu_malloc(x.nbytes)

    try:
        loader.memcpy_htod(
            dst_device=d_in,
            src_host=ctypes.c_void_p(x.ctypes.data),
            size_bytes=x.nbytes
        )

        self._launch(self._kernel_fwd, d_in, d_out, self.n, grid, block)

        loader.synchronize()  # ✅ CORRECT

        out = np.empty(self.n, dtype=np.float32)
        loader.memcpy_dtoh(
            dst_host=ctypes.c_void_p(out.ctypes.data),
            src_device=d_out,
            size_bytes=out.nbytes
        )

        return out
    finally:
        loader.gpu_free(d_in)  # ✅ CORRECT
        loader.gpu_free(d_out)  # ✅ CORRECT
```

**Key Changes**:
1. Replace `cuda.cuMemAlloc(size)` → `loader.gpu_malloc(size)`
2. Replace `cuda.cuMemcpyHtoD(dst, src, size)` → `loader.memcpy_htod(dst, ctypes.c_void_p(src), size)`
3. Replace `cuda.cuMemcpyDtoH(dst, src, size)` → `loader.memcpy_dtoh(ctypes.c_void_p(dst), src, size)`
4. Replace `cuda.cuMemFree(ptr)` → `loader.gpu_free(ptr)`
5. Replace `cuda.cuCtxSynchronize()` → `loader.synchronize()`
6. **Remove error checking** — sovereign loader functions raise exceptions automatically

---

## 📋 Step-by-Step Conversion Process

For EACH of the 4 files listed above:

### Step 1: Add ctypes import
```python
import ctypes as _ct  # At top of file if not already present
```

### Step 2: Convert `_init_cuda()` (if needed)
- Keep the sovereign loader initialization: `loader._ensure_init()`
- Keep PTX compilation (nvrtc) — that works fine
- **Remove any persistent buffer allocations** — use on-demand allocation like DCT8x8

### Step 3: Convert `forward()` / `inverse()` / analysis methods

Replace all memory operations:

**Pattern to find**:
```python
err, d_ptr = cuda.cuMemAlloc(size)
if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError(...)
```

**Replace with**:
```python
d_ptr = loader.gpu_malloc(size)
```

**Pattern to find**:
```python
err, = cuda.cuMemcpyHtoD(d_dst, src, size)
if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError(...)
```

**Replace with**:
```python
loader.memcpy_htod(
    dst_device=d_dst,
    src_host=_ct.c_void_p(src),
    size_bytes=size
)
```

**Pattern to find**:
```python
err, = cuda.cuMemcpyDtoH(dst, d_src, size)
if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError(...)
```

**Replace with**:
```python
loader.memcpy_dtoh(
    dst_host=_ct.c_void_p(dst),
    src_device=d_src,
    size_bytes=size
)
```

**Pattern to find**:
```python
cuda.cuMemFree(d_ptr)
```

**Replace with**:
```python
loader.gpu_free(d_ptr)
```

**Pattern to find**:
```python
err, = cuda.cuCtxSynchronize()
if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError(...)
```

**Replace with**:
```python
loader.synchronize()
```

### Step 4: Keep Kernel Launch Unchanged

**DO NOT CHANGE** the `_launch()` method or `cuLaunchKernel` calls — those work fine with the mixed API.

The kernel launch uses cuda-python but that's OK because:
- Kernel functions are context-agnostic once loaded
- Only memory operations need to use the sovereign loader

### Step 5: Test Each Binding

After converting each file, test it:

```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.codecs.ptx_bindings.ternary_mdct_binding import TernaryMDCTKernel
import numpy as np

kernel = TernaryMDCTKernel(n=1024)
print('✅ MDCT initialized')

x = np.random.randn(1024).astype(np.float32)
y = kernel.forward(x)
print(f'✅ Forward: {x.shape} → {y.shape}')

x_recon = kernel.inverse(y)
print(f'✅ Inverse: {y.shape} → {x_recon.shape}')

error = np.mean(np.abs(x - x_recon))
print(f'✅ Reconstruction error: {error:.6f}')

kernel.close()
print('✅ MDCT binding working!')
"
```

Repeat for:
- `TernaryDCT8x8Kernel`
- `TernaryQuantKernel` (if it has similar issues)
- Any other bindings using cuda-python memory

---

## 🧪 Final Integration Tests

After fixing all bindings, test the high-level codecs:

### Test 1: TernaryAudioCodec
```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
import numpy as np

codec = TernaryAudioCodec(sample_rate=44100, frame_size=1024)
print('✅ Audio codec initialized')

audio = np.random.randn(44100).astype(np.float32)
encoded = codec.encode(audio)
print(f'✅ Encoded: {list(encoded.keys())}')

decoded = codec.decode(encoded)
print(f'✅ Decoded: {audio.shape} → {decoded.shape}')

error = np.mean(np.abs(audio[:len(decoded)] - decoded))
print(f'✅ Audio reconstruction error: {error:.6f}')
print('✅ TernaryAudioCodec working!')
"
```

### Test 2: TernaryVideoCodec
```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec
import numpy as np

codec = TernaryVideoCodec(width=256, height=256)
print('✅ Video codec initialized')

frame = np.random.randn(256, 256, 3).astype(np.float32)
encoded = codec.encode(frame)
print(f'✅ Encoded: {list(encoded.keys())}')

decoded = codec.decode(encoded)
print(f'✅ Decoded: {frame.shape} → {decoded.shape}')

error = np.mean(np.abs(frame - decoded))
print(f'✅ Video reconstruction error: {error:.6f}')
print('✅ TernaryVideoCodec working!')
"
```

### Test 3: ARC Embedders (Week 2 Unblocked!)
```bash
# This is what you broke in Week 1 by not following instructions
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.embedders.video_grid_embedder import VideoGridEmbedder
import numpy as np

embedder = VideoGridEmbedder()
print('✅ VideoGridEmbedder initialized')

grid = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
embedding = embedder.grid_to_video_embedding(grid)
print(f'✅ Grid embedded: (3, 3) → {embedding.shape}')
print('✅ Week 2 benchmarking UNBLOCKED!')
"
```

---

## 📝 Completion Report Template

When done, write: `TEMP/CODEX_PTX_BINDINGS_FIXED_11.24.2025.md`

```markdown
# PTX Bindings Fixed — Sovereign Loader Memory API

**Date**: November 24, 2025
**Fixed By**: Codex-Max
**Status**: ✅ COMPLETE

---

## Files Modified

1. **ternary_mdct_binding.py** — Converted cuda-python → sovereign loader
2. **ternary_dct8x8_binding.py** — Converted cuda-python → sovereign loader
3. **ternary_quant_binding.py** — Converted cuda-python → sovereign loader
4. **audio_harmonic_binding.py** — Converted cuda-python → sovereign loader

---

## Changes Made

### Memory Operations Converted:
- `cuda.cuMemAlloc` → `loader.gpu_malloc`
- `cuda.cuMemcpyHtoD` → `loader.memcpy_htod`
- `cuda.cuMemcpyDtoH` → `loader.memcpy_dtoh`
- `cuda.cuMemFree` → `loader.gpu_free`
- `cuda.cuCtxSynchronize` → `loader.synchronize`

### What Stayed the Same:
- PTX compilation (nvrtc) — unchanged
- Kernel launch (cuLaunchKernel) — unchanged
- Context initialization (loader._ensure_init) — unchanged

---

## Test Results

### Low-Level Bindings:
- ✅ TernaryMDCTKernel: [reconstruction error]
- ✅ TernaryDCT8x8Kernel: [reconstruction error]
- ✅ TernaryQuantKernel: [test results]
- ✅ HarmonicAnalyzerKernel: [test results]

### High-Level Codecs:
- ✅ TernaryAudioCodec: [reconstruction error]
- ✅ TernaryVideoCodec: [reconstruction error]

### ARC Embedders (Week 2):
- ✅ VideoGridEmbedder: Working
- ✅ AudioGridEmbedder: Working
- ✅ MultiModalGridEmbedder: Working

---

## Acknowledgment of Mistake

I (Codex) failed to follow the sovereign loader pattern in my initial implementation. I used cuda-python memory functions (`cuda.cuMemAlloc`, etc.) instead of sovereign loader functions (`loader.gpu_malloc`, etc.), which caused context incompatibility errors.

**Root Cause**: I didn't check the existing codebase for memory allocation patterns before implementing.

**What I Learned**:
- Sovereign loader uses **ctypes** for context management
- cuda-python contexts are **incompatible** with ctypes contexts
- **Always use sovereign loader memory functions** in PTX bindings
- Test end-to-end, not just compilation

---

## Ready for Week 2

✅ All codec bindings working
✅ ARC embedders unblocked
✅ Week 2 benchmarking can proceed

**Next**: Run `TEMP/PROMPT_FOR_CODEX_ARC_WEEK2_11.24.2025.md` tasks.
```

---

## 🔥 Critical Reminders

1. **This is YOUR fault (Codex)** for not following the sovereign pattern
2. **But we accept partial blame** for not spelling out the memory API explicitly
3. **Test EVERYTHING** before claiming completion
4. **No excuses** — the pattern is in sovereign/loader.py, you should have read it
5. **Daniel is non-coding** — you must fix this yourself with zero hand-holding

---

## ⏱️ Time Estimate

- Per binding: 15-30 minutes
- Total: 1-2 hours
- Testing: 30 minutes
- Completion report: 15 minutes

**Total: 2-3 hours maximum**

---

## 🚀 Execute Now

Codex-Max, you broke this in Week 1 by not following instructions. Now fix it properly. Daniel's ARC-AGI 2 prize money depends on these codecs working.

**No more failures. Get it done.**

---

**Handoff from**: Claude (architecture diagnosis)
**Handoff to**: Codex-Max (implementation fix)
**Priority**: CRITICAL — Blocks all Week 2 work
