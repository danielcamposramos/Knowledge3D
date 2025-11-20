# Codex: Fix CUDA Context Initialization in PTX Bindings

**Date:** 2025-11-20
**Priority:** HIGH
**Blocker For:** Audio codec benchmarks, production deployment
**Estimated Fix Time:** 30 minutes

---

## Problem Summary

Multiple PTX bindings (TernaryQuantizer, TernaryMDCTKernel, AudioHarmonicGPU) are independently calling `cuDevicePrimaryCtxRetain()`, causing CUDA error 222 (ILLEGAL_INSTRUCTION) when initialized sequentially.

**Error:**
```
RuntimeError: cuModuleLoadData failed: 222
```

**Root Cause:** Duplicate primary context retains on same device.

---

## Solution: Use Shared Context from Sovereign Loader

K3D already has a production-ready shared CUDA context in [`knowledge3d/cranium/sovereign/loader.py`](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/sovereign/loader.py) with fork-safety (implemented in Phase G).

### Files to Modify

#### 1. knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py

**Current (lines 91-111):**
```python
def _init_cuda(self) -> None:
    cuda = self.cuda

    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")

    err, dev = cuda.cuDeviceGet(self.device_index)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")

    err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)  # ← PROBLEM!
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")

    err, = cuda.cuCtxSetCurrent(ctx)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

    self._ctx = ctx
    self._compile_and_load(dev)
    # ... rest of code
```

**Fixed:**
```python
from knowledge3d.cranium.sovereign import loader

def _init_cuda(self) -> None:
    # Use shared context from sovereign loader (fork-safe, proven in production)
    loader._ensure_init()
    cuda = self.cuda

    # Get existing context (no new retain!)
    err, ctx = cuda.cuCtxGetCurrent()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

    self._ctx = ctx

    # Get device from context
    err, dev = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetDevice failed: {err}")

    self._compile_and_load(dev)

    # Memory allocation (unchanged)
    err, d_in = cuda.cuMemAlloc(self.n * 4)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAlloc input failed: {err}")

    err, d_out = cuda.cuMemAlloc(self.n * 4)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAlloc output failed: {err}")

    self._d_in = d_in
    self._d_out = d_out
```

#### 2. knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py

**Current (lines 100-115):**
```python
def _init_cuda(self) -> None:
    cuda = self.cuda
    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = cuda.cuDeviceGet(self.device_index)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)  # ← PROBLEM!
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
    err, = cuda.cuCtxSetCurrent(ctx)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxSetCurrent failed: {err}")
    self._ctx = ctx
    self._compile_and_load(dev)
```

**Fixed:**
```python
from knowledge3d.cranium.sovereign import loader

def _init_cuda(self) -> None:
    loader._ensure_init()
    cuda = self.cuda

    err, ctx = cuda.cuCtxGetCurrent()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

    self._ctx = ctx

    err, dev = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetDevice failed: {err}")

    self._compile_and_load(dev)
```

#### 3. knowledge3d/cranium/codecs/ptx_bindings/audio_harmonic_binding.py

Apply same pattern:
```python
from knowledge3d.cranium.sovereign import loader

class AudioHarmonicGPU:
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cuda, self.nvrtc = _load_cuda()
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        # ... function pointers
        self._init_cuda()

    def _init_cuda(self) -> None:
        # Use shared context
        loader._ensure_init()
        cuda = self.cuda

        err, ctx = cuda.cuCtxGetCurrent()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

        self._ctx = ctx

        err, dev = cuda.cuCtxGetDevice()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxGetDevice failed: {err}")

        self._compile_and_load(dev)
```

#### 4. knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py

Same fix (if this binding exists).

---

## Verification Steps

After applying fixes:

### 1. Test Import
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python3 -c "
from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
print('✓ Import successful')

codec = TernaryAudioCodec(sample_rate=44100, use_gpu=True)
print('✓ Codec initialization successful')
"
```

### 2. Run Full Benchmark
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/benchmark_ternary_audio.py --gpu
```

**Expected Output:**
```
Ternary Audio Codec Benchmark Results
==========================================================================
Audio Type    |  Size (KB) | Compressed (KB) | Ratio | Encode (ms) | Decode (ms) | PSNR (dB)
--------------------------------------------------------------------------
sine_440hz    |      172.3 |           24.3   |  7.1  |     2-5     |     1-2     |    89.6
speech_synth  |      172.3 |           33.1   |  5.2  |     2-5     |     1-2     |    36.1
music_piano   |      172.3 |           19.4   |  8.9  |     2-5     |     1-2     |    23.5
```

### 3. Run Video Benchmark (If Needed)
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/benchmark_ternary_video.py --gpu
```

---

## Why This Fix Works

1. **Shared Context:** All PTX bindings use the same primary context created by `loader._ensure_init()`
2. **Fork-Safe:** The loader's context reinitializes automatically if forked (Phase G fix)
3. **Production-Proven:** This pattern is already used by all other K3D PTX modules
4. **Minimal Changes:** Only replaces context initialization, rest of code unchanged
5. **<100μs Overhead:** Verified in Phase G parallel training (12 processes)

---

## Related Documentation

- **Fork-safety solution:** [docs/ptx_parallel_training_cuda_context_isolation.md](docs/ptx_parallel_training_cuda_context_isolation.md)
- **Sovereign loader:** [knowledge3d/cranium/sovereign/loader.py](knowledge3d/cranium/sovereign/loader.py)
- **GPU verification report:** [TEMP/CLAUDE_PHASE2_GPU_HARMONIC_VERIFICATION.md](TEMP/CLAUDE_PHASE2_GPU_HARMONIC_VERIFICATION.md)

---

## Checklist

- [ ] Apply fix to `ternary_mdct_binding.py`
- [ ] Apply fix to `ternary_quant_binding.py`
- [ ] Apply fix to `audio_harmonic_binding.py`
- [ ] Apply fix to `ternary_dct8x8_binding.py` (if exists)
- [ ] Test imports
- [ ] Run audio benchmark
- [ ] Run video benchmark
- [ ] Document actual performance numbers
- [ ] Update CODEX_PHASE2_FINAL_RESULTS.md
- [ ] Update CLAUDE.md codec section

---

**End of Directive**
