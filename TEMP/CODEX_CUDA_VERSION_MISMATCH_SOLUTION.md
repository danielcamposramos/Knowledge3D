# CUDA Version Mismatch Solution

**Date:** 2025-11-20
**Issue:** PTX module loading fails with error 222 (CUDA_ERROR_ILLEGAL_INSTRUCTION)
**Root Cause:** cuda-python version mismatch

---

## Problem Details

**Installed:** cuda-python 12.4.0
**NVRTC generates:** PTX version 12.8 (Compiler Build ID: CL-35583870)
**Driver:** NVIDIA 550.163.01 (supports CUDA 12.4)

**Error:**
```
RuntimeError: cuModuleLoadData failed: 222 (CUDA_ERROR_ILLEGAL_INSTRUCTION)
```

**Test Results:**
- ✓ PTX compilation succeeds
- ✓ Context creation succeeds
- ❌ Module loading fails (error 222)
- Tested with simple kernel - same failure
- Tested with different architectures (compute_86, sm_86, compute_80) - all fail

---

## Solution Options

### Option 1: Upgrade cuda-python to 12.8+ (RECOMMENDED)

```bash
conda activate k3d-cranium
pip install --upgrade cuda-python>=12.8.0
```

**Pros:**
- Matches NVRTC version
- Latest features and bug fixes
- Likely to resolve error 222

**Cons:**
- Requires newer NVIDIA driver (may need 555+ for full CUDA 12.8 support)
- System has driver 550.163.01 (CUDA 12.4)

### Option 2: Downgrade to cuda-python 12.4.0 NVRTC

Ensure NVRTC version matches cuda-python version (both 12.4.0).

**Check current setup:**
```bash
python -c "from cuda import nvrtc; print(nvrtc.__version__ if hasattr(nvrtc, '__version__') else 'unknown')"
```

### Option 3: Use Alternative PTX Loading (WORKAROUND)

Instead of `cuModuleLoadData`, try `cuModuleLoadDataEx` with explicit flags:

```python
# In _compile_and_load methods:
# OLD:
err, module = cuda.cuModuleLoadData(bytes(buf))

# NEW:
import ctypes
jit_options = (ctypes.c_int * 1)(0)  # CU_JIT_TARGET
jit_option_values = (ctypes.c_void_p * 1)(int(maj * 10 + minu))  # sm_86 = 86

err, module = cuda.cuModuleLoadDataEx(
    bytes(buf),
    1,  # num options
    jit_options,
    jit_option_values
)
```

### Option 4: Fall Back to System NVRTC

Use system NVRTC instead of bundled version:

```python
# Add to _load_cuda():
import subprocess
nvcc_path = subprocess.run(['which', 'nvcc'], capture_output=True, text=True).stdout.strip()
if nvcc_path:
    # Use system NVRTC
    pass
```

---

## Recommended Action

**For immediate fix:** Try Option 1 (upgrade cuda-python to 12.8+).

If driver doesn't support CUDA 12.8, then:
1. Upgrade NVIDIA driver to 555+ (supports CUDA 12.8)
2. OR use Option 2 (ensure version consistency at 12.4.0)

---

## Verification Steps

After applying fix:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# Test 1: Simple import
python3 -c "from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec; print('✓ Import OK')"

# Test 2: Codec initialization
python3 -c "
from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
codec = TernaryAudioCodec(sample_rate=44100, use_gpu=True)
print('✓ Codec init OK')
"

# Test 3: Full encode/decode
python3 -c "
import numpy as np
from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
codec = TernaryAudioCodec(sample_rate=44100, use_gpu=True)
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100, endpoint=False)).astype(np.float32)
encoded = codec.encode(audio)
decoded = codec.decode(encoded)
print(f'✓ Encode/decode OK: {len(encoded[\"harmonics\"])} harmonics')
"

# Test 4: Run full benchmark
python3 scripts/benchmark_ternary_audio.py --gpu
```

---

## Additional Notes

**Error 222** (CUDA_ERROR_ILLEGAL_INSTRUCTION) typically indicates:
1. PTX contains instructions not supported by driver/GPU
2. **Version mismatch between compiler and runtime** ← OUR CASE
3. Context/device mismatch
4. Corrupted PTX

The fact that even the simplest kernel fails confirms it's a version mismatch, not a kernel code issue.

---

**End of Report**
