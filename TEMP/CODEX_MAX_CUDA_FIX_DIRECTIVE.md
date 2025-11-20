# Codex-Max: CUDA PTX Binding Fix Directive

**Date**: 2025-11-20
**Issue**: cuModuleLoadData failing with error 222 (CUDA_ERROR_ILLEGAL_ADDRESS)
**Status**: Critical — GPU path blocked

---

## 🔍 PROBLEM DIAGNOSIS

**Current Error**:
```
RuntimeError: cuModuleLoadData failed: 222
```

**Error 222** = `CUDA_ERROR_ILLEGAL_ADDRESS` — PTX data pointer is invalid or module loading API usage is incorrect.

**Environment**:
- cuda-python: 13.0.3 (just installed via pip)
- CUDA device: Accessible (cuInit=0, cuDeviceGet=0 success)
- CUDA_VISIBLE_DEVICES=0: Set correctly
- RTX 3060, sm_86 Ampere

**Root Cause**: The `cuModuleLoadData` call in `ternary_mdct_binding.py` line 146 is failing because:
1. cuda-python 13.0 uses a different API structure
2. PTX data needs proper ctypes conversion
3. Module loading may need updated parameters

---

## 🎯 FIX STRATEGY

### Step 1: Use Modern cuda-python API

**File**: `knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py`

**Current code** (lines 146-148):
```python
err, module = cuda.cuModuleLoadData(bytes(buf))
if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError(f"cuModuleLoadData failed: {err}")
```

**Problem**: `cuModuleLoadData` expects a proper ctypes pointer, not raw bytes.

**Fix**:
```python
# Create ctypes buffer from PTX
ptx_data = bytes(buf)
ptx_ctypes = (ctypes.c_char * len(ptx_data)).from_buffer_copy(ptx_data)

# Load module with proper pointer
err, module = cuda.cuModuleLoadDataEx(
    ctypes.byref(ptx_ctypes),  # Proper pointer
    0,  # numOptions
    [],  # options (CUjit_option array)
    []   # optionValues (void* array)
)
```

**Or simpler** (if cuModuleLoadData signature is correct):
```python
# Use cuda.bindings.driver API directly
from cuda.bindings import driver as cuda_driver
from cuda.bindings import nvrtc as cuda_nvrtc

# In module loading:
ptx_bytes = bytes(buf)
err, module = cuda_driver.cuModuleLoadData(ptx_bytes)
```

### Step 2: Fix Kernel Launch Arguments

**Current code** (lines 159-188):
```python
def _launch(self, func, args_list, grid, block) -> None:
    cuda = self.cuda
    arg_objs = []
    arg_ptrs = (ctypes.c_void_p * len(args_list))()
    for i, a in enumerate(args_list):
        if isinstance(a, float):
            obj = ctypes.c_float(a)
        elif isinstance(a, int):
            obj = ctypes.c_int(a)
        else:
            obj = ctypes.c_void_p(int(a))
        arg_objs.append(obj)
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(obj), ctypes.c_void_p)
    err, = cuda.cuLaunchKernel(
        func,
        int(grid[0]), int(grid[1]), int(grid[2]),
        int(block[0]), int(block[1]), int(block[2]),
        0,  # sharedMemBytes
        0,  # hStream
        arg_ptrs,  # kernelParams
        None,  # extra
    )
```

**Problem**: `cuLaunchKernel` expects `void**` (pointer to pointers), but ctypes needs special handling.

**Fix**:
```python
def _launch(self, func, args_list, grid, block) -> None:
    """Launch kernel with proper argument marshalling for cuda-python 13.0."""
    cuda = self.cuda

    # Build argument array
    # cuda-python expects a list of ctypes pointers
    kernel_args = []
    for arg in args_list:
        if isinstance(arg, int):
            # Device pointers: pass as-is (they're already int handles)
            if arg > 0xFFFFFFFF:  # Likely a device pointer
                ptr_obj = ctypes.c_void_p(arg)
                kernel_args.append(ctypes.addressof(ptr_obj))
            else:
                # Small int: kernel parameter
                int_obj = ctypes.c_int32(arg)
                kernel_args.append(ctypes.addressof(int_obj))
        elif isinstance(arg, float):
            float_obj = ctypes.c_float(arg)
            kernel_args.append(ctypes.addressof(float_obj))
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")

    # Convert to void** (array of pointers)
    param_ptrs = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

    err, = cuda.cuLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0,  # sharedMemBytes
        0,  # hStream (NULL stream)
        param_ptrs,
        None  # extra
    )
    if err != 0:  # cuda.CUresult.CUDA_SUCCESS
        raise RuntimeError(f"cuLaunchKernel failed with error {err}")
```

### Step 3: Simplify with K3D's Existing PTX Runtime

**Alternative approach**: Use K3D's existing `knowledge3d/cranium/ptx_runtime/` infrastructure instead of reinventing.

**Check if these exist**:
- `knowledge3d/cranium/ptx_runtime/cuda_context.py`
- `knowledge3d/cranium/ptx_runtime/ptx_loader.py`

**If they exist**, use them:
```python
from knowledge3d.cranium.ptx_runtime.cuda_context import get_cuda_context
from knowledge3d/cranium/ptx_runtime.ptx_loader import compile_cuda_to_ptx, load_ptx_kernel

class TernaryMDCTKernel:
    def __init__(self, n: int = 1024):
        self.n = n
        self.ctx = get_cuda_context(device_id=0)

        # Compile CUDA to PTX using K3D's runtime
        ptx_code = compile_cuda_to_ptx(MDCT_KERNEL_SRC, arch='sm_86')

        # Load kernel functions
        self.kernel_fwd = load_ptx_kernel(ptx_code, 'ternary_mdct_forward')
        self.kernel_inv = load_ptx_kernel(ptx_code, 'ternary_mdct_inverse')

        # Allocate GPU buffers
        self.d_in = self.ctx.mem_alloc(n * 4)
        self.d_out = self.ctx.mem_alloc(n * 4)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        x = frame.astype(np.float32)
        self.ctx.memcpy_htod(self.d_in, x)

        # Launch using K3D's launch helper
        block = (256, 1, 1)
        grid = ((self.n + 255) // 256, 1, 1)
        self.kernel_fwd(
            [self.d_in, np.int32(self.n), self.d_out],
            block=block,
            grid=grid
        )

        result = np.empty(self.n, dtype=np.float32)
        self.ctx.memcpy_dtoh(result, self.d_out)
        return result
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Option A: Fix cuda-python 13.0 Direct Usage

- [ ] Update `_load_cuda()` to use `cuda.bindings.driver` consistently
- [ ] Fix `cuModuleLoadData` to use proper ctypes pointer
- [ ] Fix `_launch()` argument marshalling for cuLaunchKernel
- [ ] Test with simple kernel first (print kernel, no computation)
- [ ] Validate MDCT forward/inverse

### Option B: Use K3D's Existing PTX Runtime (RECOMMENDED)

- [ ] Check if `knowledge3d/cranium/ptx_runtime/cuda_context.py` exists
- [ ] Check if `knowledge3d/cranium/ptx_runtime/ptx_loader.py` exists
- [ ] Read existing K3D PTX runtime API
- [ ] Rewrite MDCT binding to use K3D's runtime
- [ ] Rewrite DCT 8×8 binding to use K3D's runtime
- [ ] Test with existing K3D tests

---

## 🚀 EXECUTION PRIORITY

**IMMEDIATE** (Fix GPU path):
1. Investigate K3D's existing PTX runtime (`knowledge3d/cranium/ptx_runtime/`)
2. If it exists and works, use it (Option B)
3. If not, fix cuda-python 13.0 API usage (Option A)
4. Test MDCT kernel launch with simple print kernel
5. Validate full MDCT forward/inverse cycle

**NEXT** (Once GPU launches):
6. Apply same fix to DCT 8×8 binding
7. Re-run audio benchmark with `use_gpu=True`
8. Re-run video benchmark with `use_gpu=True`
9. Validate <100ms audio encode/decode
10. Add more realistic video frames for PSNR validation

**THEN** (Integration):
11. Complete RPN/Galaxy integration
12. End-to-end cross-modal tests
13. Memory budget verification

---

## 🔧 DIAGNOSTIC COMMANDS

**Test CUDA access**:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -c "
from cuda.bindings import driver as cuda
err, = cuda.cuInit(0)
print(f'cuInit: {err}')
err, dev = cuda.cuDeviceGet(0)
print(f'Device: {err}')
"
```

**Test simple PTX kernel**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -c "
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc

# Simple kernel that just prints
src = b'''
extern \"C\" __global__ void test_kernel() {
    printf(\"Hello from GPU thread %d\\n\", threadIdx.x);
}
'''

# Init
cuda.cuInit(0)
err, dev = cuda.cuDeviceGet(0)
err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)
cuda.cuCtxSetCurrent(ctx)

# Compile
res, prog = nvrtc.nvrtcCreateProgram(src, b'test.cu', 0, [], [])
res, = nvrtc.nvrtcCompileProgram(prog, 0, [])
res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
buf = bytearray(ptx_size)
nvrtc.nvrtcGetPTX(prog, buf)
nvrtc.nvrtcDestroyProgram(prog)

# Load
err, module = cuda.cuModuleLoadData(bytes(buf))
print(f'Module load: {err}')

if err == 0:
    err, func = cuda.cuModuleGetFunction(module, b'test_kernel')
    print(f'Get function: {err}')

    if err == 0:
        # Launch with no args
        err, = cuda.cuLaunchKernel(func, 1, 1, 1, 4, 1, 1, 0, 0, None, None)
        print(f'Launch: {err}')
        cuda.cuCtxSynchronize()
"
```

**Check K3D PTX runtime**:
```bash
ls -la knowledge3d/cranium/ptx_runtime/
grep -r "def load_ptx" knowledge3d/cranium/ptx_runtime/
```

---

**Codex-Max, prioritize Option B (use K3D's PTX runtime) if it exists. Otherwise implement Option A (fix cuda-python 13.0 API). NO CPU FALLBACKS — WE FIX OR WE FIX!** 🚀
