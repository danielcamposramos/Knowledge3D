#!/usr/bin/env python3
"""
Test original RPN kernel (modular_rpn_kernel.ptx) to establish baseline.
This should work if loader API is correct.
"""
import numpy as np
from pathlib import Path
from knowledge3d.cranium.sovereign import loader
import ctypes

print("=" * 70)
print("Testing Original RPN Kernel (Baseline)")
print("=" * 70)

# Load ORIGINAL kernel (not extended)
ptx_path = Path("knowledge3d/cranium/ptx/modular_rpn_kernel.ptx")
if not ptx_path.exists():
    print(f"❌ Kernel not found: {ptx_path}")
    exit(1)

print(f"\n[1/6] Loading kernel: {ptx_path}")
try:
    module = loader.load_module_from_file(str(ptx_path))
    kernel = loader.get_function(module, "modular_rpn_geometric_kernel")
    print(f"✓ Kernel loaded successfully")
except Exception as e:
    print(f"❌ Failed to load kernel: {e}")
    exit(1)

# Test simple DOT product program
# Opcode: 0x30 (DOT)
# Expected: dot([1,2,3], [4,5,6]) = 32.0
print("\n[2/6] Preparing test program (DOT product)")
op_codes = np.array([0x30], dtype=np.uint16)  # DOT opcode
scalars = np.array([], dtype=np.float32)
vectors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)  # Two 3D vectors

print(f"  Opcode: 0x30 (DOT)")
print(f"  Vectors: [1,2,3] · [4,5,6]")
print(f"  Expected result: 32.0")

# Allocate GPU memory
print("\n[3/6] Allocating GPU memory")
try:
    state_size = 15 * 1040  # 15 instances × 1040 bytes
    state_buffer = loader.gpu_malloc(state_size)
    zeros = np.zeros(state_size, dtype=np.uint8)
    loader.memcpy_htod(state_buffer, zeros.ctypes.data, state_size)

    op_codes_gpu = loader.gpu_malloc(op_codes.nbytes)
    scalars_gpu = loader.gpu_malloc(max(4, scalars.nbytes))  # At least 4 bytes
    vectors_gpu = loader.gpu_malloc(vectors.nbytes)

    loader.memcpy_htod(op_codes_gpu, op_codes.ctypes.data, op_codes.nbytes)
    loader.memcpy_htod(vectors_gpu, vectors.ctypes.data, vectors.nbytes)
    print(f"✓ GPU memory allocated and data copied")
except Exception as e:
    print(f"❌ GPU allocation failed: {e}")
    exit(1)

# Launch kernel
print("\n[4/6] Launching kernel")
try:
    loader.launch(
        kernel,
        grid=(1, 1, 1),
        block=(1, 1, 1),
        params=[
            ctypes.c_uint32(0),  # instance_id
            ctypes.c_uint64(op_codes_gpu.value),
            ctypes.c_uint64(scalars_gpu.value),
            ctypes.c_uint64(vectors_gpu.value),
            ctypes.c_uint64(state_buffer.value),
            ctypes.c_uint32(len(op_codes)),
        ],
    )
    loader.synchronize()
    print(f"✓ Kernel executed")
except Exception as e:
    print(f"❌ Kernel launch failed: {e}")
    exit(1)

# Read error code from state buffer
print("\n[5/6] Reading results from GPU")
try:
    # State layout: head(4) + size(4) + error(4) + reserved(4) + stack[...]
    error_host = ctypes.c_uint32()
    error_ptr = loader.CUdeviceptr(state_buffer.value + 8)  # Offset 8 = error field
    loader.memcpy_dtoh(ctypes.byref(error_host), error_ptr, ctypes.sizeof(error_host))
    error_code = error_host.value

    # Read result from stack
    result_host = ctypes.c_float()
    result_ptr = loader.CUdeviceptr(state_buffer.value + 16)  # Offset 16 = stack[0]
    loader.memcpy_dtoh(ctypes.byref(result_host), result_ptr, ctypes.sizeof(result_host))
    result = result_host.value

    print(f"✓ Results read from GPU")
except Exception as e:
    print(f"❌ Failed to read results: {e}")
    exit(1)

# Cleanup
print("\n[6/6] Cleaning up GPU memory")
try:
    loader.gpu_free(op_codes_gpu)
    loader.gpu_free(scalars_gpu)
    loader.gpu_free(vectors_gpu)
    loader.gpu_free(state_buffer)
    print(f"✓ GPU memory freed")
except Exception as e:
    print(f"⚠️  Cleanup warning: {e}")

# Print results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Error code: {error_code}")
if error_code == 0:
    print(f"  → kErrorNone (success)")
elif error_code == 9001:
    print(f"  → kErrorUnknownOpcode")
elif error_code == 9002:
    print(f"  → kErrorStackUnderflow")
elif error_code == 9003:
    print(f"  → kErrorStackOverflow")
else:
    print(f"  → Unknown error")

print(f"\nResult: {result:.6f}")
print(f"Expected: 32.000000")
print(f"Difference: {abs(result - 32.0):.6f}")

# Final verdict
print("\n" + "=" * 70)
if error_code == 0 and abs(result - 32.0) < 0.01:
    print("✅ PASS: Original kernel works correctly!")
    print("   → Problem is in EXTENDED kernel, not loader")
    exit(0)
else:
    print("❌ FAIL: Original kernel broken")
    print("   → Problem is in loader API or kernel execution")
    print("   → Fix loader before testing extended kernel")
    exit(1)
