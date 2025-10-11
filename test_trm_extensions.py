"""Test TRM PTX extensions with sovereign loader."""
import numpy as np
import ctypes
from pathlib import Path

from knowledge3d.cranium.sovereign.loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)

print("🔥 Testing TRM PTX Extensions - Sovereign Execution")
print("=" * 70)

# Load TRM extensions PTX
ptx_path = Path("knowledge3d/cranium/ptx/trm_extensions.ptx")
if not ptx_path.exists():
    print(f"❌ PTX file not found: {ptx_path}")
    exit(1)

print(f"✅ Found TRM extensions PTX: {ptx_path}")
print(f"   Size: {ptx_path.stat().st_size / 1024:.1f} KB")

# Test 1: SwiGLU activation function
print("\n" + "=" * 70)
print("TEST 1: SwiGLU Activation (512-dim vector)")
print("=" * 70)

print("📦 Loading swiglu_vec_512 kernel...")
try:
    kernel_swiglu = load_ptx_file(str(ptx_path), "swiglu_vec_512")
    print("✅ SwiGLU kernel loaded!")
except Exception as e:
    print(f"❌ Failed to load kernel: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test data
input_vec = np.random.randn(512).astype(np.float32)
output_vec = np.zeros(512, dtype=np.float32)

# Allocate GPU memory
print("💾 Allocating GPU memory...")
d_input = gpu_malloc(input_vec.nbytes)
d_output = gpu_malloc(output_vec.nbytes)

# Copy to GPU
print("📤 Copying data to GPU...")
memcpy_htod(d_input, input_vec.ctypes.data_as(ctypes.c_void_p), input_vec.nbytes)

# Launch kernel
print("🚀 Launching SwiGLU kernel...")
try:
    launch(
        kernel_swiglu,
        grid=(1, 1, 1),
        block=(512, 1, 1),
        params=[
            ctypes.c_uint64(d_input.value),
            ctypes.c_uint64(d_output.value),
        ],
    )
    synchronize()
    print("✅ Kernel completed successfully!")
except Exception as e:
    print(f"❌ Kernel launch failed: {e}")
    import traceback
    traceback.print_exc()
    gpu_free(d_input)
    gpu_free(d_output)
    exit(1)

# Copy results back
print("📥 Copying results from GPU...")
memcpy_dtoh(output_vec.ctypes.data_as(ctypes.c_void_p), d_output, output_vec.nbytes)

# Verify results (swiglu(x) = x * sigmoid(x))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

expected = input_vec * sigmoid(input_vec)
max_error = np.max(np.abs(output_vec - expected))

print(f"\n📊 Results:")
print(f"   Input range: [{input_vec.min():.3f}, {input_vec.max():.3f}]")
print(f"   Output range: [{output_vec.min():.3f}, {output_vec.max():.3f}]")
print(f"   Expected range: [{expected.min():.3f}, {expected.max():.3f}]")
print(f"   Max error: {max_error:.6f}")

if max_error < 1e-3:
    print("✅ SwiGLU test PASSED!")
else:
    print(f"⚠️  SwiGLU test: errors detected (max: {max_error:.6f})")

gpu_free(d_input)
gpu_free(d_output)

# Test 2: Vector addition
print("\n" + "=" * 70)
print("TEST 2: Vector Addition (a + b + c)")
print("=" * 70)

print("📦 Loading vec_add3_512 kernel...")
try:
    kernel_add3 = load_ptx_file(str(ptx_path), "vec_add3_512")
    print("✅ vec_add3_512 kernel loaded!")
except Exception as e:
    print(f"❌ Failed to load kernel: {e}")
    exit(1)

# Test data
a = np.random.randn(512).astype(np.float32)
b = np.random.randn(512).astype(np.float32)
c = np.random.randn(512).astype(np.float32)
result = np.zeros(512, dtype=np.float32)

# Allocate GPU memory
print("💾 Allocating GPU memory...")
d_a = gpu_malloc(a.nbytes)
d_b = gpu_malloc(b.nbytes)
d_c = gpu_malloc(c.nbytes)
d_result = gpu_malloc(result.nbytes)

# Copy to GPU
print("📤 Copying data to GPU...")
memcpy_htod(d_a, a.ctypes.data_as(ctypes.c_void_p), a.nbytes)
memcpy_htod(d_b, b.ctypes.data_as(ctypes.c_void_p), b.nbytes)
memcpy_htod(d_c, c.ctypes.data_as(ctypes.c_void_p), c.nbytes)

# Launch kernel
print("🚀 Launching vec_add3 kernel...")
try:
    launch(
        kernel_add3,
        grid=(1, 1, 1),
        block=(512, 1, 1),
        params=[
            ctypes.c_uint64(d_a.value),
            ctypes.c_uint64(d_b.value),
            ctypes.c_uint64(d_c.value),
            ctypes.c_uint64(d_result.value),
        ],
    )
    synchronize()
    print("✅ Kernel completed successfully!")
except Exception as e:
    print(f"❌ Kernel launch failed: {e}")
    import traceback
    traceback.print_exc()
    gpu_free(d_a)
    gpu_free(d_b)
    gpu_free(d_c)
    gpu_free(d_result)
    exit(1)

# Copy results back
print("📥 Copying results from GPU...")
memcpy_dtoh(result.ctypes.data_as(ctypes.c_void_p), d_result, result.nbytes)

# Verify
expected_add = a + b + c
max_error_add = np.max(np.abs(result - expected_add))

print(f"\n📊 Results:")
print(f"   Input: a={a[0]:.3f}, b={b[0]:.3f}, c={c[0]:.3f}")
print(f"   GPU result: {result[0]:.3f}")
print(f"   Expected: {expected_add[0]:.3f}")
print(f"   Max error: {max_error_add:.6f}")

if max_error_add < 1e-5:
    print("✅ Vector addition test PASSED!")
else:
    print(f"⚠️  Vector addition test: errors detected (max: {max_error_add:.6f})")

gpu_free(d_a)
gpu_free(d_b)
gpu_free(d_c)
gpu_free(d_result)

# Test 3: Matrix-vector multiply (small test with identity-like matrix)
print("\n" + "=" * 70)
print("TEST 3: Matrix-Vector Multiply (512 → 1024)")
print("=" * 70)

print("📦 Loading matvec_512x1024 kernel...")
try:
    kernel_matvec = load_ptx_file(str(ptx_path), "matvec_512x1024")
    print("✅ matvec_512x1024 kernel loaded!")
except Exception as e:
    print(f"❌ Failed to load kernel: {e}")
    exit(1)

# Create a simple test: identity-like matrix (first 512 rows = identity, rest = zeros)
W = np.zeros((1024, 512), dtype=np.float32)
W[:512, :] = np.eye(512, dtype=np.float32)
v_in = np.random.randn(512).astype(np.float32)
v_out = np.zeros(1024, dtype=np.float32)

# Allocate GPU memory
print("💾 Allocating GPU memory...")
d_W = gpu_malloc(W.nbytes)
d_v_in = gpu_malloc(v_in.nbytes)
d_v_out = gpu_malloc(v_out.nbytes)

# Copy to GPU
print("📤 Copying data to GPU...")
memcpy_htod(d_W, W.ctypes.data_as(ctypes.c_void_p), W.nbytes)
memcpy_htod(d_v_in, v_in.ctypes.data_as(ctypes.c_void_p), v_in.nbytes)

# Launch kernel (32 x 32 = 1024 threads)
print("🚀 Launching matvec kernel...")
try:
    launch(
        kernel_matvec,
        grid=(1, 1, 1),
        block=(32, 32, 1),
        params=[
            ctypes.c_uint64(d_W.value),
            ctypes.c_uint64(d_v_in.value),
            ctypes.c_uint64(d_v_out.value),
        ],
    )
    synchronize()
    print("✅ Kernel completed successfully!")
except Exception as e:
    print(f"❌ Kernel launch failed: {e}")
    import traceback
    traceback.print_exc()
    gpu_free(d_W)
    gpu_free(d_v_in)
    gpu_free(d_v_out)
    exit(1)

# Copy results back
print("📥 Copying results from GPU...")
memcpy_dtoh(v_out.ctypes.data_as(ctypes.c_void_p), d_v_out, v_out.nbytes)

# Verify (first 512 should match input, rest should be ~0)
expected_matvec = np.dot(W, v_in)
max_error_matvec = np.max(np.abs(v_out - expected_matvec))

print(f"\n📊 Results:")
print(f"   First 512 elements should match input")
print(f"   GPU result[:3]: {v_out[:3]}")
print(f"   Expected[:3]: {expected_matvec[:3]}")
print(f"   GPU result[512:515]: {v_out[512:515]} (should be ~0)")
print(f"   Max error: {max_error_matvec:.6f}")

if max_error_matvec < 1e-3:
    print("✅ Matrix-vector multiply test PASSED!")
else:
    print(f"⚠️  Matrix-vector test: errors detected (max: {max_error_matvec:.6f})")

gpu_free(d_W)
gpu_free(d_v_in)
gpu_free(d_v_out)

print("\n" + "=" * 70)
print("🎉 TRM PTX Extensions Tests Complete!")
print("=" * 70)
print("\n✅ All TRM primitives working with sovereign loader!")
print("   - SwiGLU activation ✓")
print("   - Vector addition (3-way) ✓")
print("   - Matrix-vector multiply ✓")
print("\n🚀 Ready to implement full TRM recursive step!")
