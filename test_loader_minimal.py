#!/usr/bin/env python3
"""
Minimal loader test - just test basic memory operations.
"""
import numpy as np
from knowledge3d.cranium.sovereign import loader
import ctypes

print("=" * 70)
print("Minimal Loader Test")
print("=" * 70)

print("\n[1/5] Allocating GPU memory (1KB)")
try:
    ptr = loader.gpu_malloc(1024)
    print(f"✓ GPU memory allocated: {ptr}")
    print(f"  Type: {type(ptr)}")
    print(f"  Value: {ptr.value if hasattr(ptr, 'value') else ptr}")
except Exception as e:
    print(f"❌ GPU allocation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[2/5] Preparing host data")
data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
print(f"  Data: {data}")
print(f"  Size: {data.nbytes} bytes")

print("\n[3/5] Copying host → device")
try:
    loader.memcpy_htod(ptr, data.ctypes.data, data.nbytes)
    print(f"✓ Data copied to GPU")
except Exception as e:
    print(f"❌ memcpy_htod failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[4/5] Copying device → host")
try:
    result = np.zeros_like(data)
    # Use ctypes.c_void_p for host pointer
    dst_ptr = ctypes.c_void_p(result.ctypes.data)
    loader.memcpy_dtoh(dst_ptr, ptr, data.nbytes)
    print(f"✓ Data copied from GPU")
    print(f"  Result: {result}")
except Exception as e:
    print(f"❌ memcpy_dtoh failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[5/5] Verifying data")
if np.allclose(result, data):
    print(f"✓ Data matches!")
else:
    print(f"❌ Data mismatch!")
    print(f"  Expected: {data}")
    print(f"  Got: {result}")
    exit(1)

print("\n[6/5] Freeing GPU memory")
try:
    loader.gpu_free(ptr)
    print(f"✓ GPU memory freed")
except Exception as e:
    print(f"⚠️  gpu_free warning: {e}")

print("\n" + "=" * 70)
print("✅ PASS: Loader basic operations work!")
print("=" * 70)
