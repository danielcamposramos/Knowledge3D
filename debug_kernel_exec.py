#!/usr/bin/env python3
"""Debug kernel execution with minimal test."""

import os
os.environ['K3D_RPN_DEBUG'] = '1'

import ctypes
import numpy as np
from knowledge3d.cranium.sovereign import loader
from pathlib import Path

print("=== Testing pixel_genesis kernel execution ===\n")

# Load module
ptx_path = Path("knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx")
module = loader.load_module_from_file(str(ptx_path))
kernel = loader.get_function(module, "execute_drawing_rpn")
print("✓ Kernel loaded\n")

# Create simple bytecode: MOVE 0.5 0.5
import struct
bytecode = bytearray()
bytecode.append(0x64)  # MOVE opcode
bytecode.extend(struct.pack('<f', 0.5))  # x = 0.5
bytecode.extend(struct.pack('<f', 0.5))  # y = 0.5

bytecode_np = np.ascontiguousarray(np.frombuffer(bytes(bytecode), dtype=np.uint8))
print(f"Bytecode length: {len(bytecode_np)} bytes")
print(f"Bytecode hex: {bytecode_np.tobytes().hex()}")
print()

# Allocate GPU memory
d_bytecode = loader.gpu_malloc(bytecode_np.nbytes)
d_segments = loader.gpu_malloc(4096 * 16)  # 4096 segments max
d_count = loader.gpu_malloc(4)

print(f"GPU allocations:")
print(f"  d_bytecode: 0x{d_bytecode.value:x} ({bytecode_np.nbytes} bytes)")
print(f"  d_segments: 0x{d_segments.value:x} ({4096*16} bytes)")
print(f"  d_count:    0x{d_count.value:x} (4 bytes)")
print()

# Copy bytecode to GPU
loader.memcpy_htod(d_bytecode, bytecode_np.ctypes.data_as(ctypes.c_void_p), bytecode_np.nbytes)
print("✓ Bytecode copied to GPU\n")

# Launch kernel
try:
    print("Launching kernel...")
    loader.launch(
        kernel,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        params=[
            ctypes.c_uint64(d_bytecode.value),
            ctypes.c_uint32(bytecode_np.nbytes),
            ctypes.c_uint64(d_segments.value),
            ctypes.c_uint64(d_count.value),
            ctypes.c_uint32(32),  # segments_per_curve
            ctypes.c_float(0.0),  # ternary_hint
        ],
    )
    print("✓ Kernel launched\n")

    print("Synchronizing...")
    loader.synchronize()
    print("✓ Kernel completed\n")

    # Read results
    count_host = np.zeros(1, dtype=np.uint32)
    loader.memcpy_dtoh(count_host.ctypes.data_as(ctypes.c_void_p), d_count, 4)
    print(f"Segment count: {count_host[0]}")

    if count_host[0] > 0:
        segments_host = np.zeros((min(int(count_host[0]), 10), 4), dtype=np.float32)
        loader.memcpy_dtoh(segments_host.ctypes.data_as(ctypes.c_void_p), d_segments, segments_host.nbytes)
        print(f"First segments:\n{segments_host}")

except Exception as e:
    print(f"✗ Kernel execution failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    loader.gpu_free(d_bytecode)
    loader.gpu_free(d_segments)
    loader.gpu_free(d_count)
    print("\n✓ GPU memory freed")

print("\n=== Debug complete ===")
