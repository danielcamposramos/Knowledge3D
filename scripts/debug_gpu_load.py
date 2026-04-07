#!/usr/bin/env python3
"""Debug GPU module loading issue."""

import os
os.environ['K3D_RPN_DEBUG'] = '1'

print("=== Testing GPU module loading ===\n")

# Test 1: Basic CUDA context
print("Test 1: Creating CUDA context...")
try:
    import cupy as cp
    print(f"  ✓ CuPy available, devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"  ✓ Free memory: {cp.cuda.runtime.memGetInfo()[0] / 1024**2:.1f} MB")
except Exception as e:
    print(f"  ✗ CuPy failed: {e}")

# Test 2: Load procedural_glyph PTX
print("\nTest 2: Loading procedural_glyph_rasterizer.ptx...")
try:
    from pathlib import Path
    from knowledge3d.cranium.sovereign import loader

    ptx_path = Path("knowledge3d/cranium/ptx/procedural_glyph_rasterizer.ptx")
    if ptx_path.exists():
        print(f"  ✓ PTX file exists: {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        print(f"  ✓ Module loaded successfully")
    else:
        print(f"  ✗ PTX file not found: {ptx_path}")
except Exception as e:
    print(f"  ✗ Loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Load pixel_genesis PTX
print("\nTest 3: Loading pixel_genesis_universal_primitive.ptx...")
try:
    ptx_path = Path("knowledge3d/cranium/ptx/pixel_genesis_universal_primitive.ptx")
    if ptx_path.exists():
        print(f"  ✓ PTX file exists: {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        print(f"  ✓ Module loaded successfully")
    else:
        print(f"  ✗ PTX file not found: {ptx_path}")
except Exception as e:
    print(f"  ✗ Loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug complete ===")
