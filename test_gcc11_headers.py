#!/usr/bin/env python3
"""Test if CuPy can use GCC 11 headers instead of GCC 15."""

import os
from pathlib import Path

# Point CuPy's NVRTC to GCC 11 headers
gcc11_include = "/usr/lib/gcc/x86_64-linux-gnu/11/include"
if Path(gcc11_include).exists():
    os.environ['CUDA_INCLUDE_DIRS'] = gcc11_include
    print(f"Set CUDA_INCLUDE_DIRS to GCC 11: {gcc11_include}")
else:
    print(f"GCC 11 include dir not found: {gcc11_include}")

# Try to force CuPy to use system GCC 11
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'

import cupy as cp

print(f"CuPy version: {cp.__version__}")

# Test the failing operation
print("\nTesting cp.linalg.norm...")
try:
    vec = cp.random.rand(10, 3).astype(cp.float32)
    norms = cp.linalg.norm(vec, axis=1)
    print(f"✓ SUCCESS with GCC 11 headers!")
except Exception as e:
    print(f"✗ FAILED: {str(e)[:100]}...")
