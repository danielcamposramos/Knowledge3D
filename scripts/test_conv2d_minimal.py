#!/usr/bin/env python3
"""Minimal test for conv2d kernel infrastructure."""

import numpy as np
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.sovereign import loader

# Create a minimal kernel that just fills output with zeros
minimal_kernel_source = """
extern "C" __global__ void test_minimal(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}
"""

def test_minimal():
    """Test minimal kernel compilation and execution."""
    print("Testing minimal kernel...")

    # Write kernel to file
    cu_path = "/tmp/test_minimal.cu"
    with open(cu_path, "w") as f:
        f.write(minimal_kernel_source)

    # Compile
    import subprocess
    ptx_path = "/tmp/test_minimal.ptx"
    cmd = ["nvcc", "-ptx", cu_path, "-o", ptx_path, "-arch=sm_75"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False

    print("✓ Compilation successful")

    # Load module
    module = loader.load_module_from_file(ptx_path)
    kernel = loader.get_function(module, "test_minimal")
    print("✓ Kernel loaded")

    # Test data
    size = 1024
    input_data = np.arange(size, dtype=np.float32)
    output_data = np.zeros(size, dtype=np.float32)

    # Allocate GPU memory
    d_input = loader.gpu_malloc(input_data.nbytes)
    d_output = loader.gpu_malloc(output_data.nbytes)

    # Copy to GPU
    loader.memcpy_htod(d_input, input_data.ctypes.data_as(ctypes.c_void_p), input_data.nbytes)

    # Launch
    grid = ((size + 255) // 256, 1, 1)
    block = (256, 1, 1)
    params = [
        ctypes.c_uint64(d_input.value),
        ctypes.c_uint64(d_output.value),
        ctypes.c_int(size),
    ]

    print(f"Launching with grid={grid}, block={block}")

    try:
        loader.launch(kernel, grid, block, params)
        loader.synchronize()
        print("✓ Kernel launched successfully")
    except Exception as e:
        print(f"❌ Kernel launch failed: {e}")
        loader.gpu_free(d_input)
        loader.gpu_free(d_output)
        return False

    # Copy back
    loader.memcpy_dtoh(output_data.ctypes.data_as(ctypes.c_void_p), d_output, output_data.nbytes)

    # Verify
    expected = input_data * 2.0
    if np.allclose(output_data, expected):
        print("✓ Results correct!")
        loader.gpu_free(d_input)
        loader.gpu_free(d_output)
        return True
    else:
        print(f"❌ Results incorrect:")
        print(f"  Expected: {expected[:10]}")
        print(f"  Got:      {output_data[:10]}")
        loader.gpu_free(d_input)
        loader.gpu_free(d_output)
        return False

if __name__ == "__main__":
    success = test_minimal()
    sys.exit(0 if success else 1)
