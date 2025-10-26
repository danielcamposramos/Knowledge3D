"""Conv2dBridge: Sovereign 3×3 convolution for DeepSeek OCR.

Phase F.1 - Foundation layer implementing GPU-native convolution without
PyTorch/TensorFlow/CuPy dependencies.

Based on:
- Kimi v1: 16×16 tiling with 2-pixel halo
- Grok: Generalized Cin chunks (64)
- Sovereign loader: Pure ctypes + libcuda.so

Architecture:
    Input: [H, W, Cin]  (height-major, channel-last)
    Weight: [Cout, 3, 3, Cin]
    Bias: [Cout]
    Output: [H, W, Cout]

Performance targets:
    - Latency: <0.5ms (typical OCR feature maps)
    - Accuracy: 99.9% bit-match with NumPy reference
    - Memory: OOM-spill manager integration

Usage:
    >>> bridge = Conv2dBridge()
    >>> output = bridge.forward(input, weight, bias, relu=True)
"""

from pathlib import Path
from typing import Optional, Tuple
import ctypes
import numpy as np
import subprocess
import tempfile

from knowledge3d.cranium.sovereign import loader


class Conv2dBridge:
    """Sovereign 3×3 convolution bridge using PTX kernels."""

    # Kernel configuration (must match conv2d_3x3.cu)
    TILE_SIZE = 16
    HALO_SIZE = 1
    CIN_CHUNK = 32  # Reduced to fit in 64 KB shared memory limit

    def __init__(self, use_oom_spill: bool = False):
        """Initialize Conv2dBridge.

        Args:
            use_oom_spill: Enable OOM spill manager (default: False)
        """
        self.use_oom_spill = use_oom_spill

        # Compile and load PTX
        self.module = self._compile_and_load()

        # Get kernel functions
        self.kernel_fused = loader.get_function(self.module, "conv2d_3x3_fused")
        self.kernel_no_relu = loader.get_function(self.module, "conv2d_3x3_no_relu")

        # Device pointers (cached for repeated calls)
        self._d_input: Optional[loader.CUdeviceptr] = None
        self._d_weight: Optional[loader.CUdeviceptr] = None
        self._d_bias: Optional[loader.CUdeviceptr] = None
        self._d_output: Optional[loader.CUdeviceptr] = None

        # Cached sizes
        self._input_size = 0
        self._weight_size = 0
        self._bias_size = 0
        self._output_size = 0

    def _compile_and_load(self) -> loader.CUmodule:
        """Compile CUDA kernel to PTX and load into module.

        Returns:
            CUmodule handle
        """
        # Find kernel source
        kernel_dir = Path(__file__).parent.parent / "ptx"
        cu_path = kernel_dir / "conv2d_3x3.cu"

        if not cu_path.exists():
            raise FileNotFoundError(f"Kernel not found: {cu_path}")

        # Compile to PTX
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as ptx_file:
            ptx_path = ptx_file.name

        try:
            # Compile with nvcc
            cmd = [
                "nvcc",
                "-ptx",
                str(cu_path),
                "-o", ptx_path,
                "-arch=sm_75",  # Target RTX 3060 (Turing)
                "-O3",  # Optimize for performance
                "--use_fast_math",  # Fast math operations
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Print compilation output if any
            if result.stdout:
                print(f"nvcc stdout: {result.stdout}")
            if result.stderr:
                print(f"nvcc stderr: {result.stderr}")

            # Load PTX module
            module = loader.load_module_from_file(ptx_path)

            return module

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to compile conv2d_3x3.cu:\n"
                f"STDOUT: {e.stdout}\n"
                f"STDERR: {e.stderr}"
            ) from e

    def _allocate_buffers(
        self,
        input_shape: Tuple[int, int, int],
        weight_shape: Tuple[int, int, int, int],
        output_shape: Tuple[int, int, int]
    ):
        """Allocate GPU buffers if needed.

        Args:
            input_shape: (H, W, Cin)
            weight_shape: (Cout, 3, 3, Cin)
            output_shape: (H, W, Cout)
        """
        H, W, Cin = input_shape
        Cout, _, _, _ = weight_shape

        # Calculate sizes
        input_size = H * W * Cin * 4  # 4 bytes per float32
        weight_size = Cout * 3 * 3 * Cin * 4
        bias_size = Cout * 4
        output_size = H * W * Cout * 4

        # Allocate or reuse buffers
        if self._input_size < input_size:
            if self._d_input is not None:
                loader.gpu_free(self._d_input)
            self._d_input = loader.gpu_malloc(input_size)
            self._input_size = input_size

        if self._weight_size < weight_size:
            if self._d_weight is not None:
                loader.gpu_free(self._d_weight)
            self._d_weight = loader.gpu_malloc(weight_size)
            self._weight_size = weight_size

        if self._bias_size < bias_size:
            if self._d_bias is not None:
                loader.gpu_free(self._d_bias)
            self._d_bias = loader.gpu_malloc(bias_size)
            self._bias_size = bias_size

        if self._output_size < output_size:
            if self._d_output is not None:
                loader.gpu_free(self._d_output)
            self._d_output = loader.gpu_malloc(output_size)
            self._output_size = output_size

    def forward(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        relu: bool = True,
        stride: int = 1,
        padding: int = 1
    ) -> np.ndarray:
        """Execute 3×3 convolution on GPU.

        Args:
            input: Input feature map [H, W, Cin] (float32)
            weight: Convolution weights [Cout, 3, 3, Cin] (float32)
            bias: Bias vector [Cout] (float32)
            relu: Apply ReLU activation (default: True)
            stride: Convolution stride (default: 1)
            padding: Padding size (default: 1)

        Returns:
            Output feature map [H, W, Cout] (float32)
        """
        # Validate inputs
        assert input.dtype == np.float32, "Input must be float32"
        assert weight.dtype == np.float32, "Weight must be float32"
        assert bias.dtype == np.float32, "Bias must be float32"
        assert len(input.shape) == 3, "Input must be [H, W, Cin]"
        assert len(weight.shape) == 4, "Weight must be [Cout, 3, 3, Cin]"
        assert len(bias.shape) == 1, "Bias must be [Cout]"

        H, W, Cin = input.shape
        Cout, kh, kw, cin_w = weight.shape

        assert kh == 3 and kw == 3, "Only 3×3 kernels supported"
        assert Cin == cin_w, "Input channels mismatch"
        assert Cout == len(bias), "Bias length mismatch"

        # Output shape (same spatial size with padding=1, stride=1)
        H_out = H
        W_out = W
        output_shape = (H_out, W_out, Cout)

        # Allocate GPU buffers
        self._allocate_buffers(input.shape, weight.shape, output_shape)

        # Copy inputs to GPU
        loader.memcpy_htod(
            self._d_input,
            input.ctypes.data_as(ctypes.c_void_p),
            H * W * Cin * 4
        )
        loader.memcpy_htod(
            self._d_weight,
            weight.ctypes.data_as(ctypes.c_void_p),
            Cout * 3 * 3 * Cin * 4
        )
        loader.memcpy_htod(
            self._d_bias,
            bias.ctypes.data_as(ctypes.c_void_p),
            Cout * 4
        )

        # Calculate grid and block dimensions
        grid_x = (W_out + self.TILE_SIZE - 1) // self.TILE_SIZE
        grid_y = (H_out + self.TILE_SIZE - 1) // self.TILE_SIZE
        grid_z = Cout

        grid = (grid_x, grid_y, grid_z)
        block = (self.TILE_SIZE, self.TILE_SIZE, 1)

        # Select kernel
        kernel = self.kernel_fused if relu else self.kernel_no_relu

        # Launch kernel
        params = [
            ctypes.c_uint64(self._d_input.value),
            ctypes.c_uint64(self._d_weight.value),
            ctypes.c_uint64(self._d_bias.value),
            ctypes.c_uint64(self._d_output.value),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(Cin),
            ctypes.c_int(Cout),
            ctypes.c_int(stride),
            ctypes.c_int(padding),
        ]

        loader.launch(kernel, grid, block, params, shared_mem=0)

        # Synchronize
        loader.synchronize()

        # Copy output back to host
        output = np.empty(output_shape, dtype=np.float32)
        loader.memcpy_dtoh(
            output.ctypes.data_as(ctypes.c_void_p),
            self._d_output,
            H_out * W_out * Cout * 4
        )

        return output

    def __del__(self):
        """Clean up GPU resources."""
        if self._d_input is not None:
            loader.gpu_free(self._d_input)
        if self._d_weight is not None:
            loader.gpu_free(self._d_weight)
        if self._d_bias is not None:
            loader.gpu_free(self._d_bias)
        if self._d_output is not None:
            loader.gpu_free(self._d_output)


def conv2d_3x3_numpy(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    relu: bool = True,
    stride: int = 1,
    padding: int = 1
) -> np.ndarray:
    """NumPy reference implementation of 3×3 convolution.

    Used for validation and testing.

    Args:
        input: Input feature map [H, W, Cin]
        weight: Convolution weights [Cout, 3, 3, Cin]
        bias: Bias vector [Cout]
        relu: Apply ReLU activation
        stride: Convolution stride
        padding: Padding size

    Returns:
        Output feature map [H, W, Cout]
    """
    H, W, Cin = input.shape
    Cout, _, _, _ = weight.shape

    # Apply padding
    if padding > 0:
        input_padded = np.pad(
            input,
            ((padding, padding), (padding, padding), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input

    H_padded, W_padded, _ = input_padded.shape

    # Output dimensions
    H_out = (H_padded - 3) // stride + 1
    W_out = (W_padded - 3) // stride + 1

    output = np.zeros((H_out, W_out, Cout), dtype=np.float32)

    # Convolve
    for out_c in range(Cout):
        for h in range(H_out):
            for w in range(W_out):
                # Extract patch
                h_start = h * stride
                w_start = w * stride
                patch = input_padded[h_start:h_start+3, w_start:w_start+3, :]

                # Convolve with kernel
                value = np.sum(patch * weight[out_c, :, :, :])

                # Add bias
                value += bias[out_c]

                # ReLU
                if relu:
                    value = max(0.0, value)

                output[h, w, out_c] = value

    return output


__all__ = ["Conv2dBridge", "conv2d_3x3_numpy"]
