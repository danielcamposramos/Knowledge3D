"""
GPU Backward Pass - PTX Kernel Interface

Loads compiled backward pass kernels and provides Python interface for:
- Conv2D backward (weights, biases, input gradients)
- BatchNorm backward (gamma, beta, input gradients)
- MaxPool backward (input gradients)
- Loss backward (classification loss)
- Weight updates (SGD with momentum)
"""

from __future__ import annotations

import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from knowledge3d.cranium.sovereign import loader


class GPUBackward:
    """GPU-accelerated backward pass for CNN training."""

    def __init__(self):
        """Initialize GPU backward pass by loading PTX kernels."""
        self.ptx_dir = Path(__file__).parent.parent / "ptx"

        # Load backward pass kernels
        self._load_kernels()

        # GPU memory cache for gradients
        self.grad_cache: Dict[str, int] = {}  # name -> device pointer

    def _load_kernels(self):
        """Load all backward pass PTX kernels."""
        # Define kernels to load: (ptx_file, [kernel_names])
        kernels_to_load = [
            ("maxpool_2x2_backward.ptx", ["maxpool_2x2_backward"]),
            ("batchnorm_backward.ptx", ["batchnorm_backward"]),
            ("conv2d_3x3_backward.ptx", ["conv2d_backward_weight", "conv2d_backward_input", "relu_backward"]),
            ("classification_loss.ptx", [
                "softmax_forward", "cross_entropy_forward", "cross_entropy_softmax_backward",
                "global_avgpool_forward", "global_avgpool_backward", "fc_forward", "fc_backward"
            ]),
            ("sgd_optimizer.ptx", ["sgd_momentum_update", "zero_grad"]),
        ]

        self.modules = {}
        self.kernels = {}

        for ptx_file, kernel_names in kernels_to_load:
            ptx_path = self.ptx_dir / ptx_file

            # Load PTX module
            module = loader.load_module_from_file(str(ptx_path))
            self.modules[ptx_file] = module

            # Get kernel functions
            for kernel_name in kernel_names:
                kernel_func = loader.get_function(module, kernel_name)
                self.kernels[kernel_name] = kernel_func

                # Also set as attributes for backward compatibility
                setattr(self, f"{kernel_name}_kernel", kernel_func)

        # Set specific kernel attributes
        self.maxpool_backward_kernel = self.kernels["maxpool_2x2_backward"]
        self.batchnorm_backward_kernel = self.kernels["batchnorm_backward"]
        self.conv_backward_weight_kernel = self.kernels["conv2d_backward_weight"]
        self.conv_backward_input_kernel = self.kernels["conv2d_backward_input"]
        self.relu_backward_kernel = self.kernels["relu_backward"]
        self.softmax_forward_kernel = self.kernels["softmax_forward"]
        self.cross_entropy_forward_kernel = self.kernels["cross_entropy_forward"]
        self.cross_entropy_softmax_backward_kernel = self.kernels["cross_entropy_softmax_backward"]
        self.global_avgpool_forward_kernel = self.kernels["global_avgpool_forward"]
        self.global_avgpool_backward_kernel = self.kernels["global_avgpool_backward"]
        self.fc_forward_kernel = self.kernels["fc_forward"]
        self.fc_backward_kernel = self.kernels["fc_backward"]
        self.sgd_momentum_update_kernel = self.kernels["sgd_momentum_update"]
        self.zero_grad_kernel = self.kernels["zero_grad"]

    def allocate_grad_buffer(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> int:
        """Allocate GPU memory for gradient buffer."""
        size_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        d_ptr = loader.gpu_malloc(size_bytes)
        self.grad_cache[name] = d_ptr

        # Zero initialize
        self.zero_gradients(d_ptr, int(np.prod(shape)))

        return d_ptr

    def zero_gradients(self, d_grad: int, n_elements: int):
        """Zero out gradient buffer on GPU."""
        block_size = 256
        grid_size = (n_elements + block_size - 1) // block_size

        loader.launch(
            self.zero_grad_kernel,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
            params=[d_grad, ctypes.c_int(n_elements)]
        )

    def maxpool_backward(
        self,
        d_out: np.ndarray,  # [H_out, W_out, C]
        x_in: np.ndarray,   # [H_in, W_in, C]
        d_input: int        # Device pointer for output
    ):
        """Route gradients through max pooling."""
        H_out, W_out, C = d_out.shape
        H_in, W_in, _ = x_in.shape

        # Upload to GPU
        d_d_out = loader.gpu_malloc(d_out.nbytes)
        d_x_in = loader.gpu_malloc(x_in.nbytes)
        loader.memcpy_htod(d_d_out, d_out.ctypes.data_as(ctypes.c_void_p), d_out.nbytes)
        loader.memcpy_htod(d_x_in, x_in.ctypes.data_as(ctypes.c_void_p), x_in.nbytes)

        # Launch kernel
        block = (16, 16, 1)
        grid = ((W_out + 15) // 16, (H_out + 15) // 16, C)

        loader.launch(
            self.maxpool_backward_kernel,
            grid=grid,
            block=block,
            params=[d_d_out, d_x_in, d_input,
                    ctypes.c_int(H_in), ctypes.c_int(W_in),
                    ctypes.c_int(H_out), ctypes.c_int(W_out), ctypes.c_int(C)]
        )

        # Cleanup
        loader.gpu_free(d_d_out)
        loader.gpu_free(d_x_in)

    def batchnorm_backward(
        self,
        d_out: np.ndarray,
        x_in: np.ndarray,
        gamma: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        d_input: int,   # Device pointer
        d_gamma: int,   # Device pointer
        d_beta: int,    # Device pointer
        eps: float = 1e-5
    ):
        """Compute BatchNorm backward pass."""
        H, W, C = d_out.shape

        # Upload to GPU
        d_d_out = loader.gpu_malloc(d_out.nbytes)
        d_x_in = loader.gpu_malloc(x_in.nbytes)
        d_gamma_val = loader.gpu_malloc(gamma.nbytes)
        d_mean = loader.gpu_malloc(running_mean.nbytes)
        d_var = loader.gpu_malloc(running_var.nbytes)

        loader.memcpy_htod(d_d_out, d_out.ctypes.data_as(ctypes.c_void_p), d_out.nbytes)
        loader.memcpy_htod(d_x_in, x_in.ctypes.data_as(ctypes.c_void_p), x_in.nbytes)
        loader.memcpy_htod(d_gamma_val, gamma.ctypes.data_as(ctypes.c_void_p), gamma.nbytes)
        loader.memcpy_htod(d_mean, running_mean.ctypes.data_as(ctypes.c_void_p), running_mean.nbytes)
        loader.memcpy_htod(d_var, running_var.ctypes.data_as(ctypes.c_void_p), running_var.nbytes)

        # Launch kernel (one block per channel)
        block = (256, 1, 1)
        grid = (C, 1, 1)

        loader.launch(
            self.batchnorm_backward_kernel,
            grid=grid,
            block=block,
            params=[d_d_out, d_x_in, d_gamma_val, d_mean, d_var,
                    d_input, d_gamma, d_beta,
                    ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(C), ctypes.c_float(eps)]
        )

        # Cleanup
        loader.gpu_free(d_d_out)
        loader.gpu_free(d_x_in)
        loader.gpu_free(d_gamma_val)
        loader.gpu_free(d_mean)
        loader.gpu_free(d_var)

    def conv_backward_weight(
        self,
        d_out: np.ndarray,  # [H_out, W_out, Cout]
        x_in_padded: np.ndarray,  # [H_in+2, W_in+2, Cin]
        d_weight: int,  # Device pointer
        d_bias: int,    # Device pointer
        Cin: int,
        Cout: int
    ):
        """Compute Conv2D weight and bias gradients."""
        H_out, W_out, _ = d_out.shape
        H_in_pad, W_in_pad, _ = x_in_padded.shape
        H_in = H_in_pad - 2
        W_in = W_in_pad - 2

        # Upload to GPU
        d_d_out = loader.gpu_malloc(d_out.nbytes)
        d_x_in = loader.gpu_malloc(x_in_padded.nbytes)
        loader.memcpy_htod(d_d_out, d_out.ctypes.data_as(ctypes.c_void_p), d_out.nbytes)
        loader.memcpy_htod(d_x_in, x_in_padded.ctypes.data_as(ctypes.c_void_p), x_in_padded.nbytes)

        # Launch kernel (one block per output channel)
        block = (256, 1, 1)
        grid = (Cout, 1, 1)

        loader.launch(
            self.conv_backward_weight_kernel,
            grid=grid,
            block=block,
            params=[d_d_out, d_x_in, d_weight, d_bias,
                    ctypes.c_int(H_out), ctypes.c_int(W_out),
                    ctypes.c_int(H_in), ctypes.c_int(W_in),
                    ctypes.c_int(Cin), ctypes.c_int(Cout)]
        )

        # Cleanup
        loader.gpu_free(d_d_out)
        loader.gpu_free(d_x_in)

    def conv_backward_input(
        self,
        d_out: np.ndarray,  # [H_out, W_out, Cout]
        weight: np.ndarray,  # [Cout, 3, 3, Cin]
        d_input_padded: int,  # Device pointer [H_in+2, W_in+2, Cin]
        H_in: int,
        W_in: int,
        Cin: int,
        Cout: int
    ):
        """Compute Conv2D input gradients."""
        H_out, W_out, _ = d_out.shape

        # Upload to GPU
        d_d_out = loader.gpu_malloc(d_out.nbytes)
        d_weight = loader.gpu_malloc(weight.nbytes)
        loader.memcpy_htod(d_d_out, d_out.ctypes.data_as(ctypes.c_void_p), d_out.nbytes)
        loader.memcpy_htod(d_weight, weight.ctypes.data_as(ctypes.c_void_p), weight.nbytes)

        # Launch kernel
        block = (16, 16, 1)
        grid = ((W_in + 2 + 15) // 16, (H_in + 2 + 15) // 16, Cin)

        loader.launch(
            self.conv_backward_input_kernel,
            grid=grid,
            block=block,
            params=[d_d_out, d_weight, d_input_padded,
                    ctypes.c_int(H_out), ctypes.c_int(W_out),
                    ctypes.c_int(H_in), ctypes.c_int(W_in),
                    ctypes.c_int(Cin), ctypes.c_int(Cout)]
        )

        # Cleanup
        loader.gpu_free(d_d_out)
        loader.gpu_free(d_weight)

    def sgd_momentum_update(
        self,
        d_param: int,  # Device pointer to parameters
        d_grad: int,   # Device pointer to gradients
        d_velocity: int,  # Device pointer to velocity
        learning_rate: float,
        momentum: float,
        n_elements: int
    ):
        """Update parameters using SGD with momentum."""
        block_size = 256
        grid_size = (n_elements + block_size - 1) // block_size

        loader.launch(
            self.sgd_momentum_update_kernel,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
            params=[d_param, d_grad, d_velocity,
                    ctypes.c_float(learning_rate), ctypes.c_float(momentum),
                    ctypes.c_int(n_elements)]
        )

    def __del__(self):
        """Cleanup GPU memory."""
        for d_ptr in self.grad_cache.values():
            try:
                loader.gpu_free(d_ptr)
            except:
                pass
