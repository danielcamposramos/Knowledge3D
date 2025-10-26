from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from . import loader


@dataclass
class LoRADeviceBuffers:
    """Holds persistent GPU buffers for specialist training."""

    # Core weights
    base_matrix: loader.CUdeviceptr
    A: loader.CUdeviceptr
    B: loader.CUdeviceptr

    # Dataset
    inputs: loader.CUdeviceptr
    targets: loader.CUdeviceptr

    # Working vectors (dimension = dims)
    base_out: loader.CUdeviceptr
    delta: loader.CUdeviceptr
    output: loader.CUdeviceptr
    error: loader.CUdeviceptr

    # Low-rank helpers (rank dimension)
    Bx: loader.CUdeviceptr
    AtE: loader.CUdeviceptr

    # Gradient buffers
    grad_A: loader.CUdeviceptr
    grad_B: loader.CUdeviceptr

    # Scalar loss
    loss: loader.CUdeviceptr

    # Dataset counts
    n_samples: int


class LoRAGPUEngine:
    """Sovereign GPU trainer for low-rank adapters."""

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "lora_gpu.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"LoRA PTX kernel not found: {ptx_path}")

        self.module = loader.load_module_from_file(str(ptx_path))
        self.k_matvec = loader.get_function(self.module, "matvec_general")
        self.k_matvec_trans = loader.get_function(self.module, "matvec_transpose")
        self.k_vec_add_scaled = loader.get_function(self.module, "vec_add_scaled")
        self.k_vec_sub_square = loader.get_function(self.module, "vec_sub_square")
        self.k_outer_scaled = loader.get_function(self.module, "outer_product_scale")
        self.k_matrix_axpy = loader.get_function(self.module, "matrix_axpy")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _offset(ptr: loader.CUdeviceptr, offset_bytes: int) -> loader.CUdeviceptr:
        return loader.CUdeviceptr(ptr.value + offset_bytes)

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32, order="C")

    # ------------------------------------------------------------------ #
    # Buffer management
    # ------------------------------------------------------------------ #
    def allocate_buffers(
        self,
        base_matrix: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> LoRADeviceBuffers:
        dims = base_matrix.shape[0]
        rank = B.shape[0]
        n_samples = inputs.shape[0]

        base_gpu = loader.gpu_malloc(base_matrix.nbytes)
        loader.memcpy_htod(base_gpu, base_matrix.ctypes.data_as(ctypes.c_void_p), base_matrix.nbytes)

        A_gpu = loader.gpu_malloc(A.nbytes)
        loader.memcpy_htod(A_gpu, A.ctypes.data_as(ctypes.c_void_p), A.nbytes)

        B_gpu = loader.gpu_malloc(B.nbytes)
        loader.memcpy_htod(B_gpu, B.ctypes.data_as(ctypes.c_void_p), B.nbytes)

        inputs_gpu = loader.gpu_malloc(inputs.nbytes)
        loader.memcpy_htod(inputs_gpu, inputs.ctypes.data_as(ctypes.c_void_p), inputs.nbytes)

        targets_gpu = loader.gpu_malloc(targets.nbytes)
        loader.memcpy_htod(targets_gpu, targets.ctypes.data_as(ctypes.c_void_p), targets.nbytes)

        def alloc(size):
            return loader.gpu_malloc(size)

        vector_bytes = dims * 4
        rank_bytes = rank * 4
        grad_A_bytes = dims * rank * 4
        grad_B_bytes = rank * dims * 4

        buffers = LoRADeviceBuffers(
            base_matrix=base_gpu,
            A=A_gpu,
            B=B_gpu,
            inputs=inputs_gpu,
            targets=targets_gpu,
            base_out=alloc(vector_bytes),
            delta=alloc(vector_bytes),
            output=alloc(vector_bytes),
            error=alloc(vector_bytes),
            Bx=alloc(rank_bytes),
            AtE=alloc(rank_bytes),
            grad_A=alloc(grad_A_bytes),
            grad_B=alloc(grad_B_bytes),
            loss=alloc(4),
            n_samples=n_samples,
        )
        return buffers

    def free_buffers(self, buffers: LoRADeviceBuffers) -> None:
        for attr in vars(buffers):
            value = getattr(buffers, attr)
            if isinstance(value, loader.CUdeviceptr):
                loader.gpu_free(value)

    # ------------------------------------------------------------------ #
    # Kernel helpers
    # ------------------------------------------------------------------ #
    def _launch_matvec(self, func, W_ptr, x_ptr, y_ptr, rows, cols):
        block = 256
        grid = (rows + block - 1) // block
        loader.launch(
            func,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(W_ptr.value),
                ctypes.c_uint64(x_ptr.value),
                ctypes.c_uint64(y_ptr.value),
                ctypes.c_int(rows),
                ctypes.c_int(cols),
            ],
        )

    def _launch_vec_add_scaled(self, a_ptr, b_ptr, scale, out_ptr, n):
        block = 256
        grid = (n + block - 1) // block
        loader.launch(
            self.k_vec_add_scaled,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(a_ptr.value),
                ctypes.c_uint64(b_ptr.value),
                ctypes.c_float(scale),
                ctypes.c_uint64(out_ptr.value),
                ctypes.c_int(n),
            ],
        )

    def _launch_vec_sub_square(self, pred_ptr, target_ptr, error_ptr, loss_ptr, n):
        block = 256
        grid = (n + block - 1) // block
        shared = block * 4
        loader.launch(
            self.k_vec_sub_square,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(pred_ptr.value),
                ctypes.c_uint64(target_ptr.value),
                ctypes.c_uint64(error_ptr.value),
                ctypes.c_uint64(loss_ptr.value),
                ctypes.c_int(n),
            ],
            shared_mem=shared,
        )

    def _launch_outer(self, a_ptr, b_ptr, out_ptr, rows, cols, scale):
        block_x = 16
        block_y = 16
        grid_x = (cols + block_x - 1) // block_x
        grid_y = (rows + block_y - 1) // block_y
        loader.launch(
            self.k_outer_scaled,
            grid=(grid_x, grid_y, 1),
            block=(block_x, block_y, 1),
            params=[
                ctypes.c_uint64(a_ptr.value),
                ctypes.c_uint64(b_ptr.value),
                ctypes.c_uint64(out_ptr.value),
                ctypes.c_int(rows),
                ctypes.c_int(cols),
                ctypes.c_float(scale),
            ],
        )

    def _launch_matrix_axpy(self, dest_ptr, src_ptr, scale, n_elements):
        block = 256
        grid = (n_elements + block - 1) // block
        loader.launch(
            self.k_matrix_axpy,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(dest_ptr.value),
                ctypes.c_uint64(src_ptr.value),
                ctypes.c_float(scale),
                ctypes.c_int(n_elements),
            ],
        )

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #
    def train_sample(
        self,
        buffers: LoRADeviceBuffers,
        sample_index: int,
        dims: int,
        rank: int,
        alpha: float,
        learning_rate: float,
    ) -> float:
        offset = sample_index * dims * 4
        input_ptr = self._offset(buffers.inputs, offset)
        target_ptr = self._offset(buffers.targets, offset)

        # base_out = W_base @ input
        self._launch_matvec(
            self.k_matvec,
            buffers.base_matrix,
            input_ptr,
            buffers.base_out,
            dims,
            dims,
        )
        # Bx = B @ input
        self._launch_matvec(
            self.k_matvec,
            buffers.B,
            input_ptr,
            buffers.Bx,
            rank,
            dims,
        )
        # delta = A @ Bx
        self._launch_matvec(
            self.k_matvec,
            buffers.A,
            buffers.Bx,
            buffers.delta,
            dims,
            rank,
        )
        # output = base_out + alpha * delta
        self._launch_vec_add_scaled(
            buffers.base_out,
            buffers.delta,
            alpha,
            buffers.output,
            dims,
        )

        # Reset loss to zero
        zero = np.zeros(1, dtype=np.float32)
        loader.memcpy_htod(buffers.loss, zero.ctypes.data_as(ctypes.c_void_p), zero.nbytes)

        # error = output - target; accumulate squared loss
        self._launch_vec_sub_square(
            buffers.output,
            target_ptr,
            buffers.error,
            buffers.loss,
            dims,
        )

        # grad_A = alpha * error ⊗ Bx
        self._launch_outer(
            buffers.error,
            buffers.Bx,
            buffers.grad_A,
            dims,
            rank,
            alpha,
        )

        # AtE = A^T @ error
        self._launch_matvec(
            self.k_matvec_trans,
            buffers.A,
            buffers.error,
            buffers.AtE,
            dims,
            rank,
        )

        # grad_B = alpha * (AtE ⊗ input)
        self._launch_outer(
            buffers.AtE,
            input_ptr,
            buffers.grad_B,
            rank,
            dims,
            alpha,
        )

        # Update A and B: dest += -lr * grad
        self._launch_matrix_axpy(
            buffers.A,
            buffers.grad_A,
            -learning_rate,
            dims * rank,
        )
        self._launch_matrix_axpy(
            buffers.B,
            buffers.grad_B,
            -learning_rate,
            rank * dims,
        )

        # Fetch loss value (sum of squared errors)
        loss_host = np.zeros(1, dtype=np.float32)
        loader.memcpy_dtoh(loss_host.ctypes.data_as(ctypes.c_void_p), buffers.loss, loss_host.nbytes)
        loader.synchronize()
        return float(loss_host[0] / float(dims))

    # ------------------------------------------------------------------ #
    # Weight extraction
    # ------------------------------------------------------------------ #
    def fetch_weights(self, buffers: LoRADeviceBuffers, dims: int, rank: int) -> Tuple[np.ndarray, np.ndarray]:
        A_host = np.zeros((dims, rank), dtype=np.float32)
        B_host = np.zeros((rank, dims), dtype=np.float32)
        loader.memcpy_dtoh(A_host.ctypes.data_as(ctypes.c_void_p), buffers.A, A_host.nbytes)
        loader.memcpy_dtoh(B_host.ctypes.data_as(ctypes.c_void_p), buffers.B, B_host.nbytes)
        loader.synchronize()
        return A_host, B_host

