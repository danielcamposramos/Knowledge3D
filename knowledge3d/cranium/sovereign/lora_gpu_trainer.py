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

    # Dataset (kept on HOST to avoid D2D copy issues)
    inputs_host: np.ndarray
    targets_host: np.ndarray

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

    # Batched workspaces
    batch_inputs: loader.CUdeviceptr
    batch_targets: loader.CUdeviceptr
    batch_base_out: loader.CUdeviceptr
    batch_delta: loader.CUdeviceptr
    batch_output: loader.CUdeviceptr
    batch_error: loader.CUdeviceptr
    batch_Bx: loader.CUdeviceptr
    batch_AtE: loader.CUdeviceptr
    batch_losses: loader.CUdeviceptr
    max_batch: int


class LoRAGPUEngine:
    """Sovereign GPU trainer for low-rank adapters."""

    def __init__(self):
        # Ensure CUDA context is initialized
        loader._ensure_init()

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
        self.k_matvec_batch = loader.get_function(self.module, "matvec_batch")
        self.k_matvec_trans_batch = loader.get_function(self.module, "matvec_transpose_batch")
        self.k_vec_add_scaled_batch = loader.get_function(self.module, "vec_add_scaled_batch")
        self.k_vec_sub_square_batch = loader.get_function(self.module, "vec_sub_square_batch")
        self.k_outer_batch = loader.get_function(self.module, "outer_product_accumulate_batch")
        self._loss_host_capacity = 0
        self._loss_host_buffer: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _offset(ptr: loader.CUdeviceptr, offset_bytes: int) -> loader.CUdeviceptr:
        return loader.CUdeviceptr(ptr.value + offset_bytes)

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32, order="C")

    @staticmethod
    def _copy_sample(
        src_ptr: loader.CUdeviceptr,
        dst_ptr: loader.CUdeviceptr,
        dims: int,
    ) -> None:
        loader.memcpy_dtod(dst_ptr, src_ptr, dims * 4)

    def _loss_buffer(self, required: int) -> np.ndarray:
        if self._loss_host_buffer is None or self._loss_host_capacity < required:
            self._loss_host_buffer = np.zeros(required, dtype=np.float32)
            self._loss_host_capacity = required
        return self._loss_host_buffer

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
        max_batch: int,
    ) -> LoRADeviceBuffers:
        dims = base_matrix.shape[0]
        rank = B.shape[0]
        n_samples = inputs.shape[0]

        # Upload weights to GPU
        base_gpu = loader.gpu_malloc(base_matrix.nbytes)
        loader.memcpy_htod(base_gpu, ctypes.c_void_p(base_matrix.ctypes.data), base_matrix.nbytes)

        A_gpu = loader.gpu_malloc(A.nbytes)
        loader.memcpy_htod(A_gpu, ctypes.c_void_p(A.ctypes.data), A.nbytes)

        B_gpu = loader.gpu_malloc(B.nbytes)
        loader.memcpy_htod(B_gpu, ctypes.c_void_p(B.ctypes.data), B.nbytes)

        # Keep dataset on HOST (avoid D2D copy issues)
        inputs_host = self._to_float32(inputs)
        targets_host = self._to_float32(targets)

        def alloc(size):
            return loader.gpu_malloc(size)

        vector_bytes = dims * 4
        rank_bytes = rank * 4
        grad_A_bytes = dims * rank * 4
        grad_B_bytes = rank * dims * 4
        batch_vector_bytes = vector_bytes * max_batch
        batch_rank_bytes = rank_bytes * max_batch

        buffers = LoRADeviceBuffers(
            base_matrix=base_gpu,
            A=A_gpu,
            B=B_gpu,
            inputs_host=inputs_host,
            targets_host=targets_host,
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
            batch_inputs=alloc(batch_vector_bytes),
            batch_targets=alloc(batch_vector_bytes),
            batch_base_out=alloc(batch_vector_bytes),
            batch_delta=alloc(batch_vector_bytes),
            batch_output=alloc(batch_vector_bytes),
            batch_error=alloc(batch_vector_bytes),
            batch_Bx=alloc(batch_rank_bytes),
            batch_AtE=alloc(batch_rank_bytes),
            batch_losses=alloc(max_batch * 4),
            max_batch=max_batch,
        )
        return buffers

    def free_buffers(self, buffers: LoRADeviceBuffers) -> None:
        for attr in vars(buffers):
            value = getattr(buffers, attr)
            if isinstance(value, loader.CUdeviceptr):
                loader.gpu_free(value)
        setattr(buffers, "n_samples", 0)

    def _prepare_batch(
        self,
        buffers: LoRADeviceBuffers,
        batch_indices: np.ndarray,
        dims: int,
    ) -> int:
        """Prepare batch by uploading samples from host to GPU.

        Uses H2D copy instead of D2D to avoid context issues.
        """
        batch_size = len(batch_indices)

        # Gather batch samples from host arrays (efficient NumPy indexing)
        batch_inputs_host = buffers.inputs_host[batch_indices]
        batch_targets_host = buffers.targets_host[batch_indices]

        # Upload to GPU batch buffers (H2D copy - works reliably)
        loader.memcpy_htod(
            buffers.batch_inputs,
            ctypes.c_void_p(batch_inputs_host.ctypes.data),
            batch_inputs_host.nbytes
        )
        loader.memcpy_htod(
            buffers.batch_targets,
            ctypes.c_void_p(batch_targets_host.ctypes.data),
            batch_targets_host.nbytes
        )

        return batch_size

    def update_dataset(self, buffers: LoRADeviceBuffers, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Update the host-side dataset (for epoch shuffling)."""
        buffers.inputs_host = self._to_float32(inputs)
        buffers.targets_host = self._to_float32(targets)
        buffers.n_samples = len(inputs)

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

    def _launch_matvec_batch(self, func, W_ptr, X_ptr, Y_ptr, rows, cols, batch):
        total = rows * batch
        block = 256
        grid = (total + block - 1) // block
        loader.launch(
            func,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(W_ptr.value),
                ctypes.c_uint64(X_ptr.value),
                ctypes.c_uint64(Y_ptr.value),
                ctypes.c_int(rows),
                ctypes.c_int(cols),
                ctypes.c_int(batch),
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

    def _launch_vec_add_scaled_batch(self, a_ptr, b_ptr, scale, out_ptr, n, batch):
        total = n * batch
        block = 256
        grid = (total + block - 1) // block
        loader.launch(
            self.k_vec_add_scaled_batch,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(a_ptr.value),
                ctypes.c_uint64(b_ptr.value),
                ctypes.c_float(scale),
                ctypes.c_uint64(out_ptr.value),
                ctypes.c_int(n),
                ctypes.c_int(batch),
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

    def _launch_vec_sub_square_batch(self, pred_ptr, target_ptr, error_ptr, losses_ptr, dims, batch):
        total = dims * batch
        block = 256
        grid = (total + block - 1) // block
        loader.launch(
            self.k_vec_sub_square_batch,
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            params=[
                ctypes.c_uint64(pred_ptr.value),
                ctypes.c_uint64(target_ptr.value),
                ctypes.c_uint64(error_ptr.value),
                ctypes.c_uint64(losses_ptr.value),
                ctypes.c_int(dims),
                ctypes.c_int(batch),
            ],
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

    def _launch_outer_batch(self, A_ptr, B_ptr, out_ptr, rows, cols, batch, scale):
        block_x = 16
        block_y = 16
        grid_x = (cols + block_x - 1) // block_x
        grid_y = (rows + block_y - 1) // block_y
        loader.launch(
            self.k_outer_batch,
            grid=(grid_x, grid_y, 1),
            block=(block_x, block_y, 1),
            params=[
                ctypes.c_uint64(A_ptr.value),
                ctypes.c_uint64(B_ptr.value),
                ctypes.c_uint64(out_ptr.value),
                ctypes.c_int(rows),
                ctypes.c_int(cols),
                ctypes.c_int(batch),
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

    @staticmethod
    def _zero_f32(ptr: loader.CUdeviceptr, count: int) -> None:
        """Zero device memory using H2D copy (more reliable than memset_d32)."""
        if count <= 0:
            return
        # Use H2D copy instead of memset_d32 to avoid context issues
        # This matches the pattern used successfully in consolidation
        zeros = np.zeros(count, dtype=np.float32)
        loader.memcpy_htod(ptr, ctypes.c_void_p(zeros.ctypes.data), zeros.nbytes)

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

    def train_batch(
        self,
        buffers: LoRADeviceBuffers,
        batch_indices: np.ndarray,
        dims: int,
        rank: int,
        alpha: float,
        learning_rate: float,
    ) -> float:
        if batch_indices.size == 0:
            return 0.0
        if batch_indices.size > buffers.max_batch:
            raise ValueError(f"Batch size {batch_indices.size} exceeds capacity {buffers.max_batch}")
        batch_size = self._prepare_batch(buffers, batch_indices, dims)

        # Forward
        self._launch_matvec_batch(
            self.k_matvec_batch,
            buffers.base_matrix,
            buffers.batch_inputs,
            buffers.batch_base_out,
            dims,
            dims,
            batch_size,
        )
        self._launch_matvec_batch(
            self.k_matvec_batch,
            buffers.B,
            buffers.batch_inputs,
            buffers.batch_Bx,
            rank,
            dims,
            batch_size,
        )
        self._launch_matvec_batch(
            self.k_matvec_batch,
            buffers.A,
            buffers.batch_Bx,
            buffers.batch_delta,
            dims,
            rank,
            batch_size,
        )
        self._launch_vec_add_scaled_batch(
            buffers.batch_base_out,
            buffers.batch_delta,
            alpha,
            buffers.batch_output,
            dims,
            batch_size,
        )

        # Loss / error
        self._zero_f32(buffers.batch_losses, buffers.max_batch)
        self._launch_vec_sub_square_batch(
            buffers.batch_output,
            buffers.batch_targets,
            buffers.batch_error,
            buffers.batch_losses,
            dims,
            batch_size,
        )

        # Gradients
        self._zero_f32(buffers.grad_A, dims * rank)
        self._zero_f32(buffers.grad_B, rank * dims)

        self._launch_outer_batch(
            buffers.batch_error,
            buffers.batch_Bx,
            buffers.grad_A,
            dims,
            rank,
            batch_size,
            alpha,
        )

        self._launch_matvec_batch(
            self.k_matvec_trans_batch,
            buffers.A,
            buffers.batch_error,
            buffers.batch_AtE,
            dims,
            rank,
            batch_size,
        )

        self._launch_outer_batch(
            buffers.batch_AtE,
            buffers.batch_inputs,
            buffers.grad_B,
            rank,
            dims,
            batch_size,
            alpha,
        )

        scale = -learning_rate / float(batch_size)
        self._launch_matrix_axpy(buffers.A, buffers.grad_A, scale, dims * rank)
        self._launch_matrix_axpy(buffers.B, buffers.grad_B, scale, rank * dims)

        # Fetch batch losses
        loss_host = self._loss_buffer(batch_size)
        loader.memcpy_dtoh(
            loss_host.ctypes.data_as(ctypes.c_void_p),
            buffers.batch_losses,
            batch_size * 4,
        )
        loader.synchronize()
        total_loss = float(loss_host[:batch_size].sum())
        return total_loss / float(dims * batch_size)

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
