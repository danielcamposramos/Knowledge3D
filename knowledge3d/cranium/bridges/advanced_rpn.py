"""
Advanced (Tier-3) RPN bridge backed by modular_rpn_kernel_extended.ptx.

This engine complements the standard sovereign RPN engine with matrix-aware
operations including matmul, transpose, determinant, inverse, and trace. It
maintains the same instance layout (15 instances, 64-deep stacks) while
extending stack metadata to encode type and dimensionality.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from knowledge3d.cranium.sovereign import loader


@dataclass(frozen=True)
class StackEntryMetadata:
    """Decoded metadata for a single stack entry."""

    item_type: int
    rows: int
    cols: int
    row_index: int


def _decode_meta(value: float) -> StackEntryMetadata:
    """Decode packed metadata emitted in the W lane."""
    bits = np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0]
    item_type = bits & 0xFF
    rows = (bits >> 8) & 0xFF
    cols = (bits >> 16) & 0xFF
    row_index = (bits >> 24) & 0xFF
    return StackEntryMetadata(item_type, rows, cols, row_index)


class AdvancedRPNEngine:
    """Tier-3 RPN bridge that activates matrix-aware PTX operations."""

    MAX_INSTANCES = 15
    STACK_DEPTH = 64
    BLOCK_DIM = 256
    INSTANCE_STRIDE = 1040  # bytes per instance (header + 64*float4)

    TYPE_SCALAR = 0
    TYPE_VECTOR = 1
    TYPE_MATRIX_ROW = 2
    TYPE_TENSOR = 3

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel_extended.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Advanced RPN PTX kernel not found: {ptx_path}")

        self._kernel = loader.load_ptx_file(str(ptx_path), "modular_rpn_kernel_extended")
        self._state = loader.gpu_malloc(self.MAX_INSTANCES * self.INSTANCE_STRIDE)

        zeros = np.zeros(self.MAX_INSTANCES * self.INSTANCE_STRIDE, dtype=np.uint8)
        loader.memcpy_htod(self._state, zeros.ctypes.data_as(ctypes.c_void_p), zeros.nbytes)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_program(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[np.ndarray] = None,
        matrices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute a Tier-3 RPN program and return the raw stack buffer."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        op_codes_np = np.ascontiguousarray(op_codes, dtype=np.uint16)
        scalars_np = np.ascontiguousarray(scalars if scalars is not None else [], dtype=np.float32)
        vectors_np = np.ascontiguousarray(vectors if vectors is not None else [], dtype=np.float32)
        matrices_np = np.ascontiguousarray(matrices if matrices is not None else [], dtype=np.float32)

        d_op_codes = self._maybe_upload(op_codes_np)
        d_scalars = self._maybe_upload(scalars_np)
        d_vectors = self._maybe_upload(vectors_np)
        d_matrices = self._maybe_upload(matrices_np)

        instance_offset = instance_id * self.INSTANCE_STRIDE

        try:
            loader.launch(
                self._kernel,
                grid=(1, 1, 1),
                block=(self.BLOCK_DIM, 1, 1),
                params=[
                    ctypes.c_uint32(instance_id),
                    ctypes.c_uint64(self._device_ptr(d_op_codes)),
                    ctypes.c_uint64(self._device_ptr(d_scalars)),
                    ctypes.c_uint64(self._device_ptr(d_vectors)),
                    ctypes.c_uint64(self._device_ptr(d_matrices)),
                    ctypes.c_uint64(self._state.value),
                    ctypes.c_uint32(len(op_codes_np)),
                ],
            )
            loader.synchronize()
        finally:
            self._maybe_free(d_op_codes)
            self._maybe_free(d_scalars)
            self._maybe_free(d_vectors)
            self._maybe_free(d_matrices)

        header = np.zeros(4, dtype=np.uint32)
        loader.memcpy_dtoh(
            header.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset),
            header.nbytes,
        )

        error_code = int(header[2])
        if error_code != 0:
            raise RuntimeError(f"Advanced RPN execution error: code {error_code}")

        stack_size = int(header[1])
        if stack_size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        stack_buffer = np.zeros((stack_size, 4), dtype=np.float32)
        loader.memcpy_dtoh(
            stack_buffer.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset + 16),
            stack_size * 16,
        )
        return stack_buffer

    def execute_prebuilt(
        self,
        instance_id: int,
        d_op_codes,
        d_scalars,
        n_opcodes: int,
    ) -> np.ndarray:
        """Execute using pre-uploaded opcode/scalar buffers."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        instance_offset = instance_id * self.INSTANCE_STRIDE

        loader.launch(
            self._kernel,
            grid=(1, 1, 1),
            block=(self.BLOCK_DIM, 1, 1),
            params=[
                ctypes.c_uint32(instance_id),
                ctypes.c_uint64(self._device_ptr(d_op_codes)),
                ctypes.c_uint64(self._device_ptr(d_scalars)),
                ctypes.c_uint64(0),
                ctypes.c_uint64(0),
                ctypes.c_uint64(self._state.value),
                ctypes.c_uint32(n_opcodes),
            ],
        )
        loader.synchronize()

        header = np.zeros(4, dtype=np.uint32)
        loader.memcpy_dtoh(
            header.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset),
            header.nbytes,
        )

        error_code = int(header[2])
        if error_code != 0:
            raise RuntimeError(f"Advanced RPN execution error: code {error_code}")

        stack_size = int(header[1])
        if stack_size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        stack_buffer = np.zeros((stack_size, 4), dtype=np.float32)
        loader.memcpy_dtoh(
            stack_buffer.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset + 16),
            stack_size * 16,
        )
        return stack_buffer

    def reset_instance(self, instance_id: int) -> None:
        """Clear stack metadata for the given instance."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        header_zero = np.zeros(4, dtype=np.uint32)
        offset = instance_id * self.INSTANCE_STRIDE
        loader.memcpy_htod(
            ctypes.c_void_p(self._state.value + offset),
            header_zero.ctypes.data_as(ctypes.c_void_p),
            header_zero.nbytes,
        )

    def execute_scalar(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[np.ndarray] = None,
        matrices: Optional[np.ndarray] = None,
    ) -> float:
        """Execute a program and return the scalar at the top of the stack."""
        stack = self.execute_program(instance_id, op_codes, scalars, vectors, matrices)
        if len(stack) == 0:
            raise RuntimeError("Advanced RPN produced an empty stack")

        meta = _decode_meta(float(stack[-1, 3]))
        if meta.item_type != self.TYPE_SCALAR:
            raise RuntimeError(f"Expected scalar result, found item_type={meta.item_type}")
        return float(stack[-1, 0])

    def execute_matrix(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        *,
        output_shape: tuple[int, int],
        scalars: Optional[Sequence[float]] = None,
        matrices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute a program and retrieve the matrix result with given shape."""
        rows, cols = output_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("output_shape must contain positive dimensions")

        stack = self.execute_program(instance_id, op_codes, scalars, matrices=matrices)
        if len(stack) < rows:
            raise RuntimeError("Insufficient stack entries for requested matrix")

        matrix = np.zeros((rows, cols), dtype=np.float32)
        relevant_entries = stack[-rows:]

        for entry in relevant_entries:
            meta = _decode_meta(float(entry[3]))
            if meta.item_type != self.TYPE_MATRIX_ROW:
                raise RuntimeError("Stack entry does not represent a matrix row")
            if meta.rows != rows or meta.cols != cols:
                raise RuntimeError(
                    f"Matrix metadata mismatch (expected {rows}x{cols}, found {meta.rows}x{meta.cols})"
                )
            if meta.row_index >= rows:
                raise RuntimeError(f"Row index {meta.row_index} exceeds output rows {rows}")
            matrix[meta.row_index, 0] = entry[0]
            if cols > 1:
                matrix[meta.row_index, 1] = entry[1]
            if cols > 2:
                matrix[meta.row_index, 2] = entry[2]

        return matrix

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _device_ptr(device_allocation: Optional[ctypes.c_void_p]) -> int:
        return device_allocation.value if device_allocation is not None else 0

    @staticmethod
    def _maybe_upload(array: np.ndarray) -> Optional[ctypes.c_void_p]:
        if array.size == 0:
            return None
        device_ptr = loader.gpu_malloc(array.nbytes)
        loader.memcpy_htod(device_ptr, array.ctypes.data_as(ctypes.c_void_p), array.nbytes)
        return device_ptr

    @staticmethod
    def _maybe_free(device_allocation: Optional[ctypes.c_void_p]) -> None:
        if device_allocation is not None:
            loader.gpu_free(device_allocation)
