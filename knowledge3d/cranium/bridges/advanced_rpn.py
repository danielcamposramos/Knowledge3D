"""Advanced (Tier-3) RPN bridge backed by modular_rpn_kernel_extended.ptx.

Sovereign resurrection (2026-04-18): the archived module
(``Old_Attempts/2026-04-18/knowledge3d/cranium/bridges/advanced_rpn.py``)
used ``numpy`` only to allocate small device-staging buffers (a 16-byte
header and a stack-size float32 buffer). The Absolute Sovereignty Purge
requires zero numpy in the hot path, so the numpy allocations have been
replaced with ``ctypes`` equivalents and the public surface is preserved
byte-for-byte.

This engine complements the standard sovereign RPN engine with
matrix-aware operations including matmul, transpose, determinant,
inverse, and trace. It maintains the same instance layout (MAX_INSTANCES
instances, deep stacks) while extending stack metadata to encode type
and dimensionality.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import struct

from knowledge3d.cranium.sovereign import loader
from .rpn_config import RPN_GRID_DIM, TIER3_BLOCK_DIM


@dataclass(frozen=True)
class StackEntryMetadata:
    """Decoded metadata for a single stack entry."""

    item_type: int
    rows: int
    cols: int
    row_index: int


def _decode_meta(value: float) -> StackEntryMetadata:
    """Decode packed metadata emitted in the W lane."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    item_type = bits & 0xFF
    rows = (bits >> 8) & 0xFF
    cols = (bits >> 16) & 0xFF
    row_index = (bits >> 24) & 0xFF
    return StackEntryMetadata(item_type, rows, cols, row_index)


class AdvancedRPNEngine:
    """Tier-3 RPN bridge that activates matrix-aware PTX operations."""

    MAX_INSTANCES = 18  # Tesla 3-6-9
    STACK_DEPTH = 69
    BLOCK_DIM = TIER3_BLOCK_DIM
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

        zeros = (ctypes.c_uint8 * (self.MAX_INSTANCES * self.INSTANCE_STRIDE))()
        loader.memcpy_htod(self._state, ctypes.cast(zeros, ctypes.c_void_p), ctypes.sizeof(zeros))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_program(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[Sequence[Sequence[float]]] = None,
        matrices: Optional[Sequence[float]] = None,
    ):
        """Execute a Tier-3 RPN program and return stack rows as [4-float]."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        op_codes_seq = list(op_codes)
        scalars_seq = list(scalars) if scalars is not None else []
        vectors_seq = [list(v) for v in vectors] if vectors is not None else []
        matrices_seq = list(matrices) if matrices is not None else []

        d_op_codes = self._maybe_upload_uint16(op_codes_seq)
        d_scalars = self._maybe_upload_float(scalars_seq)
        d_vectors = self._maybe_upload_float(vectors_seq)
        d_matrices = self._maybe_upload_float(matrices_seq)

        instance_offset = instance_id * self.INSTANCE_STRIDE

        try:
            loader.launch(
                self._kernel,
                grid=(RPN_GRID_DIM, 1, 1),
                block=(self.BLOCK_DIM, 1, 1),
                params=[
                    ctypes.c_uint32(instance_id),
                    ctypes.c_uint64(self._device_ptr(d_op_codes)),
                    ctypes.c_uint64(self._device_ptr(d_scalars)),
                    ctypes.c_uint64(self._device_ptr(d_vectors)),
                    ctypes.c_uint64(self._device_ptr(d_matrices)),
                    ctypes.c_uint64(self._state.value),
                    ctypes.c_uint32(len(op_codes_seq)),
                ],
            )
            loader.synchronize()
        finally:
            self._maybe_free(d_op_codes)
            self._maybe_free(d_scalars)
            self._maybe_free(d_vectors)
            self._maybe_free(d_matrices)

        header = (ctypes.c_uint32 * 4)()
        loader.memcpy_dtoh(
            ctypes.cast(header, ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset),
            ctypes.sizeof(header),
        )

        error_code = int(header[2])
        if error_code != 0:
            raise RuntimeError(f"Advanced RPN execution error: code {error_code}")

        stack_size = int(header[1])
        if stack_size == 0:
            return []

        stack_buffer = (ctypes.c_float * (stack_size * 4))()
        loader.memcpy_dtoh(
            ctypes.cast(stack_buffer, ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset + 16),
            stack_size * 16,
        )
        flat = [float(stack_buffer[i]) for i in range(stack_size * 4)]
        return [flat[i:i + 4] for i in range(0, len(flat), 4)]

    def execute_prebuilt(
        self,
        instance_id: int,
        d_op_codes,
        d_scalars,
        n_opcodes: int,
    ) -> List[List[float]]:
        """Execute using pre-uploaded opcode/scalar buffers.

        Returns rows of 4 floats (identical layout to ``execute_program``)
        — the legacy numpy return type was ``np.ndarray`` of shape
        ``(stack_size, 4)``; the pure-ctypes version returns a list of
        four-element lists with the same semantics.
        """
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        instance_offset = instance_id * self.INSTANCE_STRIDE

        loader.launch(
            self._kernel,
            grid=(RPN_GRID_DIM, 1, 1),
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

        header = (ctypes.c_uint32 * 4)()
        loader.memcpy_dtoh(
            ctypes.cast(header, ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset),
            ctypes.sizeof(header),
        )

        error_code = int(header[2])
        if error_code != 0:
            raise RuntimeError(f"Advanced RPN execution error: code {error_code}")

        stack_size = int(header[1])
        if stack_size == 0:
            return []

        stack_buffer = (ctypes.c_float * (stack_size * 4))()
        loader.memcpy_dtoh(
            ctypes.cast(stack_buffer, ctypes.c_void_p),
            ctypes.c_void_p(self._state.value + instance_offset + 16),
            stack_size * 16,
        )
        flat = [float(stack_buffer[i]) for i in range(stack_size * 4)]
        return [flat[i:i + 4] for i in range(0, len(flat), 4)]

    def reset_instance(self, instance_id: int) -> None:
        """Clear stack metadata for the given instance."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")

        header_zero = (ctypes.c_uint32 * 4)()
        offset = instance_id * self.INSTANCE_STRIDE
        loader.memcpy_htod(
            ctypes.c_void_p(self._state.value + offset),
            ctypes.cast(header_zero, ctypes.c_void_p),
            ctypes.sizeof(header_zero),
        )

    def cleanup(self) -> None:
        """Release device allocations associated with the Tier-3 engine."""
        if self._state is not None:
            loader.gpu_free(self._state)
            self._state = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def execute_scalar(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Optional[Sequence[float]] = None,
        vectors: Optional[Sequence[Sequence[float]]] = None,
        matrices: Optional[Sequence[float]] = None,
    ) -> float:
        """Execute a program and return the scalar at the top of the stack."""
        stack = self.execute_program(instance_id, op_codes, scalars, vectors, matrices)
        if len(stack) == 0:
            raise RuntimeError("Advanced RPN produced an empty stack")

        meta = _decode_meta(float(stack[-1][3]))
        if meta.item_type != self.TYPE_SCALAR:
            raise RuntimeError(f"Expected scalar result, found item_type={meta.item_type}")
        return float(stack[-1][0])

    def execute_matrix(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        *,
        output_shape: tuple[int, int],
        scalars: Optional[Sequence[float]] = None,
        matrices: Optional[Sequence[float]] = None,
    ):
        """Execute a program and retrieve the matrix result with given shape."""
        rows, cols = output_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("output_shape must contain positive dimensions")

        stack = self.execute_program(instance_id, op_codes, scalars, matrices=matrices)
        if len(stack) < rows:
            raise RuntimeError("Insufficient stack entries for requested matrix")

        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
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
            matrix[meta.row_index][0] = entry[0]
            if cols > 1:
                matrix[meta.row_index][1] = entry[1]
            if cols > 2:
                matrix[meta.row_index][2] = entry[2]

        return matrix

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _device_ptr(device_allocation):
        return device_allocation.value if device_allocation is not None else 0

    @staticmethod
    def _maybe_upload_float(array):
        if not array:
            return None
        flat: list[float] = []
        if isinstance(array[0], (list, tuple)):
            for row in array:
                flat.extend([float(x) for x in row])
        else:
            flat = [float(x) for x in array]
        buf = (ctypes.c_float * len(flat))(*flat)
        device_ptr = loader.gpu_malloc(ctypes.sizeof(buf))
        loader.memcpy_htod(device_ptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))
        return device_ptr

    @staticmethod
    def _maybe_upload_uint16(array):
        if not array:
            return None
        buf = (ctypes.c_uint16 * len(array))(*[int(x) for x in array])
        device_ptr = loader.gpu_malloc(ctypes.sizeof(buf))
        loader.memcpy_htod(device_ptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))
        return device_ptr

    @staticmethod
    def _maybe_free(device_allocation):
        if device_allocation is not None:
            loader.gpu_free(device_allocation)


__all__ = ["AdvancedRPNEngine", "StackEntryMetadata"]
