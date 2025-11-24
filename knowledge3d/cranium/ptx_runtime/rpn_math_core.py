"""RPN Math Core helpers for GPU-native LoRA updates."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import struct
from typing import Iterable, List, Sequence

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc
from knowledge3d.cranium.sovereign import loader


def _encode_pointer(ptr_value: int, rows: int, cols: int = 1) -> List[float]:
    """Encode device pointer metadata for OP_POINTER_LITERAL."""
    lo_bits = struct.unpack("<f", struct.pack("<I", ptr_value & 0xFFFFFFFF))[0]
    hi_bits = struct.unpack("<f", struct.pack("<I", (ptr_value >> 32) & 0xFFFFFFFF))[0]
    return [float(rows), float(cols), float(lo_bits), float(hi_bits)]


@dataclass
class DeviceTensor:
    ptr: loader.CUdeviceptr
    rows: int
    cols: int


class RPNMathCore:
    """Utility wrapper that drives Tier-3 RPN kernels for math ops."""

    def __init__(self) -> None:
        self.engine = AdvancedRPNEngine()

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    def _exec(self, op_codes: List[int], scalars: List[float]):
        op_arr = (ctypes.c_uint16 * len(op_codes))(*op_codes) if op_codes else (ctypes.c_uint16 * 1)(0)
        scalars_arr = (ctypes.c_float * len(scalars))(*scalars) if scalars else (ctypes.c_float * 1)(0.0)
        self.engine.reset_instance(0)
        return self.engine.execute_program(0, op_arr, scalars_arr)

    def vector_norm(self, tensor: DeviceTensor) -> float:
        op_codes = [ropc.OP_POINTER_LITERAL, ropc.OP_VEC_L2_NORM]
        scalars = _encode_pointer(int(tensor.ptr.value), tensor.rows * tensor.cols, 1)
        stack = self._exec(op_codes, scalars)
        if not stack:
            return 0.0
        return float(stack[-1][0])

    def fill(self, tensor: DeviceTensor, value: float) -> None:
        op_codes = [ropc.OP_POINTER_LITERAL, ropc.OP_LITERAL_SCALAR, ropc.OP_FILL_F32]
        scalars = _encode_pointer(int(tensor.ptr.value), tensor.rows * tensor.cols, 1) + [value]
        self._exec(op_codes, scalars)

    def vector_multiply(self, dest: DeviceTensor, other: DeviceTensor) -> None:
        op_codes = [
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
            ropc.OP_VECTOR_MUL_F32,
        ]
        scalars = (
            _encode_pointer(int(dest.ptr.value), dest.rows * dest.cols, 1)
            + _encode_pointer(int(other.ptr.value), other.rows * other.cols, 1)
        )
        self._exec(op_codes, scalars)

    def vec_add3(self, dest: DeviceTensor, a: DeviceTensor, b: DeviceTensor, c: DeviceTensor) -> None:
        op_codes = [
            ropc.OP_POINTER_LITERAL,  # a
            ropc.OP_POINTER_LITERAL,  # b
            ropc.OP_POINTER_LITERAL,  # c
            ropc.OP_POINTER_LITERAL,  # dest
            ropc.OP_TRM_VEC_ADD3_512,
        ]
        scalars = (
            _encode_pointer(int(a.ptr.value), a.rows * a.cols, 1)
            + _encode_pointer(int(b.ptr.value), b.rows * b.cols, 1)
            + _encode_pointer(int(c.ptr.value), c.rows * c.cols, 1)
            + _encode_pointer(int(dest.ptr.value), dest.rows * dest.cols, 1)
        )
        self._exec(op_codes, scalars)

    def matmul(self, dest: DeviceTensor, a: DeviceTensor, b: DeviceTensor) -> None:
        op_codes = [
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
            ropc.OP_MATMUL_SMALL,
        ]
        scalars = (
            _encode_pointer(int(dest.ptr.value), dest.rows, dest.cols)
            + _encode_pointer(int(a.ptr.value), a.rows, a.cols)
            + _encode_pointer(int(b.ptr.value), b.rows, b.cols)
        )
        self._exec(op_codes, scalars)

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_device(array: np.ndarray) -> loader.CUdeviceptr:
        data = [float(x) for x in array]
        nbytes = len(data) * ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * len(data))(*data)
        ptr = loader.gpu_malloc(nbytes)
        loader.memcpy_htod(ptr, ctypes.cast(buf, ctypes.c_void_p), nbytes)
        return ptr

    @staticmethod
    def copy_to_device(array: Sequence[float], ptr: loader.CUdeviceptr) -> None:
        data = [float(x) for x in array]
        buf = (ctypes.c_float * len(data))(*data)
        loader.memcpy_htod(ptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))

    @staticmethod
    def copy_to_host(ptr: loader.CUdeviceptr, array: Sequence[float]) -> List[float]:
        nbytes = len(array) * ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * len(array))()
        loader.memcpy_dtoh(ctypes.cast(buf, ctypes.c_void_p), ptr, nbytes)
        return [float(buf[i]) for i in range(len(array))]

    @staticmethod
    def free(ptr: loader.CUdeviceptr) -> None:
        loader.gpu_free(ptr)


__all__ = ["RPNMathCore", "DeviceTensor"]
