"""RPN Math Core helpers for GPU-native LoRA updates."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import random
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


class HostTensorF32:
    """ctypes-backed float32 tensor for sovereign host-side staging."""

    def __init__(
        self,
        rows: int,
        cols: int = 1,
        values: Iterable[float] | None = None,
        on_mutate=None,
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        self._on_mutate = on_mutate
        total = self.rows * self.cols
        self._buffer = (ctypes.c_float * total)()
        if values is not None:
            self.set_flat(values)

    @classmethod
    def zeros(cls, rows: int, cols: int = 1, on_mutate=None) -> "HostTensorF32":
        return cls(rows, cols, on_mutate=on_mutate)

    @classmethod
    def random_normal(
        cls,
        rows: int,
        cols: int,
        std: float = 0.01,
        on_mutate=None,
    ) -> "HostTensorF32":
        total = int(rows) * int(cols)
        return cls(rows, cols, (random.gauss(0.0, std) for _ in range(total)), on_mutate=on_mutate)

    @classmethod
    def from_array_like(
        cls,
        array: object,
        *,
        rows: int | None = None,
        cols: int | None = None,
        on_mutate=None,
    ) -> "HostTensorF32":
        inferred_rows, inferred_cols, values = _shape_and_values(array)
        target_rows = rows if rows is not None else inferred_rows
        target_cols = cols if cols is not None else inferred_cols
        if target_rows * target_cols != len(values):
            raise ValueError(
                f"Cannot reshape {len(values)} values into ({target_rows}, {target_cols})"
            )
        return cls(target_rows, target_cols, values, on_mutate=on_mutate)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def nbytes(self) -> int:
        return self.size * ctypes.sizeof(ctypes.c_float)

    @property
    def T(self) -> "HostTensorF32":
        return self.transpose()

    @property
    def flat(self) -> List[float]:
        return self.to_flat_list()

    def __len__(self) -> int:
        return self.rows

    def __iter__(self):
        for row in range(self.rows):
            yield self[row]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return float(self._buffer[row * self.cols + col])
        start = int(index) * self.cols
        return [float(self._buffer[start + col]) for col in range(self.cols)]

    def __iadd__(self, other):
        scalar = float(other)
        for idx in range(self.size):
            self._buffer[idx] = float(self._buffer[idx]) + scalar
        self._notify_mutation()
        return self

    def __isub__(self, other):
        scalar = float(other)
        for idx in range(self.size):
            self._buffer[idx] = float(self._buffer[idx]) - scalar
        self._notify_mutation()
        return self

    def copy(self) -> "HostTensorF32":
        clone = HostTensorF32(self.rows, self.cols)
        ctypes.memmove(clone.data_ptr, self.data_ptr, self.nbytes)
        return clone

    def fill(self, value: float) -> None:
        scalar = float(value)
        for idx in range(self.size):
            self._buffer[idx] = scalar
        self._notify_mutation()

    def set_flat(self, values: Iterable[float]) -> None:
        flat = [float(value) for value in values]
        if len(flat) != self.size:
            raise ValueError(f"Expected {self.size} values, got {len(flat)}")
        for idx, value in enumerate(flat):
            self._buffer[idx] = value
        self._notify_mutation()

    def copy_from(self, other: object) -> None:
        rows, cols, values = _shape_and_values(other)
        if (rows, cols) != self.shape:
            raise ValueError(f"Shape mismatch: {(rows, cols)} != {self.shape}")
        self.set_flat(values)

    def transpose(self) -> "HostTensorF32":
        out = HostTensorF32(self.cols, self.rows)
        for row in range(self.rows):
            for col in range(self.cols):
                out._buffer[col * out.cols + row] = self._buffer[row * self.cols + col]
        return out

    def scale_inplace(self, scale: float) -> None:
        factor = float(scale)
        for idx in range(self.size):
            self._buffer[idx] = float(self._buffer[idx]) * factor
        self._notify_mutation()

    def to_flat_list(self) -> List[float]:
        return [float(self._buffer[idx]) for idx in range(self.size)]

    def to_nested_list(self) -> List[List[float]]:
        return [self[row] for row in range(self.rows)]

    def tolist(self) -> List[List[float]]:
        return self.to_nested_list()

    def to_bytes(self) -> bytes:
        return ctypes.string_at(self.data_ptr, self.nbytes)

    def load_bytes(self, payload: bytes) -> None:
        if len(payload) != self.nbytes:
            raise ValueError(f"Expected {self.nbytes} bytes, got {len(payload)}")
        ctypes.memmove(self.data_ptr, payload, self.nbytes)
        self._notify_mutation()

    def set_mutation_callback(self, callback) -> None:
        self._on_mutate = callback

    def _notify_mutation(self) -> None:
        if callable(self._on_mutate):
            self._on_mutate()

    @property
    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer)


def _values_from_nested(sequence: Sequence[object]) -> tuple[int, int, List[float]]:
    if not sequence:
        return 0, 0, []
    first = sequence[0]
    if isinstance(first, (list, tuple)):
        rows = len(sequence)
        cols = len(first)
        values: List[float] = []
        for row in sequence:
            if not isinstance(row, (list, tuple)) or len(row) != cols:
                raise ValueError("Ragged nested sequence cannot be coerced to HostTensorF32")
            for item in row:
                values.append(float(item))
        return rows, cols, values
    values = [float(item) for item in sequence]
    return len(values), 1, values


def _shape_and_values(array: object) -> tuple[int, int, List[float]]:
    if isinstance(array, HostTensorF32):
        return array.rows, array.cols, array.to_flat_list()

    if isinstance(array, (list, tuple)):
        return _values_from_nested(array)

    shape = getattr(array, "shape", None)
    flat = getattr(array, "flat", None)
    if shape is not None and flat is not None:
        if len(shape) == 1:
            rows, cols = int(shape[0]), 1
        elif len(shape) == 2:
            rows, cols = int(shape[0]), int(shape[1])
        else:
            total = 1
            for dim in shape:
                total *= int(dim)
            rows, cols = int(total), 1
        return rows, cols, [float(value) for value in flat]

    tolist = getattr(array, "tolist", None)
    if callable(tolist):
        nested = tolist()
        if isinstance(nested, list):
            return _values_from_nested(nested)

    if hasattr(array, "__len__") and hasattr(array, "__getitem__"):
        values = [float(array[idx]) for idx in range(len(array))]
        return len(values), 1, values

    raise TypeError(f"Unsupported array-like object: {type(array).__name__}")


def _assign_values(array: object, rows: int, cols: int, values: Sequence[float]) -> None:
    if isinstance(array, HostTensorF32):
        if array.shape != (rows, cols):
            raise ValueError(f"Shape mismatch: {(rows, cols)} != {array.shape}")
        array.set_flat(values)
        return

    flat = getattr(array, "flat", None)
    shape = getattr(array, "shape", None)
    if flat is not None and shape is not None:
        if len(shape) == 1 and (rows, cols) != (int(shape[0]), 1):
            raise ValueError(f"Shape mismatch: {(rows, cols)} != {(int(shape[0]), 1)}")
        if len(shape) == 2 and (rows, cols) != (int(shape[0]), int(shape[1])):
            raise ValueError(f"Shape mismatch: {(rows, cols)} != {(int(shape[0]), int(shape[1]))}")
        for idx, value in enumerate(values):
            flat[idx] = value
        return

    if isinstance(array, list):
        if rows == len(array) and cols == 1:
            for idx, value in enumerate(values):
                array[idx] = value
            return
        if rows == len(array) and array and isinstance(array[0], list):
            cursor = 0
            for row in range(rows):
                if len(array[row]) != cols:
                    raise ValueError("Ragged nested list cannot receive tensor values")
                for col in range(cols):
                    array[row][col] = values[cursor]
                    cursor += 1
            return

    raise TypeError(f"Unsupported host destination: {type(array).__name__}")


class RPNMathCore:
    """Utility wrapper that drives Tier-3 RPN kernels for math ops."""

    _TRANSPOSE_MODULE = None
    _TRANSPOSE_KERNEL = None

    def __init__(self) -> None:
        self.engine = AdvancedRPNEngine()

    @classmethod
    def _get_transpose_kernel(cls):
        if cls._TRANSPOSE_KERNEL is not None:
            return cls._TRANSPOSE_KERNEL
        ptx_path = Path(__file__).resolve().parent.parent / "ptx" / "drawing_transform_ops.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Transpose PTX not found: {ptx_path}")
        if cls._TRANSPOSE_MODULE is None:
            cls._TRANSPOSE_MODULE = loader.load_module_from_file(str(ptx_path))
        cls._TRANSPOSE_KERNEL = loader.get_function(cls._TRANSPOSE_MODULE, "transpose_kernel")
        return cls._TRANSPOSE_KERNEL

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
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
            ropc.OP_POINTER_LITERAL,
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

    def transpose_2d(self, dest: DeviceTensor, src: DeviceTensor) -> None:
        if (dest.rows, dest.cols) != (src.cols, src.rows):
            raise ValueError(
                f"Transpose destination shape {(dest.rows, dest.cols)} incompatible with source {(src.rows, src.cols)}"
            )
        kernel = self._get_transpose_kernel()
        loader.launch(
            kernel,
            grid=((src.rows + 15) // 16, (src.cols + 15) // 16, 1),
            block=(16, 16, 1),
            params=[src.ptr, dest.ptr, ctypes.c_int(src.rows), ctypes.c_int(src.cols)],
        )
        loader.synchronize()

    def vector_norm_host(self, vector: object) -> float:
        rows, cols, _ = _shape_and_values(vector)
        ptr = self.to_device(vector)
        try:
            return self.vector_norm(DeviceTensor(ptr, rows, cols))
        finally:
            self.free(ptr)

    def matmul_host(self, a: object, b: object) -> HostTensorF32:
        a_rows, a_cols, _ = _shape_and_values(a)
        b_rows, b_cols, _ = _shape_and_values(b)
        if a_cols != b_rows:
            raise ValueError(f"Matmul shape mismatch: {(a_rows, a_cols)} x {(b_rows, b_cols)}")
        d_a = self.to_device(a)
        d_b = self.to_device(b)
        d_dest = loader.gpu_malloc(a_rows * b_cols * ctypes.sizeof(ctypes.c_float))
        try:
            self.matmul(
                DeviceTensor(d_dest, a_rows, b_cols),
                DeviceTensor(d_a, a_rows, a_cols),
                DeviceTensor(d_b, b_rows, b_cols),
            )
            host = HostTensorF32.zeros(a_rows, b_cols)
            self.copy_to_host(d_dest, host)
            return host
        finally:
            self.free(d_a)
            self.free(d_b)
            self.free(d_dest)

    def outer_product_host(self, left_vector: object, right_vector: object) -> HostTensorF32:
        left_rows, left_cols, _ = _shape_and_values(left_vector)
        right_rows, right_cols, _ = _shape_and_values(right_vector)
        if left_cols != 1:
            left = HostTensorF32.from_array_like(left_vector, rows=left_rows * left_cols, cols=1)
        else:
            left = HostTensorF32.from_array_like(left_vector)
        if right_cols != 1 and right_rows == 1:
            right = HostTensorF32.from_array_like(right_vector)
        else:
            right = HostTensorF32.from_array_like(right_vector, rows=1, cols=right_rows * right_cols)
        return self.matmul_host(left, right)

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def shape_of(array: object) -> tuple[int, int]:
        rows, cols, _ = _shape_and_values(array)
        return rows, cols

    @staticmethod
    def to_device(array: object) -> loader.CUdeviceptr:
        _, _, values = _shape_and_values(array)
        buf = (ctypes.c_float * len(values))(*values)
        nbytes = ctypes.sizeof(buf)
        ptr = loader.gpu_malloc(nbytes)
        loader.memcpy_htod(ptr, ctypes.cast(buf, ctypes.c_void_p), nbytes)
        return ptr

    @staticmethod
    def copy_to_device(array: object, ptr: loader.CUdeviceptr) -> None:
        _, _, values = _shape_and_values(array)
        buf = (ctypes.c_float * len(values))(*values)
        loader.memcpy_htod(ptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))

    @staticmethod
    def copy_to_host(ptr: loader.CUdeviceptr, array: object) -> None:
        rows, cols, _ = _shape_and_values(array)
        total = rows * cols
        nbytes = total * ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * total)()
        loader.memcpy_dtoh(ctypes.cast(buf, ctypes.c_void_p), ptr, nbytes)
        values = [float(buf[idx]) for idx in range(total)]
        _assign_values(array, rows, cols, values)

    @staticmethod
    def free(ptr: loader.CUdeviceptr) -> None:
        loader.gpu_free(ptr)


__all__ = ["RPNMathCore", "DeviceTensor", "HostTensorF32"]
