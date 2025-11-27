"""
GPU-resident ternary vectors {-1, 0, +1} with packed 2-bit storage.

Packing scheme (per value):
    -1 -> 0b10
     0 -> 0b00
    +1 -> 0b01
Values are packed 4 per byte (low → high bits). Storage lives on GPU via the
sovereign loader; a host copy of the packed bytes is kept for hashing/dedup.
"""

from __future__ import annotations

import ctypes
from typing import List, Sequence, Tuple

from knowledge3d.cranium.sovereign import loader


class TernaryVector:
    """GPU-resident ternary vector with packed 2-bit storage."""

    def __init__(self, values: Sequence[int]):
        for v in values:
            if v not in (-1, 0, 1):
                raise ValueError(f"Ternary values must be -1, 0, or +1, got {v}")
        self.length = len(values)
        self.packed_host = self._pack_host(values)
        self.device_ptr = self._upload(self.packed_host)

    # ------------------------------------------------------------------ #
    # Packing / unpacking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_value(v: int) -> int:
        if v == -1:
            return 0b10
        if v == 0:
            return 0b00
        return 0b01  # +1

    @staticmethod
    def _decode_value(code: int) -> int:
        if code == 0b01:
            return 1
        if code == 0b10:
            return -1
        return 0

    def _pack_host(self, values: Sequence[int]) -> bytes:
        packed: List[int] = []
        acc = 0
        count = 0
        for v in values:
            code = self._encode_value(int(v))
            shift = (count % 4) * 2
            acc |= (code & 0b11) << shift
            count += 1
            if count % 4 == 0:
                packed.append(acc & 0xFF)
                acc = 0
        if count % 4 != 0:
            packed.append(acc & 0xFF)
        return bytes(packed)

    def _upload(self, data: bytes) -> loader.CUdeviceptr:
        size = len(data)
        d_ptr = loader.gpu_malloc(max(size, 1))
        if size:
            buf = (ctypes.c_ubyte * size).from_buffer_copy(data)
            loader.memcpy_htod(d_ptr, ctypes.cast(buf, ctypes.c_void_p), size)
        return d_ptr

    def to_python(self) -> List[int]:
        """Download and unpack to Python list (debug/validation only)."""
        size = len(self.packed_host)
        host_buf = (ctypes.c_ubyte * size)()
        if size:
            loader.memcpy_dtoh(ctypes.cast(host_buf, ctypes.c_void_p), self.device_ptr, size)
        out: List[int] = []
        total = self.length
        for byte in host_buf:
            for shift in (0, 2, 4, 6):
                if len(out) >= total:
                    break
                code = (byte >> shift) & 0b11
                out.append(self._decode_value(code))
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        try:
            if hasattr(self, "device_ptr") and self.device_ptr:
                loader.gpu_free(self.device_ptr)
        except Exception:
            pass


class TernaryTensor:
    """Multi-dimensional ternary tensor wrapper."""

    def __init__(self, shape: Tuple[int, ...], values: TernaryVector):
        self.shape = shape
        self.values = values
        if self.values.length != self._numel():
            raise ValueError(
                f"Shape {shape} requires {self._numel()} values, got {self.values.length}"
            )

    def _numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def to_python(self) -> List[int]:
        return self.values.to_python()


__all__ = ["TernaryVector", "TernaryTensor"]
