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

import numpy as np

from knowledge3d.cranium.sovereign import loader


class TernaryVector:
    """GPU-resident ternary vector with packed 2-bit storage."""

    def __init__(self, values: Sequence[int]):
        quantized = self._normalize_values(values)
        self.length = int(quantized.size)
        self.packed_host = self._pack_host(quantized)
        self.device_ptr = self._upload(self.packed_host)

    # ------------------------------------------------------------------ #
    # Packing / unpacking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_value(v: int) -> int:
        """Map arbitrary numeric input to {-1,0,+1}."""
        if v in (-1, 0, 1):
            return int(v)
        # Small floats in [-1,1] get rounded to nearest ternary value
        if -1.0 <= float(v) <= 1.0:
            rounded = int(round(float(v)))
            return max(-1, min(1, rounded))
        # For byte-like ranges (e.g., 0-255), use 3-level quantization
        fv = float(v)
        if fv < 85.0:
            return 0
        if fv > 170.0:
            return 1
        return -1

    @classmethod
    def _normalize_values(cls, values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        flat = arr.reshape(-1).astype(np.float32, copy=False)

        out = np.full(flat.shape, -1, dtype=np.int8)
        exact = (flat == -1.0) | (flat == 0.0) | (flat == 1.0)
        if np.any(exact):
            out[exact] = flat[exact].astype(np.int8, copy=False)

        small = (~exact) & (flat >= -1.0) & (flat <= 1.0)
        if np.any(small):
            out[small] = np.clip(np.rint(flat[small]), -1, 1).astype(np.int8, copy=False)

        non_small = ~(exact | small)
        if np.any(non_small):
            low = non_small & (flat < 85.0)
            high = non_small & (flat > 170.0)
            out[low] = 0
            out[high] = 1
        return out

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
        arr = np.asarray(values, dtype=np.int8).reshape(-1)
        if arr.size == 0:
            return b""
        codes = np.where(arr < 0, 0b10, np.where(arr > 0, 0b01, 0b00)).astype(np.uint8, copy=False)
        pad = (-codes.size) % 4
        if pad:
            codes = np.pad(codes, (0, pad), constant_values=0)
        packed = (
            codes[0::4]
            | (codes[1::4] << 2)
            | (codes[2::4] << 4)
            | (codes[3::4] << 6)
        ).astype(np.uint8, copy=False)
        return packed.tobytes()

    def _upload(self, data: bytes) -> loader.CUdeviceptr:
        size = len(data)
        d_ptr = loader.gpu_malloc(max(size, 1))
        if size:
            buf = (ctypes.c_ubyte * size).from_buffer_copy(data)
            loader.memcpy_htod(d_ptr, ctypes.cast(buf, ctypes.c_void_p), size)
        return d_ptr

    @staticmethod
    def _unpack_bytes(host_bytes: bytes, length: int) -> np.ndarray:
        byte_arr = np.frombuffer(host_bytes, dtype=np.uint8)
        if byte_arr.size == 0:
            return np.empty((0,), dtype=np.int8)
        codes = np.empty(byte_arr.size * 4, dtype=np.uint8)
        codes[0::4] = byte_arr & 0b11
        codes[1::4] = (byte_arr >> 2) & 0b11
        codes[2::4] = (byte_arr >> 4) & 0b11
        codes[3::4] = (byte_arr >> 6) & 0b11
        values = np.zeros(codes.shape, dtype=np.int8)
        values[codes == 0b01] = 1
        values[codes == 0b10] = -1
        return values[:length]

    def to_python(self) -> List[int]:
        """Download and unpack to Python list (debug/validation only)."""
        size = len(self.packed_host)
        host_buf = (ctypes.c_ubyte * size)()
        if size:
            loader.memcpy_dtoh(ctypes.cast(host_buf, ctypes.c_void_p), self.device_ptr, size)
        values = self._unpack_bytes(bytes(host_buf), self.length)
        return values.astype(int, copy=False).tolist()

    def to_numpy(self) -> np.ndarray:
        """Unpack from the immutable host cache to a contiguous int8 numpy array."""
        return self._unpack_bytes(self.packed_host, self.length).copy()

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
