"""
Ternary quantization utilities shared by audio and video codecs.

This module provides GPU-only ternary quantisation/dequantisation (no CPU fallback),
plus simple entropy coding (run-length encoding of zeros), and convenience helpers.
"""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as np

from .ptx_bindings import TernaryQuantizer

_GPU_QUANT = None


def _get_gpu_quant():
    """Lazy initialization of GPU quantizer to avoid import-time CUDA context issues."""
    global _GPU_QUANT
    if _GPU_QUANT is None:
        _GPU_QUANT = TernaryQuantizer()
    return _GPU_QUANT


def _validate_array(coefficients) -> memoryview:
    """Validate coefficients, return a float32-contiguous buffer view."""
    if coefficients is None:
        raise ValueError("coefficients must not be None")
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(coefficients, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("coefficients must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("coefficients contain non-finite values")
    return arr


def quantize_ternary(
    coefficients: np.ndarray, threshold: float = 0.1, adaptive: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Quantize floating-point coefficients to ternary {-1, 0, +1} on GPU.

    Args:
        coefficients: Input array (any shape), converted to float32.
        threshold: Quantization threshold (explicit). Adaptive percentiles are not supported.
        adaptive: Must be False (GPU path only).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    arr = _validate_array(coefficients)
    thr = float(threshold)
    gpu_quant = _get_gpu_quant()
    if adaptive:
        max_abs = gpu_quant.max_abs(arr)
        thr = max(thr, max_abs * threshold)
    quantized = gpu_quant.quantize(arr, threshold=thr)

    sparsity = compute_sparsity(quantized)
    metadata: Dict[str, float] = {
        "threshold_used": thr,
        "adaptive": False,
        "sparsity": sparsity,
        "scale": thr,
    }
    return quantized, metadata


def dequantize_ternary(
    quantized: np.ndarray, scale: float = 1.0, metadata: Optional[dict] = None
) -> np.ndarray:
    """
    Reconstruct floating-point coefficients from ternary values.

    Args:
        quantized: int8 array with {-1, 0, +1}.
        scale: Reconstruction scale factor applied to all coefficients.
        metadata: Optional metadata (if present, `metadata['scale']` overrides scale).

    Returns:
        reconstructed: float32 array.
    """
    if quantized is None:
        raise ValueError("quantized must not be None")
    if quantized is None:
        raise ValueError("quantized must not be None")
    import numpy as np  # noqa: PLC0415

    q = np.asarray(quantized, dtype=np.int8)
    if q.size == 0:
        raise ValueError("quantized must not be empty")
    if metadata is not None and "scale" in metadata:
        scale = float(metadata["scale"])
    return _get_gpu_quant().dequantize(q, float(scale))


def compute_sparsity(quantized: np.ndarray) -> float:
    """
    Compute the fraction of zeros in a ternary array.

    Args:
        quantized: int8 array with values in {-1, 0, +1}.

    Returns:
        Sparsity ratio in [0, 1].
    """
    if quantized is None:
        raise ValueError("quantized must not be None")
    view = memoryview(quantized).cast("b")
    total = len(view)
    if total == 0:
        raise ValueError("quantized must not be empty")
    zero_count = sum(1 for v in view if v == 0)
    return zero_count / float(total)


def entropy_encode_ternary(quantized: np.ndarray) -> bytes:
    """
    Entropy-code ternary symbols with run-length encoding of zero runs.

    Encoding format (little-endian):
        [dims_count:uint8][dim_0:uint32]...[dim_n:uint32][payload...]

    Payload tokens:
        0x00 + <uint32 run_length> : run of zeros
        0x01                       : single +1
        0x02                       : single -1

    Args:
        quantized: int8 array with {-1, 0, +1}.

    Returns:
        Compressed bytes payload.
    """
    q = np.asarray(quantized, dtype=np.int8)
    if q.size == 0:
        raise ValueError("quantized must not be empty")
    if not np.all(np.isin(q, (-1, 0, 1))):
        raise ValueError("quantized must only contain -1, 0, or 1")

    flat = q.ravel()
    dims = q.shape
    if len(dims) > 255:
        raise ValueError("Cannot encode arrays with more than 255 dimensions")

    encoded = bytearray()
    encoded.append(len(dims))
    for dim in dims:
        encoded.extend(struct.pack("<I", int(dim)))

    i = 0
    n = flat.size
    while i < n:
        if flat[i] == 0:
            run_start = i
            while i < n and flat[i] == 0:
                i += 1
            run_length = i - run_start
            remaining = run_length
            while remaining > 0:
                chunk = min(remaining, 0xFFFF)
                encoded.append(0x00)
                encoded.extend(struct.pack("<H", chunk))
                remaining -= chunk
        else:
            encoded.append(0x01 if flat[i] == 1 else 0x02)
            i += 1
    return bytes(encoded)


def entropy_decode_ternary(encoded: bytes) -> np.ndarray:
    """
    Decode run-length encoded ternary symbols.

    Args:
        encoded: Bytes produced by `entropy_encode_ternary`.

    Returns:
        Decoded int8 NumPy array with original shape.
    """
    if encoded is None or len(encoded) == 0:
        raise ValueError("encoded must not be empty")

    view = memoryview(encoded)
    dims_count = view[0]
    offset = 1
    dims = []
    for _ in range(dims_count):
        if offset + 4 > len(view):
            raise ValueError("encoded payload is truncated while reading shape")
        dim = struct.unpack_from("<I", view, offset)[0]
        dims.append(dim)
        offset += 4

    values = []
    while offset < len(view):
        token = view[offset]
        offset += 1
        if token == 0x00:
            # May consist of multiple 16-bit chunks; stop consuming when next token is not part of length.
            if offset + 2 > len(view):
                raise ValueError("encoded payload is truncated while reading zero run")
            run_length = struct.unpack_from("<H", view, offset)[0]
            offset += 2
            values.extend([0] * run_length)
        elif token == 0x01:
            values.append(1)
        elif token == 0x02:
            values.append(-1)
        else:
            raise ValueError(f"Invalid token {token} in encoded payload")

    result = np.asarray(values, dtype=np.int8)
    if dims:
        expected_size = int(np.prod(dims))
        if result.size != expected_size:
            raise ValueError(
                f"Decoded size mismatch: expected {expected_size}, got {result.size}"
            )
        result = result.reshape(tuple(dims))
    return result
