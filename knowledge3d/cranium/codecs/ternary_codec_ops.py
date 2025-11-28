"""
Sovereign ternary codec ops launcher (GPU-only, no numpy).

Supports basic quantise/dequantise kernels. DCT/MDCT batch kernels will be
added subsequently; missing ops raise loudly to avoid CPU fallbacks.
"""

from __future__ import annotations

import ctypes
from typing import List, Sequence

from knowledge3d.cranium.sovereign import loader


class TernaryCodecOps:
    """GPU launchers for ternary quantise/dequantise."""

    def __init__(self, threshold: float = 0.2) -> None:
        from pathlib import Path
        ptx_path = Path(__file__).parent.parent / "ptx" / "codec_ops.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"codec_ops.ptx not found at {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        self.quant_kernel = loader.get_function(module, "ternary_quant_kernel")
        self.dequant_kernel = loader.get_function(module, "ternary_dequant_kernel")
        self.dct_fwd_kernel = loader.get_function(module, "dct8x8_forward_blocks")
        self.dct_inv_kernel = loader.get_function(module, "dct8x8_inverse_blocks")
        self.mdct_kernel = loader.get_function(module, "mdct_forward_kernel")
        self.imdct_kernel = loader.get_function(module, "imdct_inverse_kernel")
        self.threshold = float(threshold)

    def quantize(self, values: Sequence[float], *, threshold: float | None = None) -> List[int]:
        """Quantise float sequence -> {-1,0,+1} on GPU."""
        n = len(values)
        if n == 0:
            return []
        thr = self.threshold if threshold is None else float(threshold)
        in_buf = (ctypes.c_float * n)(*values)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self.quant_kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                    ctypes.c_float(thr),
                ],
            )
            loader.synchronize()
            host_out = (ctypes.c_int * n)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_int))
            return [int(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dequantize(self, values: Sequence[int]) -> List[float]:
        """Dequantise {-1,0,+1} -> float on GPU."""
        n = len(values)
        if n == 0:
            return []
        ints = [int(round(v)) for v in values]
        in_buf = (ctypes.c_int * n)(*ints)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self.dequant_kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                ],
            )
            loader.synchronize()
            host_out = (ctypes.c_float * n)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_float))
            return [float(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def mdct_forward(self, frame: Sequence[float]) -> list[float]:
        """Run MDCT on a single frame (output length = frame_size/2)."""
        frame_size = len(frame)
        if frame_size == 0:
            return []
        return self.batch_mdct(frame, frame_size=frame_size)

    def imdct_inverse(self, coeffs: Sequence[float], frame_size: int) -> list[float]:
        """Run IMDCT for a single frame of coefficients (len = frame_size/2)."""
        if frame_size <= 0 or frame_size % 2 != 0:
            raise ValueError("frame_size must be positive even")
        expected = frame_size // 2
        if len(coeffs) != expected:
            raise ValueError(f"coeffs length {len(coeffs)} does not match frame_size/2 {expected}")
        return self.batch_imdct(coeffs, frame_size=frame_size)

    def batch_mdct(self, frames: Sequence[float], frame_size: int) -> list[float]:
        """Compute MDCT for contiguous frames (output length = frames * frame_size/2)."""
        if frame_size <= 0 or frame_size % 2 != 0:
            raise ValueError("frame_size must be positive even")
        n = len(frames)
        if n == 0:
            return []
        if n % frame_size != 0:
            raise ValueError("frames length must be multiple of frame_size")
        num_frames = n // frame_size
        half = frame_size // 2
        out_len = num_frames * half

        in_buf = (ctypes.c_float * n)(*frames)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(out_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid_x = (half + block[0] - 1) // block[0]
            shared_mem = frame_size * ctypes.sizeof(ctypes.c_float)
            for idx in range(num_frames):
                input_offset_bytes = idx * frame_size * ctypes.sizeof(ctypes.c_float)
                output_offset_bytes = idx * half * ctypes.sizeof(ctypes.c_float)
                loader.launch(
                    self.mdct_kernel,
                    grid=(grid_x, 1, 1),
                    block=block,
                    shared_mem=shared_mem,
                    params=[
                        ctypes.c_uint64(d_in.value + input_offset_bytes),
                        ctypes.c_uint64(d_out.value + output_offset_bytes),
                        ctypes.c_int(frame_size),
                    ],
                )
            loader.synchronize()
            host_out = (ctypes.c_float * out_len)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, out_len * ctypes.sizeof(ctypes.c_float))
            return [float(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def batch_imdct(self, frames: Sequence[float], frame_size: int) -> list[float]:
        """Inverse MDCT for contiguous frames (input len = frames * frame_size/2)."""
        if frame_size <= 0 or frame_size % 2 != 0:
            raise ValueError("frame_size must be positive even")
        n = len(frames)
        if n == 0:
            return []
        half = frame_size // 2
        if n % half != 0:
            raise ValueError("frames length must be multiple of frame_size/2")

        num_frames = n // half
        out_len = num_frames * frame_size

        in_buf = (ctypes.c_float * n)(*frames)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(out_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid_x = (frame_size + block[0] - 1) // block[0]
            shared_mem = half * ctypes.sizeof(ctypes.c_float)
            for idx in range(num_frames):
                input_offset_bytes = idx * half * ctypes.sizeof(ctypes.c_float)
                output_offset_bytes = idx * frame_size * ctypes.sizeof(ctypes.c_float)
                loader.launch(
                    self.imdct_kernel,
                    grid=(grid_x, 1, 1),
                    block=block,
                    shared_mem=shared_mem,
                    params=[
                        ctypes.c_uint64(d_in.value + input_offset_bytes),
                        ctypes.c_uint64(d_out.value + output_offset_bytes),
                        ctypes.c_int(frame_size),
                    ],
                )
            loader.synchronize()
            host_out = (ctypes.c_float * out_len)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, out_len * ctypes.sizeof(ctypes.c_float))
            return [float(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dct8_forward(self, blocks_flat: Sequence[float]) -> list[float]:
        """Run DCT8x8 on contiguous blocks (len must be multiple of 64)."""
        n = len(blocks_flat)
        if n == 0:
            return []
        if n % 64 != 0:
            raise ValueError("blocks_flat length must be multiple of 64")
        num_blocks = n // 64
        in_buf = (ctypes.c_float * n)(*blocks_flat)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (64, 1, 1)
            grid = (num_blocks, 1, 1)
            loader.launch(
                self.dct_fwd_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_int(num_blocks),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            loader.synchronize()
            host_out = (ctypes.c_float * n)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_float))
            return [float(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dct8_inverse(self, coeffs_flat: Sequence[float]) -> list[float]:
        """Run inverse DCT8x8 on contiguous blocks (len multiple of 64)."""
        n = len(coeffs_flat)
        if n == 0:
            return []
        if n % 64 != 0:
            raise ValueError("coeffs_flat length must be multiple of 64")
        num_blocks = n // 64
        in_buf = (ctypes.c_float * n)(*coeffs_flat)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (64, 1, 1)
            grid = (num_blocks, 1, 1)
            loader.launch(
                self.dct_inv_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_int(num_blocks),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            loader.synchronize()
            host_out = (ctypes.c_float * n)()
            loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_float))
            return [float(v) for v in host_out]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)


__all__ = ["TernaryCodecOps"]
