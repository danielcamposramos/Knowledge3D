"""Sovereign ternary codec ops launcher (pure-ctypes over codec_ops.ptx).

Resurrected from the NotImplementedError stub on 2026-04-18 as part of the
Absolute Sovereignty Purge follow-through. Replaces the numpy-native
TernaryCodecOps archived at Old_Attempts/2026-04-18/ with a pure-ctypes
launcher — no numpy, no cupy, no scipy, no sympy, no torch.

Callers:
    knowledge3d.cranium.bridges.tiered_rpn.TieredRPNEngine.execute_codec
    (dct8, idct8, quant, dequant, mdct, imdct, batch_mdct, batch_dct,
     reshape_blocks, blocks_to_grid)

Kernels sourced from: knowledge3d/cranium/ptx/codec_ops.ptx
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Dict, Sequence, Union

from knowledge3d.cranium.sovereign import loader


class TernaryCodecOps:
    """Pure-ctypes launcher for ternary codec PTX kernels."""

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = float(threshold)
        ptx_path = Path(__file__).parent.parent / "ptx" / "codec_ops.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"PTX module not found at {ptx_path.resolve()}")
        
        module = loader.load_module_from_file(str(ptx_path))
        self._kernels = {
            "ternary_quant": loader.get_function(module, "ternary_quant_kernel"),
            "ternary_dequant": loader.get_function(module, "ternary_dequant_kernel"),
            "dct8_forward": loader.get_function(module, "dct8x8_forward_blocks"),
            "dct8_inverse": loader.get_function(module, "dct8x8_inverse_blocks"),
            "mdct_forward": loader.get_function(module, "mdct_forward_kernel"),
            "imdct_inverse": loader.get_function(module, "imdct_inverse_kernel"),
            "reshape_to_blocks_f32": loader.get_function(module, "reshape_to_blocks_f32_kernel"),
            "blocks_to_grid_f32": loader.get_function(module, "blocks_to_grid_f32_kernel"),
            "reshape_to_blocks_i32": loader.get_function(module, "reshape_to_blocks_i32_kernel"),
            "blocks_to_grid_i32": loader.get_function(module, "blocks_to_grid_i32_kernel"),
        }

    def quantize(self, values: Sequence[float], *, threshold: float | None = None) -> list[int]:
        if not values:
            return []
        thr = float(threshold) if threshold is not None else self.threshold
        n = len(values)
        FloatArray = ctypes.c_float * n
        IntArray = ctypes.c_int32 * n
        h_in = FloatArray(*values)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int32))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid = ((n + 255) // 256, 1, 1)
            loader.launch(
                self._kernels["ternary_quant"],
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                    ctypes.c_float(thr),
                ],
            )
            loader.synchronize()
            h_out = IntArray()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_int32))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dequantize(self, values: Sequence[int]) -> list[float]:
        if not values:
            return []
        n = len(values)
        IntArray = ctypes.c_int32 * n
        FloatArray = ctypes.c_float * n
        h_in = IntArray(*values)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int32))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_int32))
            block = (256, 1, 1)
            grid = ((n + 255) // 256, 1, 1)
            loader.launch(
                self._kernels["ternary_dequant"],
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                    ctypes.c_float(self.threshold),
                ],
            )
            loader.synchronize()
            h_out = FloatArray()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(ctypes.c_float))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dct8_forward(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        n_orig = len(values)
        padded_len = ((n_orig + 63) // 64) * 64
        padded_values = list(values) + [0.0] * (padded_len - n_orig)
        n_blocks = padded_len // 64
        FloatArray = ctypes.c_float * padded_len
        h_in = FloatArray(*padded_values)
        d_in = loader.gpu_malloc(padded_len * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(padded_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), padded_len * ctypes.sizeof(ctypes.c_float))
            loader.launch(
                self._kernels["dct8_forward"],
                grid=(n_blocks, 1, 1),
                block=(64, 1, 1),
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n_blocks),
                ],
            )
            loader.synchronize()
            h_out = FloatArray()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, padded_len * ctypes.sizeof(ctypes.c_float))
            return list(h_out)[:n_orig]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dct8_inverse(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        n_orig = len(values)
        padded_len = ((n_orig + 63) // 64) * 64
        padded_values = list(values) + [0.0] * (padded_len - n_orig)
        n_blocks = padded_len // 64
        FloatArray = ctypes.c_float * padded_len
        h_in = FloatArray(*padded_values)
        d_in = loader.gpu_malloc(padded_len * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(padded_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), padded_len * ctypes.sizeof(ctypes.c_float))
            loader.launch(
                self._kernels["dct8_inverse"],
                grid=(n_blocks, 1, 1),
                block=(64, 1, 1),
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n_blocks),
                ],
            )
            loader.synchronize()
            h_out = FloatArray()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, padded_len * ctypes.sizeof(ctypes.c_float))
            return list(h_out)[:n_orig]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def batch_mdct(self, values: Sequence[float], *, frame_size: int) -> list[float]:
        if not values:
            return []
        n_frames = len(values) // frame_size
        if n_frames == 0:
            return []
        out_len = n_frames * (frame_size // 2)
        FloatArrayIn = ctypes.c_float * len(values)
        FloatArrayOut = ctypes.c_float * out_len
        h_in = FloatArrayIn(*values)
        d_in = loader.gpu_malloc(len(values) * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(out_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), len(values) * ctypes.sizeof(ctypes.c_float))
            block = (min(frame_size, 256), 1, 1)
            loader.launch(
                self._kernels["mdct_forward"],
                grid=(n_frames, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(frame_size),
                    ctypes.c_int(n_frames),
                ],
            )
            loader.synchronize()
            h_out = FloatArrayOut()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, out_len * ctypes.sizeof(ctypes.c_float))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def batch_imdct(self, values: Sequence[float], *, frame_size: int) -> list[float]:
        if not values:
            return []
        half_frame = frame_size // 2
        n_frames = len(values) // half_frame
        if n_frames == 0:
            return []
        out_len = n_frames * frame_size
        FloatArrayIn = ctypes.c_float * len(values)
        FloatArrayOut = ctypes.c_float * out_len
        h_in = FloatArrayIn(*values)
        d_in = loader.gpu_malloc(len(values) * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(out_len * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), len(values) * ctypes.sizeof(ctypes.c_float))
            block = (min(half_frame, 256), 1, 1)
            loader.launch(
                self._kernels["imdct_inverse"],
                grid=(n_frames, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(frame_size),
                    ctypes.c_int(n_frames),
                ],
            )
            loader.synchronize()
            h_out = FloatArrayOut()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, out_len * ctypes.sizeof(ctypes.c_float))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def reshape_to_blocks(
        self, 
        values: Sequence[Union[float, int]], 
        *, 
        rows: int, 
        cols: int, 
        block_h: int = 8, 
        block_w: int = 8
    ) -> list:
        if not values:
            return []
        is_int = all(isinstance(v, int) for v in values)
        kernel_key = "reshape_to_blocks_i32" if is_int else "reshape_to_blocks_f32"
        elem_type = ctypes.c_int32 if is_int else ctypes.c_float
        n = len(values)
        ArrayType = elem_type * n
        h_in = ArrayType(*values)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(elem_type))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(elem_type))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), n * ctypes.sizeof(elem_type))
            block = (256, 1, 1)
            grid = ((n + 255) // 256, 1, 1)
            loader.launch(
                self._kernels[kernel_key],
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(rows),
                    ctypes.c_int(cols),
                    ctypes.c_int(block_h),
                    ctypes.c_int(block_w),
                ],
            )
            loader.synchronize()
            h_out = ArrayType()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(elem_type))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def blocks_to_grid(
        self, 
        values: Sequence[Union[float, int]], 
        *, 
        rows: int, 
        cols: int, 
        block_h: int = 8, 
        block_w: int = 8
    ) -> list:
        if not values:
            return []
        is_int = all(isinstance(v, int) for v in values)
        kernel_key = "blocks_to_grid_i32" if is_int else "blocks_to_grid_f32"
        elem_type = ctypes.c_int32 if is_int else ctypes.c_float
        n = len(values)
        ArrayType = elem_type * n
        h_in = ArrayType(*values)
        d_in = loader.gpu_malloc(n * ctypes.sizeof(elem_type))
        d_out = loader.gpu_malloc(n * ctypes.sizeof(elem_type))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), n * ctypes.sizeof(elem_type))
            block = (256, 1, 1)
            grid = ((n + 255) // 256, 1, 1)
            loader.launch(
                self._kernels[kernel_key],
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(rows),
                    ctypes.c_int(cols),
                    ctypes.c_int(block_h),
                    ctypes.c_int(block_w),
                ],
            )
            loader.synchronize()
            h_out = ArrayType()
            loader.memcpy_dtoh(ctypes.cast(h_out, ctypes.c_void_p), d_out, n * ctypes.sizeof(elem_type))
            return list(h_out)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def execution_plan(self, *, work_items: int, preferred_tier: int = 2) -> Dict[str, Any]:
        if work_items <= 0:
            return {"preferred_tier": int(preferred_tier), "work_items": 0, "fanout": 1, "batch_size": 1, "cascade": ["parallel_fanout", "worker_reduce"]}
        fanout = min(work_items, 8 if preferred_tier <= 1 else (4 if preferred_tier == 2 else 2))
        batch_size = max(1, work_items // max(1, fanout))
        return {
            "preferred_tier": int(preferred_tier),
            "work_items": int(work_items),
            "fanout": int(fanout),
            "batch_size": int(batch_size),
            "cascade": ["parallel_fanout", "worker_reduce"],
        }


__all__ = ["TernaryCodecOps"]