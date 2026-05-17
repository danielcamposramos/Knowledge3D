"""
Sovereign ternary image codec (VectorDotMap-style field coefficients).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.ternary import TernaryTensor, TernaryVector, TernaryGalaxy


class SovereignTernaryImageCodec:
    def __init__(self, n_coefficients: int = 512, threshold: float = 0.15) -> None:
        self.n_coefficients = int(n_coefficients)
        self.ops = TernaryCodecOps(threshold=threshold)
        self.galaxy = TernaryGalaxy()
        ptx_path = Path(__file__).parent.parent / "ptx" / "vectordotmap_encoder.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"vectordotmap_encoder.ptx not found at {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        self._fit_kernel = loader.get_function(module, "field_coefficient_fit")
        self._expand_kernel = loader.get_function(module, "field_coefficient_expand")
        # Importance weights (match kernel constant)
        self._importance = [
            1.00, 0.98, 0.95, 0.90, 0.83, 0.75, 0.65, 0.55,
            0.98, 0.95, 0.90, 0.85, 0.78, 0.70, 0.60, 0.50,
            0.95, 0.90, 0.85, 0.80, 0.73, 0.65, 0.55, 0.45,
            0.90, 0.85, 0.80, 0.75, 0.68, 0.60, 0.50, 0.40,
            0.83, 0.78, 0.73, 0.68, 0.63, 0.55, 0.45, 0.35,
            0.75, 0.70, 0.65, 0.60, 0.55, 0.48, 0.40, 0.30,
            0.65, 0.60, 0.55, 0.50, 0.45, 0.38, 0.30, 0.22,
            0.55, 0.50, 0.45, 0.40, 0.35, 0.28, 0.20, 0.15,
        ]

    def encode(self, image_id: str, image_rgb: TernaryTensor) -> Dict:
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image tensor must have shape (H, W, 3)")

        h, w, _ = image_rgb.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded_h = h + pad_h
        padded_w = w + pad_w

        coeffs_all: List[float] = []
        rgb = self._reshape_rgb(image_rgb.values.to_python(), width=w, height=h)
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        signal_plan = self.ops.execution_plan(
            work_items=3 * ((padded_h * padded_w) // 64),
            preferred_tier=2,
        )
        per_channel = self.n_coefficients // 3
        for channel in range(3):
            chan = rgb[:, :, channel].astype(np.float32, copy=False).reshape(-1).tolist()
            blocks = self.ops.reshape_to_blocks(chan, rows=padded_h, cols=padded_w, block_h=8, block_w=8)
            dct_coeffs = self.ops.dct8_forward(blocks)
            field_coeffs = self._field_fit_gpu(dct_coeffs, per_channel)
            coeffs_all.extend(field_coeffs)

        quantized = self.ops.quantize(coeffs_all)
        residual_vec = TernaryVector(quantized)
        seed_rpn = f"VECTORDOTMAP {self.n_coefficients} {h} {w}"
        self.galaxy.store_frame(
            image_id,
            seed_rpn,
            residual_vec,
            metadata={
                "original_height": h,
                "original_width": w,
                "padded_height": padded_h,
                "padded_width": padded_w,
                "channels": 3,
                "coefficients_per_channel": per_channel,
                "blocks_per_channel": (padded_h * padded_w) // 64,
                "math_core_plan": signal_plan,
            },
        )
        return {
            "image_id": image_id,
            "seed_rpn": seed_rpn,
            "original_size": (h, w),
            "padded_size": (padded_h, padded_w),
            "n_coefficients": self.n_coefficients,
            "compression_ratio": (h * w * 3) / max(1, self.n_coefficients),
            "stored_in_galaxy": True,
            "math_core_plan": signal_plan,
        }

    def decode(self, image_id: str, target_width: int | None = None, target_height: int | None = None) -> TernaryTensor:
        seed_rpn, residual, metadata = self.galaxy.load_frame_details(image_id)
        parts = seed_rpn.split()
        if len(parts) >= 4 and parts[0] == "VECTORDOTMAP":
            orig_h, orig_w = int(parts[2]), int(parts[3])
        else:
            orig_h, orig_w = 256, 256
        h = min(int(target_height or orig_h), orig_h)
        w = min(int(target_width or orig_w), orig_w)
        padded_h = int(metadata.get("padded_height", ((orig_h + 7) // 8) * 8))
        padded_w = int(metadata.get("padded_width", ((orig_w + 7) // 8) * 8))

        coeffs = self.ops.dequantize([int(v) for v in residual.to_python()])
        per_channel = int(metadata.get("coefficients_per_channel", self.n_coefficients // 3))
        channel_size = int(metadata.get("blocks_per_channel", (padded_h // 8) * (padded_w // 8))) * 64

        _decode_plan = self.ops.execution_plan(
            work_items=int(metadata.get("channels", 3)) * int(metadata.get("blocks_per_channel", channel_size // 64)),
            preferred_tier=2,
        )
        channels: List[np.ndarray] = []
        for c in range(3):
            field_coeffs = coeffs[c * per_channel : (c + 1) * per_channel]
            dct_coeffs = self._field_expand_gpu(field_coeffs, channel_size)
            blocks = self.ops.dct8_inverse(dct_coeffs)
            padded_channel = self.ops.blocks_to_grid(
                blocks,
                rows=padded_h,
                cols=padded_w,
                block_h=8,
                block_w=8,
            )
            channels.append(self._crop_channel_array(padded_channel, padded_w, padded_h, w, h))

        rgb = np.stack(channels, axis=2)
        rgb = np.clip(rgb, 0, 255).astype(np.int32, copy=False)
        ternary_rgb = np.where(rgb < 85, 0, np.where(rgb > 170, 1, -1)).astype(np.int32, copy=False)
        return TernaryTensor((h, w, 3), TernaryVector(ternary_rgb.reshape(-1).tolist()))

    # ------------------------------------------------------------------ #
    # GPU helpers
    # ------------------------------------------------------------------ #
    def _field_fit_gpu(self, dct_coeffs: List[float], n_out: int) -> List[float]:
        num_blocks = len(dct_coeffs) // 64
        in_buf = (ctypes.c_float * (num_blocks * 64))(*dct_coeffs)
        out_buf = (ctypes.c_float * n_out)()
        importance = (ctypes.c_float * 64)(*self._importance)
        d_in = loader.gpu_malloc(ctypes.sizeof(in_buf))
        d_out = loader.gpu_malloc(ctypes.sizeof(out_buf))
        d_imp = loader.gpu_malloc(ctypes.sizeof(importance))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), ctypes.sizeof(in_buf))
            loader.memcpy_htod(d_imp, ctypes.cast(importance, ctypes.c_void_p), ctypes.sizeof(importance))
            block = (256, 1, 1)
            grid_x = (n_out + block[0] - 1) // block[0]
            loader.launch(
                self._fit_kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_uint64(d_imp.value),
                    ctypes.c_int(num_blocks),
                    ctypes.c_int(n_out),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out_buf, ctypes.c_void_p), d_out, ctypes.sizeof(out_buf))
            return [float(v) for v in out_buf]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)
            loader.gpu_free(d_imp)

    def _field_expand_gpu(self, field_coeffs: List[float], total_dct: int) -> List[float]:
        num_blocks = total_dct // 64
        n_coeffs = len(field_coeffs)
        in_buf = (ctypes.c_float * n_coeffs)(*field_coeffs)
        out_buf = (ctypes.c_float * total_dct)()
        importance = (ctypes.c_float * 64)(*self._importance)
        d_in = loader.gpu_malloc(ctypes.sizeof(in_buf))
        d_out = loader.gpu_malloc(ctypes.sizeof(out_buf))
        d_imp = loader.gpu_malloc(ctypes.sizeof(importance))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), ctypes.sizeof(in_buf))
            loader.memcpy_htod(d_imp, ctypes.cast(importance, ctypes.c_void_p), ctypes.sizeof(importance))
            block = (64, 1, 1)
            grid = (num_blocks, 1, 1)
            loader.launch(
                self._expand_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_uint64(d_imp.value),
                    ctypes.c_int(num_blocks),
                    ctypes.c_int(n_coeffs),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out_buf, ctypes.c_void_p), d_out, ctypes.sizeof(out_buf))
            return [float(v) for v in out_buf]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)
            loader.gpu_free(d_imp)

    # ------------------------------------------------------------------ #
    # Block helpers
    # ------------------------------------------------------------------ #
    def _reshape_rgb(self, rgb_flat: List[int], *, width: int, height: int) -> np.ndarray:
        arr = np.asarray(rgb_flat, dtype=np.float32)
        expected = width * height * 3
        if arr.size != expected:
            raise ValueError(f"expected {expected} RGB values, got {arr.size}")
        return arr.reshape(height, width, 3)

    def _crop_channel_array(
        self,
        padded_channel: List[float],
        padded_w: int,
        padded_h: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        arr = np.asarray(padded_channel, dtype=np.float32)
        expected = padded_h * padded_w
        if arr.size != expected:
            raise ValueError(f"expected padded channel size {expected}, got {arr.size}")
        return arr.reshape(padded_h, padded_w)[: min(target_h, padded_h), : min(target_w, padded_w)]

__all__ = ["SovereignTernaryImageCodec"]
