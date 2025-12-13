"""
Sovereign ternary image codec (VectorDotMap-style field coefficients).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Tuple, List

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
        values = image_rgb.values.to_python()
        for channel in range(3):
            chan = self._extract_channel(values, image_rgb.shape, channel, padded_h, padded_w)
            blocks = self._to_blocks(chan, padded_w, padded_h)
            dct_coeffs = self.ops.dct8_forward(blocks)
            field_coeffs = self._field_fit_gpu(dct_coeffs, self.n_coefficients // 3)
            coeffs_all.extend(field_coeffs)

        quantized = self.ops.quantize(coeffs_all)
        residual_vec = TernaryVector(quantized)
        seed_rpn = f"VECTORDOTMAP {self.n_coefficients} {h} {w}"
        self.galaxy.store_frame(image_id, seed_rpn, residual_vec)
        return {
            "image_id": image_id,
            "original_size": (h, w),
            "n_coefficients": self.n_coefficients,
            "compression_ratio": (h * w * 3) / max(1, self.n_coefficients),
            "stored_in_galaxy": True,
        }

    def decode(self, image_id: str, target_width: int | None = None, target_height: int | None = None) -> TernaryTensor:
        seed_rpn, residual = self.galaxy.load_frame(image_id)
        parts = seed_rpn.split()
        if len(parts) >= 4 and parts[0] == "VECTORDOTMAP":
            orig_h, orig_w = int(parts[2]), int(parts[3])
        else:
            orig_h, orig_w = 256, 256
        h = target_height or orig_h
        w = target_width or orig_w

        coeffs = self.ops.dequantize([int(v) for v in residual.to_python()])
        per_channel = self.n_coefficients // 3
        channel_size = ((h + 7) // 8) * ((w + 7) // 8) * 64

        channels: List[List[float]] = []
        for c in range(3):
            field_coeffs = coeffs[c * per_channel : (c + 1) * per_channel]
            dct_coeffs = self._field_expand_gpu(field_coeffs, channel_size)
            blocks = self.ops.dct8_inverse(dct_coeffs)
            channel = self._from_blocks(blocks, w, h)
            channels.append(channel)

        combined: List[int] = []
        for idx in range(h * w):
            r = int(channels[0][idx]) if idx < len(channels[0]) else 0
            g = int(channels[1][idx]) if idx < len(channels[1]) else 0
            b = int(channels[2][idx]) if idx < len(channels[2]) else 0
            combined.extend([r, g, b])

        ternary_rgb = [0 if v < 85 else (1 if v > 170 else -1) for v in combined]
        return TernaryTensor((h, w, 3), TernaryVector(ternary_rgb))

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
    def _extract_channel(self, values: List[int], shape: Tuple[int, int, int], channel: int, padded_h: int, padded_w: int) -> List[float]:
        orig_h, orig_w, _ = shape
        out: List[float] = []
        for y in range(padded_h):
            for x in range(padded_w):
                oy = min(y, orig_h - 1)
                ox = min(x, orig_w - 1)
                idx = (oy * orig_w + ox) * 3 + channel
                out.append(float(values[idx]) if idx < len(values) else 0.0)
        return out

    def _to_blocks(self, channel: List[float], w: int, h: int) -> List[float]:
        blocks: List[float] = []
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                for y in range(8):
                    for x in range(8):
                        idx = (by + y) * w + (bx + x)
                        blocks.append(channel[idx] if idx < len(channel) else 0.0)
        return blocks

    def _from_blocks(self, blocks: List[float], w: int, h: int) -> List[float]:
        result = [0.0] * (w * h)
        block_idx = 0
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                for y in range(8):
                    for x in range(8):
                        dst = (by + y) * w + (bx + x)
                        src_idx = block_idx * 64 + y * 8 + x
                        if dst < len(result) and src_idx < len(blocks):
                            result[dst] = blocks[src_idx]
                block_idx += 1
        return result


__all__ = ["SovereignTernaryImageCodec"]
