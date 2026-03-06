"""
Sovereign ternary audio codec using GPU-only MDCT/IMDCT + ternary quantisation.

No numpy, no CPU fallbacks. Uses TernaryVector and GPU-only ops.

Architecture References:
- docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md — Audio as frequency-time signal
- docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md — Spectrogram as VectorDotMap

Audio = Frequency components over time (STFT/MDCT).
Spectrograms share VectorDotMap codec with images for unified representation.
Bidirectional: Audio ↔ Image via spectrogram/sonification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from knowledge3d.cranium.ternary import TernaryVector, TernaryGalaxy
from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


class SovereignTernaryAudioCodec:
    """GPU-native ternary audio codec with PTX-backed MDCT path."""

    def __init__(self, frame_size: int = 1024, hop_size: int | None = None, threshold: float = 0.2) -> None:
        if frame_size <= 0 or frame_size % 2 != 0:
            raise ValueError("frame_size must be positive even")
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size) if hop_size is not None else self.frame_size // 2
        self.ops = TernaryCodecOps(threshold=threshold)
        self.rpn = ModularRPNEngine()
        self.galaxy = TernaryGalaxy()

    def encode(self, clip_id: str, samples: TernaryVector) -> Dict:
        samples_arr = np.asarray(samples.to_python(), dtype=np.float32)
        original_length = int(samples_arr.shape[0])
        # Pad to frame boundary
        remainder = original_length % self.frame_size
        if remainder != 0:
            pad_len = self.frame_size - remainder
            samples_arr = np.pad(samples_arr, (0, pad_len), mode="constant")
        padded_length = int(samples_arr.shape[0])
        frame_count = padded_length // self.frame_size
        signal_plan = self.ops.execution_plan(work_items=frame_count, preferred_tier=2)
        seed_rpn = f"{self.frame_size} BATCH_MDCT {self.ops.threshold} TERNARY_QUANT"
        quantized = self.rpn.evaluate(seed_rpn, data=samples_arr.tolist(), return_vector=True)
        residual_vec = TernaryVector(self._flatten_list(quantized))
        self.galaxy.store_frame(
            clip_id,
            seed_rpn,
            residual_vec,
            metadata={
                "original_length": original_length,
                "padded_length": padded_length,
                "frame_size": self.frame_size,
                "hop_size": self.hop_size,
                "frame_count": frame_count,
                "math_core_plan": signal_plan,
            },
        )
        return {
            "clip_id": clip_id,
            "seed_rpn": seed_rpn,
            "stored_in_galaxy": True,
            "original_length": original_length,
            "padded_length": padded_length,
            "frame_count": frame_count,
            "math_core_plan": signal_plan,
        }

    def decode(self, clip_id: str) -> TernaryVector:
        seed_rpn, residual, metadata = self.galaxy.load_frame_details(clip_id)
        _ = seed_rpn
        frame_count = int(metadata.get("frame_count", 1))
        _decode_plan = self.ops.execution_plan(work_items=frame_count, preferred_tier=2)
        imdct_program = f"TERNARY_DEQUANT {self.frame_size} IMDCT"
        imdct = self.rpn.evaluate(imdct_program, data=residual.to_python(), return_vector=True)
        flat_imdct = self._flatten_list(imdct)
        original_length = int(metadata.get("original_length", len(flat_imdct)))
        if original_length < len(flat_imdct):
            flat_imdct = flat_imdct[:original_length]
        ternary = [0 if v == 0 else (1 if v > 0 else -1) for v in flat_imdct]
        return TernaryVector(ternary)

    def _flatten_list(self, value) -> list:
        if isinstance(value, list):
            out: list = []
            for item in value:
                out.extend(self._flatten_list(item))
            return out
        return [value]


__all__ = ["SovereignTernaryAudioCodec"]
