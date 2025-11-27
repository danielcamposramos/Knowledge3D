from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from knowledge3d.cranium.codecs import SovereignTernaryAudioCodec
from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.training.arc_agi.sovereign_utils import (
    flatten,
    mean,
    pad_or_truncate,
    std,
    zeros1d,
)


class AudioGridEmbedder:
    """
    Treat ARC grids as 1D waveforms and extract MDCT/harmonic features
    using the ternary audio codec.

    Lives on the ingestion side only. Tests can inject a lightweight codec
    to avoid GPU requirements.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 1024,
        n_harmonics: int = 20,
        codec: Any | None = None,
    ):
        """
        Args:
            sample_rate: Nominal sample rate (arbitrary for grids).
            frame_size: MDCT frame size (must be even, <=1024).
            n_harmonics: Number of harmonics to extract in codec.
            codec: Optional codec instance with an `encode(audio)` method.
                If omitted, we lazily construct `TernaryAudioCodec`.
        """
        if frame_size <= 0 or frame_size % 2 != 0 or frame_size > 1024:
            raise ValueError("frame_size must be a positive even integer <=1024")

        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.n_harmonics = int(n_harmonics)

        self.codec = codec or SovereignTernaryAudioCodec(
            frame_size=self.frame_size, hop_size=self.hop_size, threshold=0.2
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def grid_to_audio_embedding(
        self,
        grid: Sequence[Sequence[int]],
        target_dim: int = 512,
    ) -> List[float]:
        """
        Convert grid to audio-style embedding using MDCT/harmonic features.

        Returns:
            1D float32 vector of length `target_dim` (Matryoshka-friendly).
        """
        waveform = self._grid_to_waveform(grid)

        # Normalise grid colors [0,9] → audio range [-1, 1].
        waveform_norm = [(v / 9.0) * 2.0 - 1.0 for v in waveform] if waveform else []

        # Wrap waveform into ternary vector (simple sign mapping) and encode via sovereign codec
        ternary_samples = [0 if v == 0 else (1 if v > 0 else -1) for v in waveform]
        tvec = TernaryVector(ternary_samples)
        if not isinstance(self.codec, SovereignTernaryAudioCodec):
            import numpy as np  # local legacy path

            # Legacy codec path (tests inject fakes)
            encoded = self.codec.encode(np.array(waveform_norm, dtype=np.float32))  # type: ignore[arg-type]
            harmonics_raw = encoded.get("harmonics", [])
            mdct_quantized_raw = encoded.get("mdct_quantized")
            mdct_quantized = mdct_quantized_raw if mdct_quantized_raw is not None else []
            stats = self._compute_ternary_stats(mdct_quantized)
            features: List[float] = []
            if hasattr(harmonics_raw, "ravel"):
                features.extend([float(v) for v in harmonics_raw.ravel().tolist()])  # type: ignore[arg-type]
            else:
                features.extend([float(v) for v in flatten([harmonics_raw])])
            features.extend([float(stats["sparsity"]), float(stats["entropy"])])
            return pad_or_truncate(features, target_dim, 0.0)
        else:
            meta = self.codec.encode("clip_embed", tvec)  # type: ignore[call-arg]
            _ = meta  # unused
            decoded = self.codec.decode("clip_embed")  # type: ignore[call-arg]
            decoded_vals = decoded.to_python()

        # Project to target_dim
        return pad_or_truncate([float(v) for v in decoded_vals], target_dim, 0.0)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _grid_to_waveform(self, grid: Sequence[Sequence[int]]) -> List[float]:
        """
        Flatten grid into a 1D waveform (row-major) and pad/trim to frame_size.
        """
        if not grid or not grid[0]:
            return zeros1d(self.frame_size, 0.0)

        flat = [float(v) for v in flatten(grid)]
        waveform = zeros1d(self.frame_size, 0.0)
        size = min(len(flat), self.frame_size)
        for i in range(size):
            waveform[i] = flat[i]
        return waveform

    def _compute_ternary_stats(self, quantized: Sequence[Sequence[int]] | Sequence[int]) -> Dict[str, float]:
        """Placeholder stats (no numpy)."""
        if hasattr(quantized, "ravel"):
            q = [int(v) for v in quantized.ravel().tolist()]  # type: ignore[arg-type]
        elif quantized and isinstance(quantized[0], (list, tuple)):  # type: ignore[index]
            q = [int(v) for v in flatten(quantized)]  # type: ignore[arg-type]
        else:
            q = [int(v) for v in quantized]  # type: ignore[arg-type]
        total = len(q)
        if total == 0:
            return {"sparsity": 1.0, "entropy": 0.0}
        zeros = sum(1 for v in q if v == 0)
        p_zero = zeros / total
        p_nonzero = 1.0 - p_zero
        entropy = 0.0
        if p_zero > 0:
            entropy -= p_zero * math.log2(p_zero)
        if p_nonzero > 0:
            entropy -= p_nonzero * math.log2(p_nonzero / 2.0)
        return {"sparsity": float(p_zero), "entropy": float(entropy)}
