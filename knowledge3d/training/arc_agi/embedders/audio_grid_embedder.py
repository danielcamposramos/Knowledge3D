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
        self.hop_size = self.frame_size // 2  # 50% overlap for MDCT/IMDCT
        self.n_harmonics = int(n_harmonics)

        if codec is None:
            self.codec = SovereignTernaryAudioCodec(
                frame_size=self.frame_size, hop_size=self.hop_size, threshold=0.2
            )
        else:
            if not isinstance(codec, SovereignTernaryAudioCodec):
                raise RuntimeError("Non-sovereign audio codecs are forbidden in the sovereign path")
            self.codec = codec

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

        # Wrap waveform into ternary vector (simple sign mapping) and encode via sovereign codec only.
        if not isinstance(self.codec, SovereignTernaryAudioCodec):
            raise RuntimeError("Non-sovereign audio codec detected; CPU fallbacks are forbidden")
        ternary_samples = [0 if v == 0 else (1 if v > 0 else -1) for v in waveform]
        tvec = TernaryVector(ternary_samples)
        self.codec.encode("clip_embed", tvec)  # type: ignore[call-arg]
        decoded = self.codec.decode("clip_embed")  # type: ignore[call-arg]
        decoded_vals = decoded.to_python()

        # Project to target_dim
        return pad_or_truncate([float(v) for v in decoded_vals], target_dim, 0.0)

    def grid_to_audio_embedding_batch(
        self,
        grids: Sequence[Sequence[Sequence[int]]],
        target_dim: int = 512,
    ) -> List[List[float]]:
        """
        Batch audio-style embeddings; uses sovereign codec when available.
        Falls back to per-grid embedding for legacy codecs.
        """
        if not grids:
            return []

        if not isinstance(self.codec, SovereignTernaryAudioCodec):
            return [self.grid_to_audio_embedding(g, target_dim=target_dim) for g in grids]

        waveforms: List[List[float]] = [self._grid_to_waveform(grid) for grid in grids]

        samples: List[int] = []
        for waveform in waveforms:
            ternary_samples = [0 if v == 0 else (1 if v > 0 else -1) for v in waveform]
            samples.extend(ternary_samples)

        # One batched MDCT + quant; IMDCT not needed for embedding
        frame_size = self.frame_size
        program = f"{frame_size} BATCH_MDCT {self.codec.ops.threshold} TERNARY_QUANT"
        quantized_all = self.codec.rpn.evaluate(program, data=samples, return_vector=True)

        coeffs_per_grid = frame_size // 2
        embeddings: List[List[float]] = []
        for idx in range(len(grids)):
            start = idx * coeffs_per_grid
            end = start + coeffs_per_grid
            coeff_slice = quantized_all[start:end]
            embeddings.append(pad_or_truncate([float(v) for v in coeff_slice], target_dim, 0.0))

        return embeddings

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
