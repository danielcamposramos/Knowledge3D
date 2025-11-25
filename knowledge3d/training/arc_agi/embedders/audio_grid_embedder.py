from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


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

        if codec is None:
            from knowledge3d.cranium.codecs.ternary_audio_codec import (
                TernaryAudioCodec,
            )

            self.codec = TernaryAudioCodec(
                sample_rate=self.sample_rate,
                frame_size=self.frame_size,
                n_harmonics=self.n_harmonics,
                use_gpu=True,
            )
        else:
            self.codec = codec

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def grid_to_audio_embedding(
        self,
        grid: Sequence[Sequence[int]],
        target_dim: int = 512,
    ) -> np.ndarray:
        """
        Convert grid to audio-style embedding using MDCT/harmonic features.

        Returns:
            1D float32 vector of length `target_dim` (Matryoshka-friendly).
        """
        waveform = self._grid_to_waveform(grid)

        # Normalise grid colors [0,9] → audio range [-1, 1].
        if waveform.size > 0:
            waveform_norm = waveform / 9.0
            waveform_norm = (waveform_norm * 2.0) - 1.0
        else:
            waveform_norm = waveform

        encoded = self.codec.encode(waveform_norm.astype(np.float32))

        harmonics = np.asarray(encoded["harmonics"], dtype=np.float32)
        harmonics_flat = harmonics.ravel()  # (n_harmonics * 3,)

        mdct_quantized = np.asarray(encoded.get("mdct_quantized"), dtype=np.int8)
        stats = self._compute_ternary_stats(mdct_quantized)

        # Summarise MDCT frames by mean/std per frame.
        mdct_summary = []
        if mdct_quantized.ndim == 2:
            for frame in mdct_quantized:
                frame_f = frame.astype(np.float32)
                mdct_summary.append(float(np.mean(frame_f)))
                mdct_summary.append(float(np.std(frame_f)))
        mdct_summary = np.asarray(mdct_summary, dtype=np.float32)

        # Truncate MDCT summary to keep vector compact.
        max_summary = 200
        if mdct_summary.size > max_summary:
            mdct_summary = mdct_summary[:max_summary]

        features = np.concatenate(
            [
                harmonics_flat,
                np.array([stats["sparsity"], stats["entropy"]], dtype=np.float32),
                mdct_summary,
            ]
        )

        # Matryoshka projection to target_dim (truncate or zero-pad).
        if features.size >= target_dim:
            return features[:target_dim].astype(np.float32, copy=False)

        out = np.zeros(target_dim, dtype=np.float32)
        out[: features.size] = features.astype(np.float32)
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _grid_to_waveform(self, grid: Sequence[Sequence[int]]) -> np.ndarray:
        """
        Flatten grid into a 1D waveform (row-major) and pad/trim to frame_size.
        """
        if not grid or not grid[0]:
            return np.zeros(self.frame_size, dtype=np.float32)

        flat = np.asarray(grid, dtype=np.float32).ravel()
        waveform = np.zeros(self.frame_size, dtype=np.float32)
        size = min(flat.size, self.frame_size)
        waveform[:size] = flat[:size]
        return waveform

    @staticmethod
    def _compute_ternary_stats(quantized: np.ndarray) -> Dict[str, float]:
        """Compute basic sparsity/entropy of ternary MDCT coefficients."""
        q = np.asarray(quantized, dtype=np.int8)
        total = int(q.size)
        if total == 0:
            return {"sparsity": 1.0, "entropy": 0.0}

        zeros = int(np.sum(q == 0))
        p_zero = zeros / total
        p_nonzero = 1.0 - p_zero

        entropy = 0.0
        if p_zero > 0.0:
            entropy -= p_zero * float(np.log2(p_zero))
        if p_nonzero > 0.0:
            # +1 and -1 assumed equiprobable inside non-zero mass.
            entropy -= p_nonzero * float(np.log2(p_nonzero / 2.0))

        return {"sparsity": float(p_zero), "entropy": float(entropy)}

