"""
Sovereign audio ingestion pipeline.

Transforms short phoneme clips into 128-dimensional embeddings using lightweight
CPU preprocessing (librosa) and sovereign GPU kernels (TemporalReasoning).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import librosa
import numpy as np

from knowledge3d.cranium.ptx_runtime.temporal_reasoning import TemporalReasoning


def _normalize(v: np.ndarray) -> np.ndarray:
    """Min-max normalise a vector to [0, 1]."""
    v_min = v.min()
    v_max = v.max()
    if v_max - v_min < 1e-6:
        return np.zeros_like(v)
    return (v - v_min) / (v_max - v_min)


@dataclass
class SovereignAudioIngestor:
    """
    Sovereign audio/phoneme ingestion helper.

    Uses librosa for lightweight CPU preprocessing and TemporalReasoning to
    derive temporal deltas which are fused into a 128-dimensional embedding.
    """

    target_sr: int = 16_000
    clip_duration: float = 1.0

    def __post_init__(self) -> None:
        self.temporal_engine = TemporalReasoning()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_phoneme(self, audio_path: str | Path, phoneme: str, lang: str) -> Dict:
        """
        Convert a phoneme waveform into a sovereign embedding and 3D position.

        Returns:
            {
                "phoneme": str,
                "position_3d": np.ndarray(3,),
                "embedding_128": np.ndarray(128,),
                "language": str,
                "formants": np.ndarray(3,),
            }
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        audio, sr = librosa.load(
            audio_path,
            sr=self.target_sr,
            mono=True,
            duration=self.clip_duration,
        )
        if audio.size == 0:
            raise ValueError(f"{audio_path} contains no samples")

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db.astype(np.float32)

        mel_avg = mel_db.mean(axis=1).astype(np.float32)  # (128,)
        # Normalise to [-1, 1] range to avoid excessively large values.
        mel_avg = 2.0 * (_normalize(mel_avg) - 0.5)

        # Temporal reasoning over (time, features)
        mel_seq = mel_db.T  # (frames, 128)
        deltas = self.temporal_engine.compute_deltas(mel_seq)
        temporal_signature = deltas.mean(axis=0).astype(np.float32)

        # Formant extraction via LPC (CPU)
        formants = self._extract_formants(audio, sr)
        position_3d = self._formants_to_position(formants)

        # Fuse features into 128-d vector
        embedding = self._compose_embedding(mel_avg, temporal_signature, formants)

        return {
            "phoneme": phoneme,
            "position_3d": position_3d,
            "embedding_128": embedding,
            "language": lang,
            "formants": formants,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compose_embedding(
        self,
        mel_avg: np.ndarray,
        temporal_signature: np.ndarray,
        formants: np.ndarray,
    ) -> np.ndarray:
        """
        Combine spectral, temporal, and formant features into a 128-d vector.
        """
        emb = mel_avg.copy()

        # Blend temporal dynamics
        emb = 0.65 * emb + 0.35 * temporal_signature

        # Inject normalised formants into first three positions
        formants_norm = np.array(
            [
                np.clip(formants[0] / 1_000.0, 0.0, 1.0),
                np.clip(formants[1] / 3_000.0, 0.0, 1.0),
                np.clip(formants[2] / 4_000.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        emb[:3] = formants_norm * 2.0 - 1.0  # map to [-1, 1]

        return emb.astype(np.float32)

    def _extract_formants(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Estimate the first three formants (F1–F3) using LPC analysis.
        """
        order = max(12, sr // 1_000)
        try:
            coeffs = librosa.lpc(audio, order=order)
        except np.linalg.LinAlgError:
            return np.array([500.0, 1_500.0, 2_500.0], dtype=np.float32)

        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = np.sort(angles * (sr / (2 * np.pi)))
        freqs = freqs[freqs > 90.0]

        defaults = np.array([500.0, 1_500.0, 2_500.0], dtype=np.float32)
        if freqs.size >= 3:
            return freqs[:3].astype(np.float32)
        elif freqs.size > 0:
            defaults[: freqs.size] = freqs.astype(np.float32)
        return defaults

    @staticmethod
    def _formants_to_position(formants: np.ndarray) -> np.ndarray:
        """Map formant frequencies to a 3D point in [0, 1]^3."""
        f1_norm = np.clip(formants[0] / 1_000.0, 0.0, 1.0)
        f2_norm = np.clip(formants[1] / 3_000.0, 0.0, 1.0)
        f3_norm = np.clip(formants[2] / 4_000.0, 0.0, 1.0)
        return np.array([f1_norm, f2_norm, f3_norm], dtype=np.float32)


__all__ = ["SovereignAudioIngestor"]
