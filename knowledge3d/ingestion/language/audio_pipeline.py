"""
Audio-language ingestion pipeline.

Extracts phonetic features from raw waveforms, builds fused embeddings, and
maps phonemes into a 3D acoustic space compatible with the Knowledge3D Galaxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .text_pipeline import _require_optional_dependency


@dataclass
class AudioLanguageIngestor:
    """
    Audio / phonetic ingestion helper.

    Parameters
    ----------
    whisper_model_name:
        Model size to load via `whisper.load_model`.
    sample_rate:
        Target sample rate when loading audio snippets.
    device:
        Torch device string for Whisper (e.g. 'cpu', 'cuda').
    """

    whisper_model_name: str = "medium"
    sample_rate: int = 16_000
    device: str = "cpu"

    _whisper_model: "whisper.Whisper | None" = field(init=False, default=None)

    def _load_whisper_model(self):
        if self._whisper_model is None:
            _require_optional_dependency(
                "whisper",
                "pip install git+https://github.com/openai/whisper.git",
            )
            import whisper  # type: ignore

            self._whisper_model = whisper.load_model(
                self.whisper_model_name, device=self.device
            )
        return self._whisper_model

    def ingest_phoneme(self, audio_path: str | Path, phoneme: str, lang: str) -> Dict:
        """
        Convert a single phoneme recording into a fused embedding + 3D position.
        """
        _require_optional_dependency("librosa", "pip install librosa")
        import librosa  # type: ignore

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        if audio.size == 0:
            raise ValueError(f"Audio file '{audio_path}' contains no samples")

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_avg = mel_spec_db.mean(axis=1).astype(np.float32)

        whisper_embedding = self._compute_whisper_embedding(audio, sr)
        fused_embedding = self._fuse_audio_features(mel_avg, whisper_embedding)
        position_3d = self._phoneme_to_3d(phoneme, audio, sr)

        return {
            "phoneme": phoneme,
            "position_3d": position_3d,
            "embedding_128": fused_embedding,
            "language": lang,
            "sample_rate": sr,
            "duration_seconds": audio.size / sr,
        }

    # ------------------------------------------------------------------ #
    # Feature extraction helpers
    # ------------------------------------------------------------------ #
    def _compute_whisper_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        model = self._load_whisper_model()

        _require_optional_dependency("torch", "pip install torch")
        import torch  # type: ignore
        import whisper  # type: ignore

        device = torch.device(self.device)
        tensor = torch.tensor(audio, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(tensor, padding=0)
            emb = model.embed_audio(mel)

        return emb.cpu().numpy().astype(np.float32).flatten()

    def _fuse_audio_features(
        self, mel: np.ndarray, whisper_embedding: np.ndarray
    ) -> np.ndarray:
        if mel.ndim != 1 or whisper_embedding.ndim != 1:
            raise ValueError("Inputs to _fuse_audio_features must be 1D")
        combined = np.concatenate([mel, whisper_embedding], axis=0)
        return self._downsample_vector(combined, target_dim=128)

    def _phoneme_to_3d(
        self, phoneme: str, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        phoneme = phoneme.strip()
        ipa_mapping = {
            "a": np.array([0.85, 0.15, 0.2], dtype=np.float32),
            "i": np.array([0.05, 0.85, 0.25], dtype=np.float32),
            "u": np.array([0.05, 0.15, 0.85], dtype=np.float32),
            "e": np.array([0.45, 0.65, 0.3], dtype=np.float32),
            "o": np.array([0.45, 0.25, 0.65], dtype=np.float32),
        }
        if phoneme in ipa_mapping:
            return ipa_mapping[phoneme]

        formants = self._extract_formants(audio, sr)
        f1_norm = np.clip(formants[0] / 1_000.0, 0.0, 1.0)
        f2_norm = np.clip(formants[1] / 3_000.0, 0.0, 1.0)
        f3_norm = np.clip(formants[2] / 4_000.0, 0.0, 1.0)
        return np.array([f1_norm, f2_norm, f3_norm], dtype=np.float32)

    @staticmethod
    def _extract_formants(audio: np.ndarray, sr: int) -> np.ndarray:
        """Estimate the first three formants using an LPC spectral envelope."""
        _require_optional_dependency("librosa", "pip install librosa")
        import librosa  # type: ignore

        if audio.ndim != 1:
            raise ValueError("Audio must be mono for formant extraction")

        pre_emphasised = librosa.effects.preemphasis(audio)
        frame = librosa.util.normalize(pre_emphasised)
        order = max(12, sr // 1000)
        lpc = librosa.lpc(frame, order=order)
        roots = np.roots(lpc)
        roots = roots[np.imag(roots) >= 0.01]

        angles = np.angle(roots)
        freqs = angles * (sr / (2 * np.pi))
        freqs = np.sort(freqs)
        freqs = freqs[freqs > 90]  # ignore sub-voice-band artefacts

        if freqs.size < 3:
            # Fallback: approximate vowel triangle
            return np.array([500.0, 1500.0, 2500.0], dtype=np.float32)

        return freqs[:3].astype(np.float32)

    @staticmethod
    def _downsample_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
        if vector.size == target_dim:
            return vector.astype(np.float32, copy=False)
        if vector.size < target_dim:
            return np.pad(vector, (0, target_dim - vector.size)).astype(
                np.float32, copy=False
            )

        bins = np.array_split(vector, target_dim)
        collapsed = np.array([segment.mean() for segment in bins], dtype=np.float32)
        if collapsed.size != target_dim:
            collapsed = np.resize(collapsed, target_dim).astype(np.float32)
        return collapsed


__all__ = ["AudioLanguageIngestor"]
