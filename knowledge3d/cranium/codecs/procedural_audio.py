"""
Procedural audio synthesis and analysis utilities.

The synthesizer extracts harmonic parameters from signals via FFT peak picking
and can reconstruct approximations using additive synthesis. Residuals can be
computed for ternary quantisation in the codec pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class ProceduralAudioSynthesizer:
    """
    Procedural audio synthesis using an additive harmonic model.

    The model stores harmonics as triples of (frequency_hz, amplitude, phase_rad)
    and reconstructs signals by summing sinusoids.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize synthesizer.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)

    def analyze(self, audio: np.ndarray, n_harmonics: int = 20) -> List[Tuple[float, float, float]]:
        """
        Extract harmonic parameters from an audio signal using FFT peak detection.

        Args:
            audio: Input audio samples (mono, float32 or convertible), 1-D.
            n_harmonics: Number of harmonics to extract.

        Returns:
            List of (frequency_hz, amplitude, phase_rad) tuples sorted by amplitude.
        """
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive")

        signal = np.asarray(audio, dtype=np.float32).flatten()
        if signal.size == 0:
            raise ValueError("audio must not be empty")

        spectrum = np.fft.rfft(signal)
        magnitudes = 2.0 * np.abs(spectrum) / max(signal.size, 1)
        phases = np.angle(spectrum)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / self.sample_rate)

        start_idx = 1 if magnitudes.size > 1 else 0
        indices = np.argsort(magnitudes[start_idx:])[::-1][:n_harmonics] + start_idx

        harmonics: List[Tuple[float, float, float]] = []
        for idx in indices:
            amp = float(magnitudes[idx])
            freq = float(freqs[idx])
            phase = float(phases[idx])
            harmonics.append((freq, amp, phase))

        harmonics.sort(key=lambda x: x[1], reverse=True)
        return harmonics

    def synthesize(self, harmonics: List[Tuple[float, float, float]], duration_sec: float) -> np.ndarray:
        """
        Generate audio from harmonic parameters.

        Args:
            harmonics: List of (frequency_hz, amplitude, phase_rad).
            duration_sec: Duration of the output in seconds.

        Returns:
            Synthesized mono signal as float32 array.
        """
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        num_samples = int(np.round(duration_sec * self.sample_rate))
        if num_samples <= 0:
            raise ValueError("Computed number of samples is zero; increase duration or sample_rate")
        if len(harmonics) == 0:
            return np.zeros(num_samples, dtype=np.float32)

        t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
        harmonics_arr = np.asarray(harmonics, dtype=np.float32)
        freqs = harmonics_arr[:, 0][:, None]
        amps = harmonics_arr[:, 1][:, None]
        phases = harmonics_arr[:, 2][:, None]

        # Vectorised additive synthesis.
        waveform = amps * np.cos(2.0 * np.pi * freqs * t + phases)
        synthesized = np.sum(waveform, axis=0).astype(np.float32)
        return synthesized

    def compute_residual(self, original: np.ndarray, harmonics: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute residual: original - procedural_synthesis(harmonics).

        Args:
            original: Original audio samples.
            harmonics: Harmonic parameters extracted from the signal.

        Returns:
            Residual signal aligned to original length.
        """
        original_arr = np.asarray(original, dtype=np.float32).flatten()
        if original_arr.size == 0:
            raise ValueError("original must not be empty")
        duration_sec = float(original_arr.size) / float(self.sample_rate)
        approximation = self.synthesize(harmonics, duration_sec)
        approximation = approximation[: original_arr.size]
        return (original_arr - approximation).astype(np.float32)

    def adaptive_dimension(self, harmonics: List[Tuple[float, float, float]]) -> int:
        """
        Choose Matryoshka dimension based on harmonic complexity.

        Returns:
            Selected embedding dimension.
        """
        count = len(harmonics)
        if count < 5:
            return 64
        if count < 15:
            return 256
        if count < 30:
            return 1024
        return 2048
