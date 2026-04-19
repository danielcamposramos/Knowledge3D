"""
GPU-only procedural audio analysis and synthesis using sovereign PTX bindings.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .ptx_bindings import AudioHarmonicGPU, TernaryMDCTKernel


class ProceduralAudioSynthesizer:
    """
    Procedural audio synthesis using GPU additive harmonic model and MDCT-based analysis.
    All math runs on GPU via PTX bindings; no CPU fallbacks.
    """

    def __init__(self, sample_rate: int = 44100, frame_size: int = 1024):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_size <= 0 or frame_size % 2 != 0 or frame_size > 1024:
            raise ValueError("frame_size must be a positive even integer <=1024")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.mdct = TernaryMDCTKernel(n=self.frame_size)
        self.harm_gpu = AudioHarmonicGPU()

    def analyze(self, audio: np.ndarray, n_harmonics: int = 20) -> List[Tuple[float, float, float]]:
        """
        Extract harmonic parameters from an audio signal using GPU MDCT + top-K magnitude bins.
        Phases are set to 0 (MDCT is real-valued).
        """
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive")
        signal = np.asarray(audio, dtype=np.float32).flatten()
        if signal.size == 0:
            raise ValueError("audio must not be empty")
        # Use first frame_size samples (or pad) for MDCT-based analysis.
        if signal.size < self.frame_size:
            pad = np.zeros(self.frame_size, dtype=np.float32)
            pad[: signal.size] = signal
            signal = pad
        else:
            signal = signal[: self.frame_size]

        coeffs = self.mdct.forward(signal.astype(np.float32))
        idx, mag = self.harm_gpu.harmonic_topk(coeffs, k=min(n_harmonics, coeffs.size))
        harmonics: List[Tuple[float, float, float]] = []
        for i, a in zip(idx.tolist(), mag.tolist()):
            freq_hz = float(i) * (self.sample_rate / float(self.frame_size))
            harmonics.append((freq_hz, float(a), 0.0))
        return harmonics

    def synthesize(self, harmonics: List[Tuple[float, float, float]], duration_sec: float) -> np.ndarray:
        """
        Generate audio from harmonic parameters using GPU additive synthesis.
        """
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if len(harmonics) == 0:
            return np.zeros(int(np.round(duration_sec * self.sample_rate)), dtype=np.float32)
        num_samples = int(np.round(duration_sec * self.sample_rate))
        freq = np.asarray([h[0] for h in harmonics], dtype=np.float32)
        amp = np.asarray([h[1] for h in harmonics], dtype=np.float32)
        phase = np.asarray([h[2] for h in harmonics], dtype=np.float32)
        return self.harm_gpu.synthesize(freq, amp, phase, sample_rate=float(self.sample_rate), num_samples=num_samples)

    def compute_residual(self, original: np.ndarray, harmonics: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Compute residual on GPU: original - synth(harmonics, duration).
        """
        orig = np.asarray(original, dtype=np.float32).flatten()
        duration_sec = float(orig.size) / float(self.sample_rate)
        approx = self.synthesize(harmonics, duration_sec)
        if approx.size != orig.size:
            approx = approx[: orig.size]
        return self.harm_gpu.subtract_residual(orig, approx)

    def adaptive_dimension(self, harmonics: List[Tuple[float, float, float]]) -> int:
        """
        Choose Matryoshka dimension based on harmonic count.
        """
        count = len(harmonics)
        if count < 5:
            return 64
        if count < 15:
            return 256
        if count < 30:
            return 1024
        return 2048
