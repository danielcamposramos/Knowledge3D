"""
Ternary audio codec built on procedural synthesis and ternary MDCT residuals.

This implementation combines a procedural approximation (additive synthesis)
with ternary-quantised transform residuals for efficient compression.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .procedural_audio import ProceduralAudioSynthesizer
from .ternary_quantization import (
    dequantize_ternary,
    entropy_decode_ternary,
    entropy_encode_ternary,
    quantize_ternary,
)
from .ptx_bindings import TernaryMDCTKernel
from ..ptx_runtime.modular_rpn_engine import ModularRPNEngine

logger = logging.getLogger(__name__)


class TernaryAudioCodec:
    """
    Ternary audio codec using procedural synthesis + ternary MDCT.
    """

    def __init__(self, sample_rate: int = 44100, frame_size: int = 1024, n_harmonics: int = 20, use_gpu: bool = False):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_size <= 0 or frame_size % 2 != 0:
            raise ValueError("frame_size must be a positive even integer")
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.hop_size = self.frame_size // 2
        self.n_harmonics = int(n_harmonics)
        self.synthesizer = ProceduralAudioSynthesizer(sample_rate)
        # Precompute transform matrix for DCT-IV (self-inverse).
        n = np.arange(self.frame_size)
        self._dct_matrix = np.cos(np.pi / self.frame_size * (n + 0.5)[:, None] * (n + 0.5))
        self._dct_norm = math.sqrt(2.0 / self.frame_size)
        self._mdct_gpu: Optional[TernaryMDCTKernel] = None
        self._require_gpu = use_gpu
        if use_gpu:
            self._mdct_gpu = TernaryMDCTKernel(n=self.frame_size)
            logger.info("TernaryMDCTKernel initialised for frame_size=%d", self.frame_size)
        self.rpn = ModularRPNEngine()

    def encode(self, audio: np.ndarray) -> Dict:
        """
        Encode audio to ternary compressed format.

        Args:
            audio: Input audio (float32, mono).

        Returns:
            encoded dictionary with harmonics, ternary MDCT residuals and metadata.
        """
        samples = np.asarray(audio, dtype=np.float32).flatten()
        if samples.size == 0:
            raise ValueError("audio must not be empty")

        harmonics = self.synthesizer.analyze(samples, n_harmonics=self.n_harmonics)
        approximation = self.synthesizer.synthesize(
            harmonics, duration_sec=float(samples.size) / self.sample_rate
        )
        approximation = approximation[: samples.size]
        residual = (samples - approximation).astype(np.float32)

        window = np.hanning(self.frame_size).astype(np.float32)
        frames_q: List[np.ndarray] = []
        mdct_metadata: List[Dict] = []
        encoded_frames: List[bytes] = []

        num_frames = max(1, int(math.ceil((samples.size - self.frame_size) / self.hop_size)) + 1)
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frame = residual[start:end]
            if frame.size < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - frame.size), mode="constant")
            windowed = frame * window
            mdct_coeffs = self.mdct_frame(windowed)
            quantized, meta = quantize_ternary(mdct_coeffs, adaptive=True, threshold=0.1)
            encoded = entropy_encode_ternary(quantized)
            frames_q.append(quantized)
            mdct_metadata.append(meta)
            encoded_frames.append(encoded)

        encoded_dict: Dict = {
            "harmonics": harmonics,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "sample_rate": self.sample_rate,
            "num_samples": samples.size,
            "mdct_quantized": np.stack(frames_q, axis=0),
            "mdct_metadata": mdct_metadata,
            "encoded_frames": encoded_frames,
            "window": "hann",
        }
        return encoded_dict

    def decode(self, encoded: Dict) -> np.ndarray:
        """
        Decode ternary audio back to samples.

        Args:
            encoded: Output from encode().

        Returns:
            Reconstructed audio (float32, mono).
        """
        required_keys = ("harmonics", "frame_size", "hop_size", "sample_rate", "num_samples")
        for key in required_keys:
            if key not in encoded:
                raise ValueError(f"encoded missing required key '{key}'")

        frame_size = int(encoded["frame_size"])
        hop_size = int(encoded["hop_size"])
        num_samples = int(encoded["num_samples"])
        window = np.hanning(frame_size).astype(np.float32)

        quantized_frames = encoded.get("mdct_quantized")
        metadata = encoded.get("mdct_metadata", [])
        encoded_frames = encoded.get("encoded_frames")
        if quantized_frames is None and encoded_frames is None:
            raise ValueError("encoded must contain either 'mdct_quantized' or 'encoded_frames'")

        frames_list = []
        if encoded_frames is not None and len(encoded_frames) > 0:
            for data in encoded_frames:
                frames_list.append(entropy_decode_ternary(data))
        else:
            frames_list = [np.asarray(frame, dtype=np.int8) for frame in quantized_frames]

        residual = np.zeros(num_samples + frame_size, dtype=np.float32)
        for i, q_frame in enumerate(frames_list):
            meta = metadata[i] if i < len(metadata) else None
            deq = dequantize_ternary(q_frame, metadata=meta)
            time_frame = self.imdct_frame(deq)
            time_frame = time_frame * window
            start = i * hop_size
            end = start + frame_size
            residual[start:end] += time_frame[:frame_size]

        residual = residual[:num_samples]
        duration_sec = float(num_samples) / encoded["sample_rate"]
        procedural = self.synthesizer.synthesize(encoded["harmonics"], duration_sec)
        output = procedural[:num_samples] + residual
        return output.astype(np.float32)

    def imdct_frame(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Inverse transform matching mdct_frame (DCT-IV is self-inverse).
        Uses GPU if available, otherwise CPU.
        """
        c = np.asarray(coeffs, dtype=np.float32)
        if c.size != self.frame_size:
            raise ValueError(f"coeffs must have length {self.frame_size}")
        if self._mdct_gpu is not None:
            try:
                return self._mdct_gpu.inverse(c).astype(np.float32)
            except Exception as exc:
                if self._require_gpu:
                    raise
                logger.warning("GPU iMDCT failed, falling back to CPU: %s", exc)
        c64 = c.astype(np.float64, copy=False)
        reconstructed = self._dct_norm * np.dot(c64, self._dct_matrix)
        return reconstructed.astype(np.float32)

    def mdct_frame(self, frame: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """
        Compute MDCT using GPU if available, otherwise CPU.
        """
        x = np.asarray(frame, dtype=np.float32)
        if x.size != self.frame_size:
            raise ValueError(f"frame must have length {self.frame_size}")
        if self._mdct_gpu is not None:
            try:
                return self._mdct_gpu.forward(x).astype(np.float32)
            except Exception as exc:
                if self._require_gpu:
                    raise
                logger.warning("GPU MDCT failed, falling back to CPU: %s", exc)
        # CPU path
        x64 = x.astype(np.float64, copy=False)
        coeffs = self._dct_norm * np.dot(x64, self._dct_matrix)
        return coeffs.astype(np.float32)

    def compute_compression_ratio(self, original_size: int, encoded: Dict) -> float:
        """
        Compute achieved compression ratio.

        Args:
            original_size: Size in bytes of original PCM (e.g., len(audio) * 4).
            encoded: Encoded dictionary from `encode`.
        """
        harmonics = encoded.get("harmonics", [])
        harmonics_size = len(harmonics) * 3 * 4  # 3 floats per harmonic.

        encoded_frames = encoded.get("encoded_frames", [])
        if encoded_frames:
            mdct_size = sum(len(f) for f in encoded_frames)
        else:
            mdct_q = encoded.get("mdct_quantized")
            mdct_size = mdct_q.nbytes if mdct_q is not None else 0

        compressed_size = harmonics_size + mdct_size
        if compressed_size == 0:
            return float("inf")
        return float(original_size) / float(compressed_size)

    def encode_to_rpn(self, audio: np.ndarray) -> Dict:
        """
        Encode audio and push harmonics to RPN Stack 7, returning metadata.
        """
        harmonics = self.synthesizer.analyze(audio, n_harmonics=self.n_harmonics)
        stack_id = 7
        offsets = []
        for freq, amp, phase in harmonics:
            offsets.append(self.rpn.get_depth(stack_id=stack_id))
            self.rpn.push(float(freq), stack_id=stack_id)
            self.rpn.push(float(amp), stack_id=stack_id)
            self.rpn.push(float(phase), stack_id=stack_id)
        encoded = self.encode(audio)
        metadata = {
            "stack_id": stack_id,
            "n_harmonics": len(harmonics),
            "offsets": offsets,
            "encoded_residual": encoded,
            "duration_sec": len(audio) / self.sample_rate,
            "sample_rate": self.sample_rate,
        }
        return metadata

    def decode_from_rpn(self, metadata: Dict) -> np.ndarray:
        """
        Decode audio using harmonics from RPN Stack 7 and stored residual metadata.
        """
        stack_id = metadata.get("stack_id", 7)
        n_harmonics = int(metadata.get("n_harmonics", 0))
        harmonics = []
        for _ in range(n_harmonics):
            phase = self.rpn.pop(stack_id=stack_id)
            amp = self.rpn.pop(stack_id=stack_id)
            freq = self.rpn.pop(stack_id=stack_id)
            harmonics.insert(0, (freq, amp, phase))
        encoded = dict(metadata.get("encoded_residual", {}))
        encoded["harmonics"] = harmonics
        return self.decode(encoded)
