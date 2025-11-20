#!/usr/bin/env python3
"""
Validate basic quality metrics for ternary codecs.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec
from knowledge3d.cranium.codecs.procedural_video import ProceduralVideoGenerator


def validate_audio():
    codec = TernaryAudioCodec(sample_rate=8000)
    t = np.linspace(0, 1, 8000, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    encoded = codec.encode(audio)
    decoded = codec.decode(encoded)
    mse = float(np.mean((audio - decoded[: len(audio)]) ** 2))
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    ratio = codec.compute_compression_ratio(len(audio) * 4, encoded)
    print(f"[audio] PSNR={psnr:.2f} dB, ratio={ratio:.1f}x")


def validate_video():
    codec = TernaryVideoCodec(width=64, height=64)
    generator = ProceduralVideoGenerator(width=64, height=64)
    seed = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    frame = generator.generate_frame(seed, time_param=0.1)
    encoded = codec.encode(frame, seed=seed)
    decoded = codec.decode(encoded)
    mse = float(np.mean((frame.astype(np.float32) - decoded.astype(np.float32)) ** 2))
    psnr = 10 * np.log10((255.0**2) / (mse + 1e-12))
    ratio = codec.compute_compression_ratio(frame.size, encoded)
    print(f"[video] PSNR={psnr:.2f} dB, ratio={ratio:.1f}x")


if __name__ == "__main__":
    validate_audio()
    validate_video()
