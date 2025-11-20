#!/usr/bin/env python3
"""
Benchmark ternary audio codec with synthetic signals.

Usage:
    python scripts/benchmark_ternary_audio.py
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec
from knowledge3d.cranium.codecs.ternary_quantization import entropy_encode_ternary


def generate_sine_wave(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def load_or_generate_speech(duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    # Lightweight speech-like synthetic sample (formant-inspired).
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freqs = [120, 240, 360]
    amps = [1.0, 0.6, 0.4]
    signal = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, freqs))
    # Add mild noise to mimic articulation.
    signal += 0.02 * np.random.standard_normal(signal.shape)
    return signal.astype(np.float32)


def load_or_generate_music(duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    chords = [261.63, 329.63, 392.0]  # C major triad.
    signal = sum(np.sin(2 * np.pi * f * t) for f in chords) / len(chords)
    signal += 0.05 * np.sin(2 * np.pi * 2 * chords[0] * t)
    return signal.astype(np.float32)


def compute_encoded_size(encoded: Dict) -> int:
    harmonics = encoded.get("harmonics", [])
    harmonics_size = len(harmonics) * 3 * 4
    frames = encoded.get("mdct_quantized")
    if frames is not None:
        rle_bytes = [entropy_encode_ternary(f) for f in frames]
        mdct_size = sum(len(b) for b in rle_bytes)
    else:
        mdct_size = 0
    return harmonics_size + mdct_size


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def print_benchmark_table(results: List[Dict]) -> None:
    headers = [
        "Audio Type",
        "Size (KB)",
        "Compressed (KB)",
        "Ratio",
        "Encode (ms)",
        "Decode (ms)",
        "PSNR (dB)",
    ]
    row_fmt = "{:<13} | {:>9} | {:>15} | {:>5} | {:>11} | {:>11} | {:>8}"
    print("Ternary Audio Codec Benchmark Results")
    print("=" * 75)
    print(row_fmt.format(*headers))
    print("-" * 75)
    for r in results:
        print(
            row_fmt.format(
                r["name"],
                f"{r['original_kb']:.1f}",
                f"{r['compressed_kb']:.1f}",
                f"{r['ratio']:.1f}",
                f"{r['encode_ms']:.1f}",
                f"{r['decode_ms']:.1f}",
                f"{r['psnr_db']:.1f}",
            )
        )


def validate_targets(results: List[Dict]) -> None:
    for r in results:
        if r["name"].startswith("speech") and (r["ratio"] < 5 or r["psnr_db"] < 20):
            raise AssertionError("Speech benchmark did not meet quality/ratio targets")


def benchmark_audio_codec(use_gpu: bool = False) -> List[Dict]:
    codec = TernaryAudioCodec(sample_rate=44100, use_gpu=use_gpu)
    test_cases: List[Tuple[str, np.ndarray]] = [
        ("sine_440hz", generate_sine_wave(440, duration=1.0)),
        ("speech_synth", load_or_generate_speech(duration=1.0)),
        ("music_piano", load_or_generate_music(duration=1.0)),
    ]

    results: List[Dict] = []
    for name, audio in test_cases:
        start = time.perf_counter()
        encoded = codec.encode(audio)
        encode_time_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        decoded = codec.decode(encoded)
        decode_time_ms = (time.perf_counter() - start) * 1000

        original_size = len(audio) * 4
        compressed_size = compute_encoded_size(encoded)
        ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
        psnr = compute_psnr(audio, decoded[: len(audio)])

        results.append(
            {
                "name": name,
                "original_kb": original_size / 1024,
                "compressed_kb": compressed_size / 1024,
                "ratio": ratio,
                "encode_ms": encode_time_ms,
                "decode_ms": decode_time_ms,
                "psnr_db": psnr,
            }
        )
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Enable GPU MDCT path if available")
    args = ap.parse_args()

    results = benchmark_audio_codec(use_gpu=args.gpu)
    print_benchmark_table(results)
    validate_targets(results)
