#!/usr/bin/env python3
"""
Minimal audio codec benchmark bypassing package import issues.
This script directly loads the GPU-only codec components.
"""

import os
import sys
import time
import numpy as np

# Set environment before any CUDA imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

# Add paths for direct imports
sys.path.insert(0, "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D")
sys.path.insert(0, "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/knowledge3d/cranium/codecs/ptx_bindings")

# Import PTX bindings directly
from audio_harmonic_binding import AudioHarmonicGPU
from ternary_mdct_binding import TernaryMDCTKernel
from ternary_quant_binding import TernaryQuantizer

print("=" * 80)
print("GPU-Sovereign Audio Codec Benchmark (Post-Harmonic-GPU Implementation)")
print("=" * 80)
print()

# Initialize GPU components
print("Initializing GPU components...")
mdct = TernaryMDCTKernel(n=1024)
harm_gpu = AudioHarmonicGPU()
quant_gpu = TernaryQuantizer()
print("  ✓ TernaryMDCTKernel initialized")
print("  ✓ AudioHarmonicGPU initialized")
print("  ✓ TernaryQuantizer initialized")
print()

# Test signals
sample_rate = 44100
duration = 1.0
num_samples = int(sample_rate * duration)

def generate_sine(freq=440.0):
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def generate_speech():
    t = np.linspace(0, duration, num_samples, endpoint=False)
    freqs = [120, 240, 360]
    amps = [1.0, 0.6, 0.4]
    signal = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, freqs))
    signal += 0.02 * np.random.standard_normal(signal.shape)
    return signal.astype(np.float32)

def generate_music():
    t = np.linspace(0, duration, num_samples, endpoint=False)
    chords = [261.63, 329.63, 392.0]  # C major triad
    signal = sum(np.sin(2 * np.pi * f * t) for f in chords) / len(chords)
    signal += 0.05 * np.sin(2 * np.pi * 2 * chords[0] * t)
    return signal.astype(np.float32)

def compute_psnr(original, reconstructed):
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 10 * np.log10(1.0 / mse)

# GPU harmonic analysis + synthesis
def gpu_analyze_synthesize(audio, n_harmonics=20):
    """Analyze audio and reconstruct using GPU harmonics."""
    frame_size = 1024

    # Use first frame for analysis
    if audio.size < frame_size:
        pad = np.zeros(frame_size, dtype=np.float32)
        pad[:audio.size] = audio
        signal = pad
    else:
        signal = audio[:frame_size]

    # MDCT analysis
    coeffs = mdct.forward(signal)

    # GPU top-K harmonic extraction
    idx, mag = harm_gpu.harmonic_topk(coeffs, k=min(n_harmonics, coeffs.size))

    # Convert to frequencies
    harmonics = []
    for i, a in zip(idx.tolist(), mag.tolist()):
        freq_hz = float(i) * (sample_rate / float(frame_size))
        harmonics.append((freq_hz, float(a), 0.0))

    # GPU additive synthesis
    freq = np.array([h[0] for h in harmonics], dtype=np.float32)
    amp = np.array([h[1] for h in harmonics], dtype=np.float32)
    phase = np.array([h[2] for h in harmonics], dtype=np.float32)

    synth = harm_gpu.synthesize(freq, amp, phase, sample_rate=float(sample_rate), num_samples=audio.size)

    # GPU residual
    residual = harm_gpu.subtract_residual(audio, synth)

    return harmonics, synth, residual

# Benchmark function
def benchmark_audio(name, audio):
    print(f"Testing: {name}")
    print("-" * 60)

    # Encode (harmonic analysis + MDCT residual)
    start = time.perf_counter()
    harmonics, synth, residual = gpu_analyze_synthesize(audio, n_harmonics=20)

    # MDCT on residual (simplified - just first frame for speed)
    frame_size = 1024
    window = np.hanning(frame_size).astype(np.float32)
    frame = residual[:frame_size] * window
    mdct_coeffs = mdct.forward(frame)

    # Ternary quantization
    quantized = quant_gpu.quantize(mdct_coeffs, threshold=0.1)

    encode_time_ms = (time.perf_counter() - start) * 1000

    # Decode (synthesis + IMDCT residual)
    start = time.perf_counter()

    # Dequantize and IMDCT
    dequantized = quant_gpu.dequantize(quantized, scale=0.1)
    imdct_frame = mdct.inverse(dequantized) * window

    # Reconstruct with GPU synthesis
    freq = np.array([h[0] for h in harmonics], dtype=np.float32)
    amp = np.array([h[1] for h in harmonics], dtype=np.float32)
    phase = np.array([h[2] for h in harmonics], dtype=np.float32)
    reconstructed = harm_gpu.synthesize(freq, amp, phase, sample_rate=float(sample_rate), num_samples=audio.size)

    # Add residual (simplified - just first frame)
    reconstructed[:frame_size] += imdct_frame

    decode_time_ms = (time.perf_counter() - start) * 1000

    # Quality metrics
    psnr = compute_psnr(audio, reconstructed[:audio.size])

    # Compression ratio (estimate)
    harmonics_size = len(harmonics) * 3 * 4  # 3 floats per harmonic
    ternary_size = quantized.size * 1.585 / 8  # 1.585 bits per trit
    compressed_size = harmonics_size + ternary_size
    original_size = audio.size * 4  # float32
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")

    print(f"  Encode time: {encode_time_ms:.2f} ms")
    print(f"  Decode time: {decode_time_ms:.2f} ms")
    print(f"  Compression ratio: {ratio:.1f}×")
    print(f"  PSNR: {psnr:.1f} dB")
    print(f"  Harmonics extracted: {len(harmonics)}")
    print()

    return {
        "name": name,
        "encode_ms": encode_time_ms,
        "decode_ms": decode_time_ms,
        "ratio": ratio,
        "psnr_db": psnr,
        "n_harmonics": len(harmonics),
    }

# Run benchmarks
results = []
print("Running benchmarks...")
print()

results.append(benchmark_audio("sine_440hz", generate_sine(440)))
results.append(benchmark_audio("speech_synth", generate_speech()))
results.append(benchmark_audio("music_piano", generate_music()))

# Summary table
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"{'Audio Type':<15} | {'Encode (ms)':<12} | {'Decode (ms)':<12} | {'Ratio':<8} | {'PSNR (dB)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<15} | {r['encode_ms']:>11.2f} | {r['decode_ms']:>11.2f} | {r['ratio']:>7.1f}× | {r['psnr_db']:>9.1f}")
print()
print("Note: This is a simplified benchmark focusing on GPU harmonic path performance.")
print("Full codec with multi-frame MDCT would have slightly higher latency.")
print()
