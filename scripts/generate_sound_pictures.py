#!/usr/bin/env python3
"""
Sound Picture Generator
=======================

Generates spectrograms (sound pictures) from audio files for multi-modal training.

This creates visual representations of audio that complement temporal embeddings:
- Temporal: 1D audio waveform (sequential)
- Spatial: 2D mel spectrogram (frequency × time)
- Combined: Multi-modal understanding of audio patterns

Usage:
    python scripts/generate_sound_pictures.py \
        --audio-dir /K3D/Knowledge3D.local/datasets/speech/audio \
        --output-dir /K3D/Knowledge3D.local/datasets/speech/spectrograms \
        --n-mels 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not installed. Install with: pip install librosa")


def generate_spectrogram(
    audio_path: Path,
    output_path: Path,
    n_mels: int = 128,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: int = 8000,
) -> np.ndarray:
    """
    Generate mel spectrogram image from audio file.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        output_path: Path to save spectrogram image (.png)
        n_mels: Number of mel frequency bins (default: 128, matches embedding dim!)
        sr: Target sample rate (default: 22050 Hz)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop size between frames (default: 512 samples ~23ms)
        fmax: Maximum frequency (default: 8000 Hz, covers speech)

    Returns:
        Mel spectrogram as 2D numpy array (n_mels × n_frames)
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa is required. Install with: pip install librosa")

    # Load audio
    y, sr_actual = librosa.load(audio_path, sr=sr)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr_actual,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )

    # Convert to dB scale (log magnitude)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 255] for image
    # Typical range: [-80, 0] dB
    mel_norm = ((mel_spec_db + 80) / 80 * 255).astype(np.uint8)

    # Clip to valid range
    mel_norm = np.clip(mel_norm, 0, 255)

    # Save as PNG image
    # Note: PIL expects (height, width) so we use mel_norm directly
    # Mel bins (frequency) = vertical axis (height)
    # Time frames = horizontal axis (width)
    img = Image.fromarray(mel_norm, mode='L')  # Grayscale
    img.save(output_path)

    return mel_spec_db


def generate_spectrogram_colorized(
    audio_path: Path,
    output_path: Path,
    n_mels: int = 128,
    sr: int = 22050,
    colormap: str = 'viridis',
) -> np.ndarray:
    """
    Generate colorized mel spectrogram (more visually distinct).

    Args:
        audio_path: Path to audio file
        output_path: Path to save spectrogram image
        n_mels: Number of mel bins
        sr: Sample rate
        colormap: Matplotlib colormap name

    Returns:
        Mel spectrogram in dB scale
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa is required. Install with: pip install librosa")

    # Load and compute spectrogram
    y, sr_actual = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr_actual,
        n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Apply colormap using matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Normalize to [0, 1]
        mel_norm = (mel_spec_db + 80) / 80
        mel_norm = np.clip(mel_norm, 0, 1)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(mel_norm)  # Returns RGBA

        # Convert to RGB uint8
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        # Save as PNG
        img = Image.fromarray(rgb, mode='RGB')
        img.save(output_path)

    except ImportError:
        print("WARNING: matplotlib not available, falling back to grayscale")
        # Fallback to grayscale
        mel_norm = ((mel_spec_db + 80) / 80 * 255).astype(np.uint8)
        mel_norm = np.clip(mel_norm, 0, 255)
        img = Image.fromarray(mel_norm, mode='L')
        img.save(output_path)

    return mel_spec_db


def process_dataset(
    audio_dir: Path,
    output_dir: Path,
    n_mels: int = 128,
    colorized: bool = False,
    extensions: tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg'),
    verbose: bool = True,
) -> dict:
    """
    Process all audio files in a directory to generate spectrograms.

    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save spectrogram images
        n_mels: Number of mel frequency bins
        colorized: Generate colorized spectrograms (RGB) vs grayscale
        extensions: Audio file extensions to process
        verbose: Print progress messages

    Returns:
        Statistics dict with counts and errors
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))

    if not audio_files:
        print(f"WARNING: No audio files found in {audio_dir}")
        return {'processed': 0, 'errors': 0, 'skipped': 0}

    if verbose:
        print(f"Found {len(audio_files)} audio files in {audio_dir}")
        print(f"Generating {n_mels}-bin mel spectrograms...")

    processed = 0
    errors = 0
    skipped = 0

    for audio_path in audio_files:
        try:
            # Generate output path
            suffix = '_spectrogram_color.png' if colorized else '_spectrogram.png'
            output_path = output_dir / f"{audio_path.stem}{suffix}"

            # Skip if already exists
            if output_path.exists():
                if verbose and processed % 100 == 0:
                    print(f"  Skipped (exists): {audio_path.name}")
                skipped += 1
                continue

            # Generate spectrogram
            if colorized:
                generate_spectrogram_colorized(
                    audio_path,
                    output_path,
                    n_mels=n_mels
                )
            else:
                generate_spectrogram(
                    audio_path,
                    output_path,
                    n_mels=n_mels
                )

            processed += 1

            if verbose and processed % 10 == 0:
                print(f"  Processed {processed}/{len(audio_files)}: {audio_path.name}")

        except Exception as e:
            errors += 1
            print(f"ERROR processing {audio_path}: {e}")
            continue

    if verbose:
        print(f"\n=== Summary ===")
        print(f"Processed: {processed}")
        print(f"Skipped:   {skipped}")
        print(f"Errors:    {errors}")
        print(f"Total:     {len(audio_files)}")
        print(f"Output:    {output_dir}")

    return {
        'processed': processed,
        'errors': errors,
        'skipped': skipped,
        'total': len(audio_files)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate sound pictures (spectrograms) from audio files"
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Directory to save spectrogram images"
    )
    parser.add_argument(
        '--n-mels',
        type=int,
        default=128,
        help="Number of mel frequency bins (default: 128, matches embedding dimension)"
    )
    parser.add_argument(
        '--colorized',
        action='store_true',
        help="Generate colorized spectrograms (RGB) instead of grayscale"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    if not HAS_LIBROSA:
        print("ERROR: librosa is required but not installed.")
        print("Install with: pip install librosa")
        return 1

    # Process dataset
    stats = process_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        n_mels=args.n_mels,
        colorized=args.colorized,
        verbose=not args.quiet
    )

    return 0 if stats['errors'] == 0 else 1


if __name__ == '__main__':
    exit(main())
