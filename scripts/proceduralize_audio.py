#!/usr/bin/env python3
"""
Proceduralize existing audio into compact harmonic seeds (letters/characters).

Supports two modes:
- Manifest-driven: provide a CSV/JSONL manifest with fields {path,text,lang}; only
  entries whose text is a single character are converted. Use this for datasets
  like minds14 / multilingual librispeech where filenames do not contain labels.
- Filename fallback: scans audio roots and attempts to parse lang/phoneme from
  basename "<lang>_<phoneme>_*.ext" (legacy path).

Advantages: produces small per-letter seeds to attach to character stars; no
assumption of filename labels when a manifest is provided.

Usage:
    PYTHONPATH=. python3 scripts/proceduralize_audio.py \
        --output /K3D/Knowledge3D.local/datasets/procedural_audio_seeds.jsonl \
        --manifest /path/to/manifest.csv --manifest-format csv
    # or with JSONL manifest containing {"path": "...", "text": "A", "lang": "en"}

Audio roots searched (fallback):
    /K3D/Knowledge3D.local/datasets/
    /K3D/K3D_llama_cpp/datasets/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np


DATA_ROOTS = [
    Path("/K3D/Knowledge3D.local/datasets"),
    Path("/K3D/K3D_llama_cpp/datasets"),
]

SUPPORTED_EXTS = {".wav", ".flac", ".mp3"}


@dataclass
class AudioSeed:
    language: str
    phoneme: str
    sample_rate: int
    duration_sec: float
    harmonics: List[Tuple[float, float, float]]  # (freq, amp, phase)
    envelope: Tuple[float, float, float, float]  # ADSR in seconds
    noise_level: float
    source: str


def list_audio_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in SUPPORTED_EXTS:
            files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def parse_labels(path: Path) -> Tuple[str, str]:
    """
    Parse language and phoneme from basename: lang_phoneme_*.ext -> (lang, phoneme).
    Fallbacks to ("unknown", stem).
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "unknown", stem


def load_audio(path: Path) -> Tuple[int, np.ndarray]:
    """
    Load audio as mono float32. Uses soundfile if available; falls back to wave
    for WAV; or uses ffmpeg to decode to WAV in a temp file.
    """
    # Try soundfile
    try:
        import soundfile as sf

        data, sr = sf.read(path)
    except Exception:
        # Try wave (WAV only)
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                raw = wf.readframes(n)
                dtype = np.int16 if wf.getsampwidth() == 2 else np.int8
                data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                data /= np.abs(data).max() + 1e-9
        except Exception:
            # Fallback to ffmpeg decode
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(tmp_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                import soundfile as sf

                data, sr = sf.read(tmp_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    if data.ndim > 1:
        data = data.mean(axis=1)
    return int(sr), data.astype(np.float32)


def dominant_harmonics(samples: np.ndarray, sr: int, n_harmonics: int = 6) -> List[Tuple[float, float, float]]:
    """Return top-N harmonics (freq, amp, phase)."""
    n = len(samples)
    if n == 0:
        return []
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec = np.fft.rfft(samples)
    mags = np.abs(spec)
    idx = np.argsort(mags)[::-1][:n_harmonics]
    harmonics: List[Tuple[float, float, float]] = []
    for i in idx:
        freq = float(freqs[i])
        amp = float(mags[i] / (mags.max() + 1e-9))
        phase = float(np.angle(spec[i]))
        harmonics.append((freq, amp, phase))
    return harmonics


def estimate_envelope(samples: np.ndarray, sr: int) -> Tuple[float, float, float, float]:
    """Crude ADSR estimation using amplitude percentiles."""
    n = len(samples)
    if n == 0:
        return 0.005, 0.03, 0.7, 0.05
    env = np.abs(samples)
    attack_end = max(1, int(0.05 * n))
    attack = attack_end / sr
    sustain_level = float(np.percentile(env, 70))
    decay = 0.03
    release = 0.05
    return float(attack), float(decay), sustain_level, float(release)


def compute_noise_level(samples: np.ndarray) -> float:
    """Estimate noise as proportion of RMS."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return float(min(0.2, rms * 0.5))


def process_file(path: Path) -> Optional[AudioSeed]:
    lang, phoneme = parse_labels(path)
    try:
        sr, audio = load_audio(path)
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
        return None
    duration = len(audio) / float(sr) if sr > 0 else 0.0
    harms = dominant_harmonics(audio, sr)
    env = estimate_envelope(audio, sr)
    noise = compute_noise_level(audio)
    return AudioSeed(
        language=lang,
        phoneme=phoneme,
        sample_rate=sr,
        duration_sec=duration,
        harmonics=harms,
        envelope=env,
        noise_level=noise,
        source=str(path),
    )


def load_manifest(manifest_path: Path, fmt: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if fmt == "csv":
        import csv

        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    elif fmt == "jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    else:
        raise ValueError("manifest-format must be 'csv' or 'jsonl'")
    return rows


def process_manifest(rows: List[Dict[str, str]]) -> List[AudioSeed]:
    seeds: List[AudioSeed] = []
    for row in rows:
        path = Path(row.get("path", "").strip())
        text = row.get("text", "").strip()
        phoneme = row.get("phoneme", "").strip()
        lang = row.get("lang", row.get("language", "unknown")).strip() or "unknown"
        if len(text) != 1 and not phoneme:
            continue  # only letters/single characters for now
        if not phoneme:
            phoneme = text
        if not path.exists():
            continue
        seed = process_file_with_override(path, lang, phoneme)
        if seed:
            seeds.append(seed)
    return seeds


def process_file_with_override(path: Path, lang: str, phoneme: str) -> Optional[AudioSeed]:
    try:
        sr, audio = load_audio(path)
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
        return None
    duration = len(audio) / float(sr) if sr > 0 else 0.0
    harms = dominant_harmonics(audio, sr)
    env = estimate_envelope(audio, sr)
    noise = compute_noise_level(audio)
    return AudioSeed(
        language=lang,
        phoneme=phoneme,
        sample_rate=sr,
        duration_sec=duration,
        harmonics=harms,
        envelope=env,
        noise_level=noise,
        source=str(path),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/procedural_audio_seeds.jsonl"),
        help="Output JSONL path for procedural seeds",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV/JSONL manifest with fields path,text,lang; only single-char text is processed",
    )
    ap.add_argument(
        "--manifest-format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Manifest format if provided",
    )
    args = ap.parse_args()

    seeds: List[AudioSeed] = []
    if args.manifest:
        rows = load_manifest(args.manifest, args.manifest_format)
        seeds.extend(process_manifest(rows))
    else:
        files = list_audio_files(DATA_ROOTS)
        for path in files:
            seed = process_file(path)
            if seed:
                seeds.append(seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(json.dumps(asdict(seed)) + "\n")

    # Summary
    by_lang = {}
    for s in seeds:
        by_lang.setdefault(s.language, 0)
        by_lang[s.language] += 1
    print(f"Seeds written: {len(seeds)} -> {args.output}")
    print("By language:", by_lang)


if __name__ == "__main__":
    main()
