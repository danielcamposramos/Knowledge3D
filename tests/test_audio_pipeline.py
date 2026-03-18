from __future__ import annotations

import math
from pathlib import Path
import struct
import wave

from knowledge3d.tools.audio_pipeline import (
    _find_ffmpeg,
    _is_whisper_compatible,
    get_audio_metadata,
    preprocess_audio,
    split_by_silence,
)


def _create_test_wav(path: Path, *, rate: int = 16000, channels: int = 1, duration_s: float = 0.25) -> Path:
    frames = int(rate * duration_s)
    amplitude = 16000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for index in range(frames):
            sample = int(amplitude * math.sin((2.0 * math.pi * 440.0 * index) / rate))
            frame = struct.pack("<h", sample)
            handle.writeframesraw(frame * channels)
    return path


def test_find_ffmpeg() -> None:
    path = _find_ffmpeg()
    assert "ffmpeg" in path


def test_preprocess_audio_wav_passthrough(tmp_path: Path) -> None:
    wav_path = _create_test_wav(tmp_path / "test.wav", rate=16000, channels=1)
    result = preprocess_audio(str(wav_path))
    assert result == str(wav_path)
    assert _is_whisper_compatible(result) is True


def test_preprocess_audio_converts_stereo(tmp_path: Path) -> None:
    wav_path = _create_test_wav(tmp_path / "stereo.wav", rate=44100, channels=2)
    result = preprocess_audio(str(wav_path), output_dir=str(tmp_path))
    assert Path(result).is_file()
    assert result != str(wav_path)
    assert _is_whisper_compatible(result) is True


def test_get_audio_metadata(tmp_path: Path) -> None:
    wav_path = _create_test_wav(tmp_path / "probe.wav")
    metadata = get_audio_metadata(str(wav_path))
    assert "format" in metadata or "streams" in metadata


def test_split_by_silence_empty_returns_original(tmp_path: Path) -> None:
    wav_path = _create_test_wav(tmp_path / "continuous.wav")
    segments = split_by_silence(str(wav_path), output_dir=str(tmp_path / "segments"))
    assert isinstance(segments, list)
    assert segments
