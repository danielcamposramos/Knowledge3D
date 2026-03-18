"""FFmpeg-backed audio preprocessing and pipeline execution for ingestion-path audio."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
import wave


def _find_ffmpeg() -> str:
    for candidate in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if not resolved:
            continue
        try:
            result = subprocess.run(
                [resolved, "-version"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return resolved
    raise FileNotFoundError("FFmpeg not found. Install with: apt install ffmpeg")


def _find_ffprobe() -> str:
    for candidate in ("ffprobe", "/usr/bin/ffprobe", "/usr/local/bin/ffprobe"):
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if not resolved:
            continue
        try:
            result = subprocess.run(
                [resolved, "-version"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return resolved
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path.endswith("ffmpeg"):
        return ffmpeg_path[:-6] + "ffprobe"
    raise FileNotFoundError("ffprobe not found")


def _is_whisper_compatible(path: str) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getframerate() == 16000
                and handle.getnchannels() == 1
                and handle.getsampwidth() == 2
            )
    except Exception:
        return False


def preprocess_audio(
    input_path: str,
    *,
    output_dir: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> str:
    """Convert audio/video to Vibe/Whisper-compatible WAV."""
    inp = Path(input_path)
    if not inp.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if inp.suffix.lower() == ".wav" and _is_whisper_compatible(str(inp)):
        return str(inp)
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_audio_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inp.stem}_16k_mono.wav"
    cmd = [
        _find_ffmpeg(),
        "-y",
        "-i",
        str(inp),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        codec,
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[:500]}")
    return str(out_path)


def extract_audio_from_video(video_path: str, *, output_dir: str | None = None) -> str:
    inp = Path(video_path)
    if not inp.is_file():
        raise FileNotFoundError(video_path)
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_audio_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inp.stem}_audio.wav"
    cmd = [
        _find_ffmpeg(),
        "-y",
        "-i",
        str(inp),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"FFmpeg extraction failed: {result.stderr[:500]}")
    return str(out_path)


def split_by_silence(
    audio_path: str,
    *,
    min_silence_ms: int = 500,
    silence_thresh_db: int = -40,
    output_dir: str | None = None,
) -> list[str]:
    """Split audio into segments using FFmpeg silencedetect."""
    inp = Path(audio_path)
    if not inp.is_file():
        raise FileNotFoundError(audio_path)
    ffmpeg_bin = _find_ffmpeg()
    detect_cmd = [
        ffmpeg_bin,
        "-i",
        str(inp),
        "-af",
        f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_ms / 1000.0}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(detect_cmd, check=False, capture_output=True, text=True, timeout=120)
    boundaries = [float(match.group(1)) for match in re.finditer(r"silence_end:\s*([\d.]+)", result.stderr or "")]
    if not boundaries:
        return [str(inp)]
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="k3d_segments_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    segments: list[str] = []
    previous = 0.0
    for index, boundary in enumerate(boundaries):
        if boundary <= previous:
            continue
        seg_path = out_dir / f"segment_{index:04d}.wav"
        split_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(inp),
            "-ss",
            str(previous),
            "-to",
            str(boundary),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(seg_path),
        ]
        subprocess.run(split_cmd, check=False, capture_output=True, timeout=60)
        if seg_path.is_file() and seg_path.stat().st_size > 100:
            segments.append(str(seg_path))
        previous = boundary
    final_seg = out_dir / f"segment_{len(segments):04d}.wav"
    tail_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(inp),
        "-ss",
        str(previous),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(final_seg),
    ]
    subprocess.run(tail_cmd, check=False, capture_output=True, timeout=60)
    if final_seg.is_file() and final_seg.stat().st_size > 100:
        segments.append(str(final_seg))
    return segments or [str(inp)]


def get_audio_metadata(audio_path: str) -> dict[str, Any]:
    """Probe audio metadata via ffprobe with a wave fallback."""
    inp = Path(audio_path)
    if not inp.is_file():
        raise FileNotFoundError(audio_path)
    try:
        cmd = [
            _find_ffprobe(),
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(inp),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    try:
        with wave.open(str(inp), "rb") as handle:
            return {
                "format": {
                    "filename": str(inp),
                    "duration": handle.getnframes() / float(handle.getframerate() or 1),
                },
                "streams": [
                    {
                        "codec_name": "pcm_s16le",
                        "sample_rate": str(handle.getframerate()),
                        "channels": handle.getnchannels(),
                    }
                ],
            }
    except Exception:
        return {}


def _ffmpeg_convert(input_path: str, *, ar: int | None = None, ac: int | None = None) -> str:
    inp = Path(input_path)
    out_path = Path(tempfile.mkdtemp(prefix="k3d_audio_op_")) / f"{inp.stem}_convert.wav"
    cmd = [_find_ffmpeg(), "-y", "-i", str(inp)]
    if ar is not None:
        cmd.extend(["-ar", str(ar)])
    if ac is not None:
        cmd.extend(["-ac", str(ac)])
    cmd.extend(["-c:a", "pcm_s16le", str(out_path)])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg convert failed: {result.stderr[:500]}")
    return str(out_path)


def _ffmpeg_filter(input_path: str, filter_expr: str) -> str:
    inp = Path(input_path)
    out_path = Path(tempfile.mkdtemp(prefix="k3d_audio_op_")) / f"{inp.stem}_filter.wav"
    cmd = [
        _find_ffmpeg(),
        "-y",
        "-i",
        str(inp),
        "-af",
        filter_expr,
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg filter failed: {result.stderr[:500]}")
    return str(out_path)


def _ffmpeg_trim(input_path: str, start: float, end: float) -> str:
    inp = Path(input_path)
    out_path = Path(tempfile.mkdtemp(prefix="k3d_audio_op_")) / f"{inp.stem}_trim.wav"
    cmd = [
        _find_ffmpeg(),
        "-y",
        "-i",
        str(inp),
        "-ss",
        str(float(start)),
        "-to",
        str(float(end)),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg trim failed: {result.stderr[:500]}")
    return str(out_path)


def _ffmpeg_concat(inputs: tuple[str, ...]) -> str:
    if not inputs:
        raise ValueError("AUDIO_CONCAT requires at least one input")
    file_list_dir = Path(tempfile.mkdtemp(prefix="k3d_audio_concat_"))
    file_list_path = file_list_dir / "inputs.txt"
    out_path = file_list_dir / "concat.wav"
    lines = [f"file '{Path(item).resolve()}'" for item in inputs]
    file_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        _find_ffmpeg(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[:500]}")
    return str(out_path)


AUDIO_OPERATIONS: dict[str, Callable[..., Any]] = {
    "AUDIO_RESAMPLE": lambda inp, rate: _ffmpeg_convert(inp, ar=int(rate)),
    "AUDIO_MONO": lambda inp: _ffmpeg_convert(inp, ac=1),
    "AUDIO_NORMALIZE": lambda inp: _ffmpeg_filter(inp, "loudnorm"),
    "AUDIO_TRIM": lambda inp, start, end: _ffmpeg_trim(inp, float(start), float(end)),
    "AUDIO_CONCAT": lambda *inputs: _ffmpeg_concat(tuple(str(item) for item in inputs)),
    "AUDIO_GAIN": lambda inp, db: _ffmpeg_filter(inp, f"volume={float(db)}dB"),
    "AUDIO_FADE_IN": lambda inp, dur: _ffmpeg_filter(inp, f"afade=t=in:d={float(dur)}"),
    "AUDIO_FADE_OUT": lambda inp, dur: _ffmpeg_filter(inp, f"afade=t=out:d={float(dur)}"),
    "AUDIO_PROBE": lambda inp: get_audio_metadata(inp),
    "AUDIO_SPLIT_SILENCE": lambda inp, **kwargs: split_by_silence(inp, **kwargs),
}


def execute_audio_pipeline(input_path: str, operations: list[tuple[Any, ...]]) -> Any:
    current: Any = str(input_path)
    for op_tuple in operations:
        if not op_tuple:
            continue
        op_name = str(op_tuple[0]).strip().upper()
        fn = AUDIO_OPERATIONS.get(op_name)
        if fn is None:
            raise ValueError(f"Unknown audio operation: {op_name}")
        args = tuple(op_tuple[1:])
        if op_name == "AUDIO_CONCAT":
            current = fn(current, *args)
        else:
            current = fn(current, *args)
    return current


__all__ = [
    "AUDIO_OPERATIONS",
    "_find_ffmpeg",
    "_find_ffprobe",
    "_is_whisper_compatible",
    "execute_audio_pipeline",
    "extract_audio_from_video",
    "get_audio_metadata",
    "preprocess_audio",
    "split_by_silence",
]
