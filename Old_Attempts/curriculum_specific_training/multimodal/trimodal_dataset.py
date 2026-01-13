"""
Trimodal dataset utilities for Phase G multi-modal training.

Provides dataclass wrappers for each modality plus helpers to derive compact
embeddings from text, image, and audio payloads.

The goal is to transform the JSONL dataset produced by
`scripts/prepare_trimodal_dataset.py` into numerical tensors that Phase G
training and specialist pipelines can consume.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.ptx.modality_ops import PTXModalityOps

__all__ = [
    "TextEntry",
    "ImageEntry",
    "AudioEntry",
    "TrimodalRecord",
    "load_trimodal_dataset",
    "compute_embeddings",
    "save_embeddings",
    "embed_image",
    "embed_audio",
    "hash_embedding",
]


# ---------------------------------------------------------------------------
# Dataclasses representing each modality
# ---------------------------------------------------------------------------

@dataclass
class TextEntry:
    content: str
    metadata: Optional[Dict[str, object]] = None


@dataclass
class ImageEntry:
    path: str
    caption: Optional[str] = None


@dataclass
class AudioEntry:
    path: str
    transcript: Optional[str] = None
    language: Optional[str] = None
    source: Optional[str] = None


@dataclass
class TrimodalRecord:
    """Single sample from the tri-modal dataset."""

    id: str
    source: str
    modalities: Sequence[str]
    text: Optional[TextEntry]
    image: Optional[ImageEntry]
    audio: Optional[AudioEntry]
    extra: Optional[Dict[str, object]] = None

    def has_text(self) -> bool:
        return self.text is not None and bool(self.text.content.strip())

    def has_image(self) -> bool:
        return self.image is not None and bool(self.image.path)

    def has_audio(self) -> bool:
        return self.audio is not None and bool(self.audio.path)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _stable_seed(payload: str) -> int:
    """Return a deterministic uint32 seed for the provided payload."""
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def hash_embedding(payload: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding when raw modality data is unavailable."""
    rng = np.random.default_rng(seed=_stable_seed(payload))
    vec = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
    return _normalise(vec)


def embed_image(image_path: Path, caption: Optional[str], dim: int = 128, gpu_ops: Optional[PTXModalityOps] = None) -> np.ndarray:
    """
    Load an image from disk and derive a compact embedding using GPU acceleration.

    Falls back to hashing when the file cannot be read or GPU is unavailable.
    """
    if not image_path.exists():
        source = caption or image_path.as_posix()
        return hash_embedding(f"image-missing:{source}", dim=dim)

    try:
        # Try GPU-accelerated path first
        if gpu_ops is not None:
            features, _ = gpu_ops.image_features(str(image_path))
            # Tile or truncate to match desired dimension
            if features.size >= dim:
                return _normalise(features[:dim])
            else:
                padded = np.zeros(dim, dtype=np.float32)
                padded[:features.size] = features
                return _normalise(padded)
    except Exception:
        pass  # Fall through to CPU fallback

    # CPU fallback: PIL-based histogram
    try:
        from PIL import Image  # type: ignore
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((32, 32))
            pixels = np.asarray(gray, dtype=np.float32).reshape(-1)
            hist, _ = np.histogram(pixels, bins=dim, range=(0, 255), density=True)
            return _normalise(hist.astype(np.float32))
    except Exception:
        source = caption or image_path.as_posix()
        return hash_embedding(f"image-error:{source}", dim=dim)


def embed_audio(audio_path: Path, transcript: Optional[str], dim: int = 128, gpu_ops: Optional[PTXModalityOps] = None) -> np.ndarray:
    """
    Load an audio waveform and compute a lightweight spectral fingerprint using GPU acceleration.

    Falls back to CPU FFT when GPU is unavailable or file cannot be read.
    """
    if not audio_path.exists():
        payload = transcript or audio_path.as_posix()
        return hash_embedding(f"audio-missing:{payload}", dim=dim)

    try:
        # Try GPU-accelerated path first
        if gpu_ops is not None:
            features, _ = gpu_ops.audio_features(str(audio_path))
            # Tile or truncate to match desired dimension
            if features.size >= dim:
                return _normalise(features[:dim])
            else:
                padded = np.zeros(dim, dtype=np.float32)
                padded[:features.size] = features
                return _normalise(padded)
    except Exception:
        pass  # Fall through to CPU fallback

    # CPU fallback: NumPy FFT-based
    suffix = audio_path.suffix.lower()
    if suffix != ".wav":
        payload = transcript or audio_path.as_posix()
        return hash_embedding(f"audio-hash:{payload}", dim=dim)

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                payload = transcript or audio_path.as_posix()
                return hash_embedding(f"audio-width:{payload}", dim=dim)

            waveform = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if wav_file.getnchannels() > 1:
                waveform = waveform.reshape(-1, wav_file.getnchannels()).mean(axis=1)

            spectrum = np.fft.rfft(waveform)
            magnitudes = np.abs(spectrum)
            if magnitudes.size < dim:
                padded = np.zeros(dim, dtype=np.float32)
                padded[: magnitudes.size] = magnitudes.astype(np.float32)
                magnitudes = padded
            else:
                magnitudes = magnitudes[:dim].astype(np.float32)
            return _normalise(magnitudes)
    except (wave.Error, OSError):
        payload = transcript or audio_path.as_posix()
        return hash_embedding(f"audio-error:{payload}", dim=dim)


# ---------------------------------------------------------------------------
# Dataset loading and embedding generation
# ---------------------------------------------------------------------------

def load_trimodal_dataset(jsonl_path: Path) -> Iterator[TrimodalRecord]:
    """Yield `TrimodalRecord` entries from the combined JSONL dataset."""
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text")
            image = record.get("image")
            audio = record.get("audio")

            yield TrimodalRecord(
                id=record["id"],
                source=record.get("source", ""),
                modalities=record.get("modalities", []),
                text=TextEntry(**text) if isinstance(text, dict) else None,
                image=ImageEntry(**image) if isinstance(image, dict) else None,
                audio=AudioEntry(**audio) if isinstance(audio, dict) else None,
                extra=record.get("extra"),
            )


def compute_embeddings(
    dataset_path: Path,
    embedding_dim: int = 128,
    limit: Optional[int] = None,
    use_gpu: bool = True,
) -> List[Dict[str, object]]:
    """
    Compute per-sample embeddings for all modalities.

    Args:
        dataset_path: Path to trimodal JSONL dataset
        embedding_dim: Target embedding dimension
        limit: Optional limit on number of records to process
        use_gpu: Whether to use GPU acceleration for image/audio (default: True)

    Returns a list of dictionaries with the following keys:
        - id
        - source
        - modalities
        - text_embedding (optional)
        - image_embedding (optional)
        - audio_embedding (optional)
        - fused_embedding (modal-average)
    """
    embedder = RPNEmbeddingEngine(embedding_dim=embedding_dim)

    # Initialize GPU ops if requested
    gpu_ops: Optional[PTXModalityOps] = None
    if use_gpu:
        try:
            gpu_ops = PTXModalityOps()
            print("[GPU] PTXModalityOps initialized for image/audio processing")
        except Exception as exc:
            print(f"[GPU] Failed to initialize PTXModalityOps, falling back to CPU: {exc}")

    outputs: List[Dict[str, object]] = []
    for idx, record in enumerate(load_trimodal_dataset(dataset_path), start=1):
        if limit is not None and idx > limit:
            break

        result: Dict[str, object] = {
            "id": record.id,
            "source": record.source,
            "modalities": list(record.modalities),
        }

        embeddings: List[np.ndarray] = []

        if record.has_text():
            text_content = record.text.content
            text_embedding = embedder.embed_sentence(text_content)
            result["text_embedding"] = text_embedding.tolist()
            embeddings.append(text_embedding)

        if record.has_image():
            image_path = Path(record.image.path)
            image_embedding = embed_image(image_path, record.image.caption, dim=embedding_dim, gpu_ops=gpu_ops)
            result["image_embedding"] = image_embedding.tolist()
            embeddings.append(image_embedding)

        if record.has_audio():
            audio_path = Path(record.audio.path)
            audio_embedding = embed_audio(audio_path, record.audio.transcript, dim=embedding_dim, gpu_ops=gpu_ops)
            result["audio_embedding"] = audio_embedding.tolist()
            embeddings.append(audio_embedding)

        if embeddings:
            fused = _normalise(np.mean(np.vstack(embeddings), axis=0))
            result["fused_embedding"] = fused.tolist()

        outputs.append(result)

    return outputs


def save_embeddings(
    embeddings: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    """Persist computed embeddings to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in embeddings:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")
