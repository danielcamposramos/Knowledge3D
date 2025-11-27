from __future__ import annotations

import numpy as np

from knowledge3d.training.arc_agi.embedders.video_grid_embedder import (
    VideoGridEmbedder,
)
from knowledge3d.training.arc_agi.embedders.audio_grid_embedder import (
    AudioGridEmbedder,
)
from knowledge3d.training.arc_agi.embedders.multimodal_grid_embedder import (
    MultiModalGridEmbedder,
)


class _FakeVideoCodec:
    """Minimal stand-in for TernaryVideoCodec used in tests."""

    def __init__(self, width: int = 32, height: int = 32):
        self.width = width
        self.height = height

    def encode(self, frame: np.ndarray):
        h, w, _ = frame.shape
        seed = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        # Simple ternary pattern: +1 where frame > 0, else 0.
        quantized = np.zeros((h, w, 3), dtype=np.int8)
        mask = np.any(frame > 0, axis=-1)
        quantized[mask] = 1
        return {"seed": seed, "quantized": quantized, "metadata": {}}


class _FakeAudioCodec:
    """Minimal stand-in for TernaryAudioCodec used in tests."""

    def __init__(self, frame_size: int = 1024, n_harmonics: int = 20):
        self.frame_size = frame_size
        self.n_harmonics = n_harmonics

    def encode(self, audio: np.ndarray):
        harmonics = np.ones((self.n_harmonics, 3), dtype=np.float32)
        # Two frames of simple ternary coefficients.
        mdct_q = np.ones((2, self.frame_size), dtype=np.int8)
        return {
            "harmonics": harmonics,
            "mdct_quantized": mdct_q,
            "mdct_metadata": [],
            "frame_size": self.frame_size,
            "hop_size": self.frame_size // 2,
            "sample_rate": 44100,
            "num_samples": int(audio.size),
        }


def _sample_grid():
    return [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]


def test_video_grid_embedder_shape_and_dtype():
    codec = _FakeVideoCodec(width=32, height=32)
    embedder = VideoGridEmbedder(width=32, height=32, codec=codec)
    grid = _sample_grid()
    emb = embedder.grid_to_video_embedding(grid)
    assert isinstance(emb, (list, np.ndarray))
    emb_list = emb if isinstance(emb, list) else emb.tolist()
    assert len(emb_list) == 510


def test_audio_grid_embedder_shape_and_dtype():
    codec = _FakeAudioCodec(frame_size=1024, n_harmonics=20)
    embedder = AudioGridEmbedder(codec=codec)
    grid = _sample_grid()
    emb = embedder.grid_to_audio_embedding(grid, target_dim=512)
    assert isinstance(emb, (list, np.ndarray))
    emb_list = emb if isinstance(emb, list) else emb.tolist()
    assert len(emb_list) == 512


def test_multimodal_grid_embedder_routing_and_shape():
    video_codec = _FakeVideoCodec(width=32, height=32)
    audio_codec = _FakeAudioCodec(frame_size=1024, n_harmonics=20)
    video_embedder = VideoGridEmbedder(width=32, height=32, codec=video_codec)
    audio_embedder = AudioGridEmbedder(codec=audio_codec)

    mm = MultiModalGridEmbedder(
        matryoshka_dim=256,
        video_embedder=video_embedder,
        audio_embedder=audio_embedder,
    )
    grid = _sample_grid()

    emb_balanced = mm.grid_to_multimodal_embedding(grid, routing=0)
    emb_video = mm.grid_to_multimodal_embedding(grid, routing=-1)
    emb_audio = mm.grid_to_multimodal_embedding(grid, routing=1)

    if hasattr(emb_balanced, "shape"):
        assert emb_balanced.shape == (256,)
    else:
        assert len(emb_balanced) == 256

    # Routing should change the fused embedding.
    assert not np.allclose(emb_video, emb_audio)
