from __future__ import annotations

from typing import Sequence

import numpy as np

from .video_grid_embedder import VideoGridEmbedder
from .audio_grid_embedder import AudioGridEmbedder


class MultiModalGridEmbedder:
    """
    Fuse video (spatial) and audio (temporal) embeddings for ARC grids.

    Uses ternary routing:
        -1 → video-heavy
         0 → balanced
        +1 → audio-heavy
    """

    def __init__(
        self,
        matryoshka_dim: int = 512,
        video_embedder: VideoGridEmbedder | None = None,
        audio_embedder: AudioGridEmbedder | None = None,
    ):
        self.matryoshka_dim = int(matryoshka_dim)
        self.video_embedder = video_embedder or VideoGridEmbedder()
        self.audio_embedder = audio_embedder or AudioGridEmbedder()

    def grid_to_multimodal_embedding(
        self,
        grid: Sequence[Sequence[int]],
        routing: int = 0,
    ) -> np.ndarray:
        """
        Compute fused embedding with ternary routing.

        Args:
            grid: ARC grid.
            routing: -1 (video-heavy), 0 (balanced), +1 (audio-heavy).
        """
        video_emb = self.video_embedder.grid_to_video_embedding(grid)
        audio_emb = self.audio_embedder.grid_to_audio_embedding(
            grid, target_dim=512
        )

        # Ternary routing weights.
        if routing == -1:
            w_video, w_audio = 0.8, 0.2
        elif routing == 1:
            w_video, w_audio = 0.2, 0.8
        else:
            w_video, w_audio = 0.5, 0.5

        # Pad video to match audio length.
        video_padded = np.zeros_like(audio_emb, dtype=np.float32)
        v_len = min(video_emb.size, video_padded.size)
        video_padded[:v_len] = video_emb[:v_len].astype(np.float32)

        fused = (w_video * video_padded) + (w_audio * audio_emb.astype(np.float32))

        # Matryoshka projection.
        if fused.size == self.matryoshka_dim:
            return fused.astype(np.float32, copy=False)
        if fused.size > self.matryoshka_dim:
            return fused[: self.matryoshka_dim].astype(np.float32, copy=False)

        out = np.zeros(self.matryoshka_dim, dtype=np.float32)
        out[: fused.size] = fused.astype(np.float32)
        return out

