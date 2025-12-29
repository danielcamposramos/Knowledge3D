from __future__ import annotations

from typing import List, Sequence

from knowledge3d.training.arc_agi.sovereign_utils import pad_or_truncate

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
    ) -> List[float]:
        """
        Compute fused embedding with ternary routing.

        Args:
            grid: ARC grid.
            routing: -1 (video-heavy), 0 (balanced), +1 (audio-heavy).
        """
        video_emb = self.video_embedder.grid_to_video_embedding(grid)
        audio_emb = self.audio_embedder.grid_to_audio_embedding(grid, target_dim=512)

        # Ternary routing weights.
        if routing == -1:
            w_video, w_audio = 0.8, 0.2
        elif routing == 1:
            w_video, w_audio = 0.2, 0.8
        else:
            w_video, w_audio = 0.5, 0.5

        # Pad video to match audio length.
        max_len = max(len(video_emb), len(audio_emb))
        video_padded = pad_or_truncate(video_emb, max_len, 0.0)
        audio_padded = pad_or_truncate(audio_emb, max_len, 0.0)

        fused = [
            w_video * float(video_padded[i]) + w_audio * float(audio_padded[i])
            for i in range(max_len)
        ]

        return pad_or_truncate(fused, self.matryoshka_dim, 0.0)
