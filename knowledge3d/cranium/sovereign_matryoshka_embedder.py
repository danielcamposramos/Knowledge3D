"""Sovereign text-to-Matryoshka embedding bridge.

The RPN embedding engine owns the text projection. Matryoshka consumers see
prefix tiers cut from the same max-dimension vector.
"""

from __future__ import annotations

import math
import threading

from knowledge3d.cranium.bridges.trigram_embed_bridge import TrigramEmbedBridge
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


class SovereignMatryoshkaTextEmbedder:
    """RPN text embedder with a one-projection, many-prefix tier contract."""

    TIER_DIMS: tuple[int, ...] = (64, 128, 512, 2048)

    def __init__(
        self,
        *,
        engine: RPNEmbeddingEngine | None = None,
        max_dim: int = 2048,
        gpu_bridge: TrigramEmbedBridge | None = None,
    ) -> None:
        self.max_dim = int(max_dim)
        if self.max_dim != int(self.TIER_DIMS[-1]):
            raise ValueError(f"unsupported_matryoshka_max_dim:{self.max_dim}")
        self.engine = engine or RPNEmbeddingEngine(embedding_dim=self.max_dim)
        if int(self.engine.embedding_dim) != self.max_dim:
            raise ValueError(
                f"rpn_embedding_dim_mismatch:{int(self.engine.embedding_dim)}:{self.max_dim}"
            )
        if not self.engine.has_gpu_bridge():
            self.engine.attach_gpu_bridge(gpu_bridge or TrigramEmbedBridge())
        self._lock = threading.RLock()

    @staticmethod
    def _normalize(values) -> list[float]:
        flat = [float(value) for value in values]
        if not flat:
            return []
        norm_sq = sum(value * value for value in flat)
        if norm_sq != norm_sq or norm_sq <= 1.0e-16:
            return [0.0 for _ in flat]
        norm = math.sqrt(norm_sq)
        if norm <= 1.0e-8:
            return [0.0 for _ in flat]
        return [float(value / norm) for value in flat]

    def embed_max(self, text: str) -> list[float]:
        """Return the normalized max-dimension RPN embedding for text."""
        with self._lock:
            vector = self.engine.embed_sentence_gpu(str(text or ""))
        values = [float(vector[index]) for index in range(min(len(vector), self.max_dim))]
        if len(values) < self.max_dim:
            values.extend(0.0 for _ in range(self.max_dim - len(values)))
        return self._normalize(values[: self.max_dim])

    def embed_stack(self, text: str) -> dict[int, list[float]]:
        """Return normalized prefix tiers from one max-dimension projection."""
        max_vector = self.embed_max(text)
        return {tier: self._normalize(max_vector[:tier]) for tier in self.TIER_DIMS}

    def embed_tier(self, text: str, tier: int = 64) -> list[float]:
        tier = int(tier)
        if tier not in self.TIER_DIMS:
            raise ValueError(f"unsupported_matryoshka_tier:{tier}")
        return self.embed_stack(text)[tier]


_SINGLETON_LOCK = threading.Lock()
_SINGLETON: SovereignMatryoshkaTextEmbedder | None = None


def get_sovereign_matryoshka_text_embedder(
    *,
    engine: RPNEmbeddingEngine | None = None,
    max_dim: int = 2048,
    gpu_bridge: TrigramEmbedBridge | None = None,
) -> SovereignMatryoshkaTextEmbedder:
    """Return the process-wide sovereign Matryoshka text embedder."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = SovereignMatryoshkaTextEmbedder(
                engine=engine,
                max_dim=max_dim,
                gpu_bridge=gpu_bridge,
            )
        return _SINGLETON


__all__ = [
    "SovereignMatryoshkaTextEmbedder",
    "get_sovereign_matryoshka_text_embedder",
]
