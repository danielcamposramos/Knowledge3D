"""
Swarm integration utilities for language ingestion.

Provides a high-level API that accepts modality-specific embeddings, routes them
through the specialised nine-chain swarm, and returns refined vectors alongside
diagnostics and suggested Galaxy coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from knowledge3d.cranium.bridges.nine_chain_specialized_bridge import (
    NineChainSpecializedBridge,
    SwarmDiagnostics,
)
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@dataclass
class SwarmResult:
    refined_embedding: np.ndarray
    position_3d: np.ndarray
    diagnostics: Optional[SwarmDiagnostics]
    modality: str
    language: str
    label: str | None = None


class LanguageSwarmProcessor:
    """
    Wraps the specialised swarm and RPN bridge to provide a simple ingestion API.
    """

    def __init__(
        self,
        swarm_bridge: NineChainSpecializedBridge | None = None,
        rpn_bridge: ThinkingTagRPNBridge | None = None,
        *,
        default_iterations: int = 2,
    ) -> None:
        self.swarm = swarm_bridge or NineChainSpecializedBridge()
        self.rpn = rpn_bridge or ThinkingTagRPNBridge(
            tier=2,
            use_specialized_swarm=True,
            swarm_iterations=default_iterations,
        )
        self.default_iterations = max(1, default_iterations)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_language_embedding(
        self,
        embedding_128: np.ndarray,
        modality: str,
        language: str,
        *,
        iterations: int | None = None,
        include_diagnostics: bool = True,
    ) -> SwarmResult:
        embedding_128 = np.asarray(embedding_128, dtype=np.float32).reshape(-1)
        if embedding_128.shape[0] != 128:
            raise ValueError("Expected embedding of shape (128,) for swarm routing")

        iters = max(1, iterations or self.default_iterations)
        readback = "diagnostics" if include_diagnostics else "output"

        output_embedding, _, _ = self.swarm.execute_swarm(
            embedding_128,
            num_iterations=iters,
            readback_mode=readback,
        )

        diagnostics = (
            self.swarm.get_chain_diagnostics() if include_diagnostics else None
        )

        position_3d = self._embedding_to_position(output_embedding)
        return SwarmResult(
            refined_embedding=output_embedding,
            position_3d=position_3d,
            diagnostics=diagnostics,
            modality=modality,
            language=language,
        )

    def batch_process_language_corpus(
        self,
        embeddings: Sequence[np.ndarray],
        metadata: Sequence[Dict],
        *,
        iterations: int | None = None,
        include_diagnostics: bool = False,
    ) -> List[SwarmResult]:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata sequences must align")

        results: List[SwarmResult] = []
        for emb, meta in zip(embeddings, metadata):
            result = self.process_language_embedding(
                emb,
                modality=meta.get("modality", "unknown"),
                language=meta.get("language", "unknown"),
                iterations=iterations,
                include_diagnostics=include_diagnostics,
            )
            result.label = meta.get("label")
            results.append(result)
        return results

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self.swarm is not None:
            self.swarm.cleanup()
        if self.rpn is not None:
            self.rpn.cleanup()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def fuse_multimodal_embedding(
        self,
        *,
        text_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        visual_emb: Optional[np.ndarray] = None,
        language: str = "en",
        iterations: int | None = None,
        include_diagnostics: bool = True,
    ) -> Dict[str, object]:
        """
        Fuse available modality embeddings and route through the specialised swarm.
        """
        sources = []
        labels = []

        for emb, name in (
            (text_emb, "text"),
            (audio_emb, "audio"),
            (visual_emb, "visual"),
        ):
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                if emb.shape[0] != 128:
                    raise ValueError(f"{name} embedding must have shape (128,), got {emb.shape}")
                sources.append(emb)
                labels.append(name)

        if not sources:
            raise ValueError("At least one modality embedding must be provided")

        if len(sources) == 1:
            fused = sources[0]
        else:
            fused = np.mean(np.vstack(sources), axis=0).astype(np.float32)

        swarm_result = self.process_language_embedding(
            fused,
            modality="multi",
            language=language,
            iterations=iterations,
            include_diagnostics=include_diagnostics,
        )

        return {
            "refined_embedding": swarm_result.refined_embedding,
            "position_3d": swarm_result.position_3d,
            "diagnostics": swarm_result.diagnostics,
            "modalities_used": labels,
            "language": language,
        }

    @staticmethod
    def _embedding_to_position(embedding: np.ndarray) -> np.ndarray:
        """
        Reduce a 128-dim vector to a stable 3D coordinate by splitting the vector
        into three equal bins and computing mean activations.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError("Embedding for position mapping must be 1D")

        bins = np.array_split(embedding, 3)
        coords = np.array([segment.mean() for segment in bins], dtype=np.float32)
        coords -= coords.min()
        denom = coords.max() or 1.0
        return coords / denom


SovereignLanguageSwarmProcessor = LanguageSwarmProcessor


__all__ = ["LanguageSwarmProcessor", "SovereignLanguageSwarmProcessor", "SwarmResult"]
