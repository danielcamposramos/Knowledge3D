"""
Phase H integration helper for adaptive procedural compression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .adaptive_procedural_bridge import AdaptiveDimensionCompressor, QualityLevel
from .procedural_galaxy import ProceduralGalaxy


class PhaseHProceduralIntegration:
    """
    Connects Matryoshka embeddings to adaptive procedural compression.

    The integration assumes the caller already produced a Matryoshka-aligned
    embedding (e.g., via `MatryoshkaTRM.project_vector`). This class simply
    handles compression/decompression and metadata capture.
    """

    def __init__(
        self,
        compressor: AdaptiveDimensionCompressor,
        matryoshka_model=None,
        procedural_galaxy: Optional[ProceduralGalaxy] = None,
        enable_compression: bool = True,
    ) -> None:
        self.compressor = compressor
        self.matryoshka_model = matryoshka_model
        self.procedural_galaxy = procedural_galaxy
        self.enable_compression = enable_compression

    def compress_embedding(
        self,
        embedding: np.ndarray,
        quality: QualityLevel = "fast",
        return_metadata: bool = False,
    ) -> Tuple[bytes, dict] | bytes:
        """
        Compress a precomputed embedding.
        """
        if not self.enable_compression:
            payload = embedding.astype(np.float32).tobytes()
            return (payload, {"compression": 1.0}) if return_metadata else payload
        return self.compressor.compress(embedding, quality=quality, return_metadata=return_metadata)

    def decompress_embedding(
        self,
        payload: bytes,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decompress a procedural program back to an embedding.
        """
        if not self.enable_compression:
            return np.frombuffer(payload, dtype=np.float32)
        return self.compressor.decompress(payload, target_dim=target_dim)

    # ------------------------------------------------------------------ #
    # Matryoshka helpers
    # ------------------------------------------------------------------ #
    def compress_matryoshka_vector(
        self,
        base_vector: np.ndarray,
        quality: QualityLevel = "fast",
        store_key: Optional[str] = None,
    ) -> Tuple[bytes, dict]:
        """
        Project a base vector via Matryoshka (if available) and compress it.
        """
        if self.matryoshka_model is None:
            raise RuntimeError("Matryoshka model not attached to integration helper.")

        target_dim = self.compressor.dimension_map[quality]
        projected = self.matryoshka_model.project_vector(base_vector, target_dim)
        program, metadata = self.compress_embedding(projected, quality=quality, return_metadata=True)

        if store_key and self.procedural_galaxy is not None:
            compression = metadata.get("actual_compression", 1.0)
            self.procedural_galaxy.store_program(store_key, program, compression_ratio=compression)

        return program, metadata

    def embed_project_and_compress(
        self,
        rpn_embedding: np.ndarray,
        quality: QualityLevel = "fast",
        store_key: Optional[str] = None,
    ) -> Tuple[bytes, dict]:
        """
        Alias for `compress_matryoshka_vector`, provided for clarity.
        """
        return self.compress_matryoshka_vector(rpn_embedding, quality=quality, store_key=store_key)

    def store_program(
        self,
        key: str,
        program: bytes,
        metadata: Optional[dict] = None,
    ) -> None:
        """Persist program in Procedural Galaxy if configured."""
        if self.procedural_galaxy is None:
            raise RuntimeError("ProceduralGalaxy not configured; cannot store program.")
        compression = 1.0
        if metadata and "actual_compression" in metadata:
            compression = metadata["actual_compression"]
        self.procedural_galaxy.store_program(key, program, compression_ratio=compression)
