"""
Utilities to validate procedural compression fidelity.

Implements the Phase 1 Task 1.2 flow:
text → RPN embedding → procedural compression → reconstruction → metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .procedural_compiler import ProceduralCompiler
from .rpn_embedding_engine import RPNEmbeddingEngine


@dataclass(slots=True)
class ProceduralFidelityResult:
    """Container for fidelity metrics."""

    token: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    cosine_similarity: float
    valid: bool
    extra: dict | None = None

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "cosine_similarity": self.cosine_similarity,
            "valid": self.valid,
            "extra": self.extra or {},
        }


class ProceduralFidelityValidator:
    """Validate that procedural programs faithfully reconstruct embeddings."""

    def __init__(
        self,
        rpn_engine: RPNEmbeddingEngine | None = None,
        compiler: ProceduralCompiler | None = None,
        similarity_threshold: float = 0.99,
    ) -> None:
        self.rpn_engine = rpn_engine or RPNEmbeddingEngine(embedding_dim=2048)
        self.compiler = compiler or ProceduralCompiler()
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------ #
    def validate_round_trip(self, text: str) -> ProceduralFidelityResult:
        """Run the compression/decompression loop for a single token."""
        embedding = self.rpn_engine.embed_word(text)
        program = self.compiler.compile_embedding_simple(embedding)
        reconstructed = self.compiler.decompile_simple(program)

        similarity = self._cosine_similarity(embedding, reconstructed)
        compressed_size = len(program)
        original_size = embedding.nbytes
        ratio = float(original_size) / max(1, compressed_size)

        return ProceduralFidelityResult(
            token=text,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            cosine_similarity=similarity,
            valid=similarity >= self.similarity_threshold,
            extra={"mode": "simple"},
        )

    def validate_prototype_sparse(self, text: str) -> ProceduralFidelityResult:
        """Validate prototype-delta sparse compression for a token."""
        embedding = self.rpn_engine.embed_word(text)
        program, meta = self.compiler.compile_prototype_delta(embedding, return_metadata=True)
        if meta.get("codec") == "simple_fallback":
            reconstructed = self.compiler.decompile_simple(program)
        else:
            reconstructed = self.compiler.decompile_prototype_delta(program)

        similarity = self._cosine_similarity(embedding, reconstructed)
        compressed_size = len(program)
        original_size = embedding.nbytes
        ratio = float(original_size) / max(1, compressed_size)

        extra = {"mode": "prototype_delta_sparse", **meta}

        return ProceduralFidelityResult(
            token=text,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            cosine_similarity=similarity,
            valid=similarity >= self.similarity_threshold,
            extra=extra,
        )

    def validate_prototype_dense(self, text: str) -> ProceduralFidelityResult:
        """Validate dense PD02 codec."""
        embedding = self.rpn_engine.embed_word(text)
        program, meta = self.compiler.compile_prototype_delta_dense(embedding, return_metadata=True)
        if meta.get("codec") == "simple_fallback":
            reconstructed = self.compiler.decompile_simple(program)
        else:
            reconstructed = self.compiler.decompile_prototype_delta_dense(program)

        similarity = self._cosine_similarity(embedding, reconstructed)
        compressed_size = len(program)
        original_size = embedding.nbytes
        ratio = float(original_size) / max(1, compressed_size)

        extra = {"mode": "prototype_delta_dense", **meta}

        return ProceduralFidelityResult(
            token=text,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            cosine_similarity=similarity,
            valid=similarity >= self.similarity_threshold,
            extra=extra,
        )

    def validate_prototype_multi(self, text: str) -> ProceduralFidelityResult:
        """Validate multi-prototype PD03 codec."""
        embedding = self.rpn_engine.embed_word(text)
        program, meta = self.compiler.compile_prototype_multi(embedding, return_metadata=True)
        codec = meta.get("codec")
        if codec == "simple_fallback":
            reconstructed = self.compiler.decompile_simple(program)
        elif codec == "multi":
            reconstructed = self.compiler.decompile_prototype_multi(program)
        elif codec == "dense":
            reconstructed = self.compiler.decompile_prototype_delta_dense(program)
        else:
            reconstructed = self.compiler.decompile_prototype_delta(program)

        similarity = self._cosine_similarity(embedding, reconstructed)
        compressed_size = len(program)
        original_size = embedding.nbytes
        ratio = float(original_size) / max(1, compressed_size)

        extra = {"mode": "prototype_delta_multi", **meta}

        return ProceduralFidelityResult(
            token=text,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            cosine_similarity=similarity,
            valid=similarity >= self.similarity_threshold,
            extra=extra,
        )

    def validate_dictionary_sparse(self, text: str) -> ProceduralFidelityResult:
        """Validate dictionary sparse codec."""
        embedding = self.rpn_engine.embed_word(text)
        program, meta = self.compiler.compile_dictionary_sparse(embedding, return_metadata=True)
        codec = meta.get("codec")
        if codec == "simple_fallback":
            reconstructed = self.compiler.decompile_simple(program)
        elif codec == "dict":
            reconstructed = self.compiler.decompile_dictionary_sparse(program)
        elif codec == "dense":
            reconstructed = self.compiler.decompile_prototype_delta_dense(program)
        else:
            reconstructed = self.compiler.decompile_prototype_delta(program)

        similarity = self._cosine_similarity(embedding, reconstructed)
        compressed_size = len(program)
        original_size = embedding.nbytes
        ratio = float(original_size) / max(1, compressed_size)
        extra = {"mode": "dictionary_sparse", **meta}

        return ProceduralFidelityResult(
            token=text,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            cosine_similarity=similarity,
            valid=similarity >= self.similarity_threshold,
            extra=extra,
        )

    def batch_validate(self, tokens: Sequence[str], mode: str = "simple") -> List[ProceduralFidelityResult]:
        """Validate a corpus of tokens."""
        if mode == "prototype_sparse":
            return [self.validate_prototype_sparse(token) for token in tokens]
        if mode == "prototype_dense":
            return [self.validate_prototype_dense(token) for token in tokens]
        if mode == "prototype_multi":
            return [self.validate_prototype_multi(token) for token in tokens]
        if mode == "dictionary_sparse":
            return [self.validate_dictionary_sparse(token) for token in tokens]
        return [self.validate_round_trip(token) for token in tokens]

    # Backwards compatibility alias
    validate_prototype_delta = validate_prototype_sparse

    def summarize(self, results: Iterable[ProceduralFidelityResult]) -> dict:
        """Aggregate batch statistics."""
        results_list = list(results)
        if not results_list:
            return {
                "count": 0,
                "average_compression": 0.0,
                "average_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "valid_ratio": 0.0,
            }

        compression = [item.compression_ratio for item in results_list]
        similarity = [item.cosine_similarity for item in results_list]
        valid = [item.valid for item in results_list]

        return {
            "count": len(results_list),
            "average_compression": float(np.mean(compression)),
            "average_similarity": float(np.mean(similarity)),
            "min_similarity": float(np.min(similarity)),
            "max_similarity": float(np.max(similarity)),
            "valid_ratio": float(np.mean(valid)),
        }

    # ------------------------------------------------------------------ #
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity with epsilon guard."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)
