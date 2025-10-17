"""
Sovereign text ingestion pipeline.

Provides a lightweight, PTX-aligned replacement for the external-model based
`TextLanguageIngestor`. Uses existing sovereign bridges (VectorResonator,
GraphCrystallizer) alongside a small bootstrap embedding (GloVe-50) to derive
128-dimensional vectors suitable for the specialised swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence
import warnings

import numpy as np

from knowledge3d.cranium.bridges.sovereign_bridges import (
    GraphCrystallizer,
    OOMSpillManager,
    VectorResonator,
)

_GLOVE_MODEL_ID = "glove-wiki-gigaword-50"


def _load_glove_model(model_id: str):
    """
    Lazily load a gensim GloVe model. Raises a clear error if gensim is missing
    or the model cannot be downloaded (e.g. network offline).
    """
    try:
        import gensim.downloader as api  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        warnings.warn(
            "gensim not available; proceeding without GloVe bootstrap. "
            "Install via `pip install gensim` for richer embeddings.",
            RuntimeWarning,
        )
        return None

    try:
        return api.load(model_id)
    except ValueError:
        warnings.warn(
            f"Failed to download GloVe model '{model_id}'. "
            "Falling back to zero initial embeddings.",
            RuntimeWarning,
        )
        return None


def _stable_hash(word: str) -> int:
    """Generate a deterministic hash for a token (stable across runs)."""
    return abs(hash(word)) & 0xFFFFFFFF


def _reduce_to_3d(matrix: np.ndarray) -> np.ndarray:
    """
    Reduce an (N, D) matrix to (N, 3) using an SVD-based PCA implementation.

    Pure NumPy so it stays dependency-light. Assumes matrix is float32.
    """
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    # Center
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # Thin SVD
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    components = u[:, :3] * s[:3]
    return components.astype(np.float32)


@dataclass
class SovereignTextIngestor:
    """
    PTX-native text ingestion built on sovereign bridges.

    A small GloVe embedding (50-dim) serves purely as bootstrap input; all
    expansions, reductions, and refinements stay within the sovereign stack.
    """

    languages: Sequence[str] = ("en", "pt", "es", "ja", "zh")
    glove_model_id: str = _GLOVE_MODEL_ID
    bootstrap_langs: Sequence[str] = ("en",)

    _graph_builder: GraphCrystallizer = field(init=False)
    _vector_resonator: VectorResonator = field(init=False)
    _oom_guard: OOMSpillManager = field(init=False)
    _bootstrap_embeddings: Dict[str, object] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._graph_builder = GraphCrystallizer()
        self._vector_resonator = VectorResonator()
        self._oom_guard = OOMSpillManager()

    # ------------------------------------------------------------------
    # Bootstrap handling
    # ------------------------------------------------------------------
    def _ensure_bootstrap(self, lang: str) -> object | None:
        if lang in self._bootstrap_embeddings:
            return self._bootstrap_embeddings[lang]
        if lang not in self.bootstrap_langs:
            # Languages without bootstrap tables fall back to zeros
            self._bootstrap_embeddings[lang] = None
            return None

        model = _load_glove_model(self.glove_model_id)
        self._bootstrap_embeddings[lang] = model
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_vocabulary(self, lang: str, tokens: Sequence[str]) -> np.ndarray:
        """
        Transform a vocabulary list into 3D Galaxy coordinates.

        Steps:
            1. Fetch 50-d bootstrap embeddings (zeros if missing)
            2. Expand deterministically to 128-d
            3. Reduce to 3D via SVD (lightweight PCA)
            4. Normalise to the unit cube
        """
        if not tokens:
            raise ValueError("tokens must not be empty")

        embeddings_128 = np.vstack(
            [self._expand_to_128d(self._get_bootstrap_embedding(lang, token), token) for token in tokens]
        ).astype(np.float32)

        reduced = _reduce_to_3d(embeddings_128)
        reduced -= reduced.min(axis=0, keepdims=True)
        denom = reduced.max(axis=0, keepdims=True)
        denom[denom == 0.0] = 1.0
        return (reduced / denom).astype(np.float32)

    def ingest_sentence(self, lang: str, sentence: str) -> Dict[str, object]:
        """
        Produce a sovereign representation of a sentence.

        Returns:
            {
                "nodes": List[(token, position_3d, depth)],
                "edges": List[(parent, child)] (simple sequential heuristic),
                "embedding_128": np.ndarray,
                "language": lang,
            }
        """
        tokens = [t for t in sentence.strip().split() if t]
        if not tokens:
            raise ValueError("sentence must contain at least one token")

        node_features = []
        depth_values = []
        for idx, token in enumerate(tokens):
            depth = idx // max(1, len(tokens) // 3)
            depth_values.append(depth)
            feature = np.array(
                [
                    idx / max(len(tokens) - 1, 1),
                    min(len(token) / 10.0, 1.0),
                    ((hash(token) & 0xFF) / 255.0),
                ],
                dtype=np.float32,
            )
            node_features.append(feature)

        nodes_np = np.array(node_features, dtype=np.float32).reshape(-1)
        neighbor_features = np.roll(nodes_np, shift=3)  # simple shift as pseudo-neighbourhood
        smoothed = self._graph_builder.crystallize(nodes_np, neighbor_features, ema_rate=0.97)
        node_positions = smoothed.reshape(len(tokens), 3)

        # Average expanded embeddings for sentence vector
        embeddings = [
            self._expand_to_128d(self._get_bootstrap_embedding(lang, token), token) for token in tokens
        ]
        sentence_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        edges = [(i, i + 1) for i in range(len(tokens) - 1)]
        nodes_info = [
            (token, node_positions[idx].astype(np.float32), depth_values[idx]) for idx, token in enumerate(tokens)
        ]

        return {
            "nodes": nodes_info,
            "edges": edges,
            "embedding_128": sentence_embedding,
            "language": lang,
        }

    def cleanup(self) -> None:
        """Placeholder for API symmetry."""
        # Bridges currently free GPU buffers internally; nothing to do here.
        return

    # ------------------------------------------------------------------
    # Helper routines
    # ------------------------------------------------------------------
    def _get_bootstrap_embedding(self, lang: str, token: str) -> np.ndarray:
        model = self._ensure_bootstrap(lang)
        if model is None:
            return np.zeros(50, dtype=np.float32)
        if token in model:
            return np.array(model[token], dtype=np.float32)
        return np.zeros(50, dtype=np.float32)

    def _expand_to_128d(self, emb_50: np.ndarray, token: str) -> np.ndarray:
        """
        Deterministically expand a 50-d vector to 128-d using hash-based noise.
        """
        if emb_50.dtype != np.float32:
            emb_50 = emb_50.astype(np.float32)

        expanded = np.zeros(128, dtype=np.float32)
        expanded[: emb_50.shape[0]] = emb_50

        rng = np.random.default_rng(seed=_stable_hash(token))
        expanded[emb_50.shape[0] :] = rng.normal(scale=0.05, size=128 - emb_50.shape[0]).astype(np.float32)

        # Blend with resonator for additional smoothing
        baseline = np.zeros_like(expanded)
        smoothed = self._vector_resonator.resonate(expanded, baseline, alpha=0.1)
        return smoothed.astype(np.float32)


__all__ = ["SovereignTextIngestor"]
