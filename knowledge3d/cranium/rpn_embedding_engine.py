"""
CPU prototype for the sovereign RPN embedding engine.

Implements character trigram hashing, sparse embedding lookup, and lightweight
sentence aggregation so we can eliminate the GloVe bootstrap during ingestion.

This module intentionally stays NumPy-only. The PTX version will reuse the same
opcode semantics once ported to the GPU execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import pickle
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

_EPS = np.finfo(np.float32).eps


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return an L2-normalised copy of `vec` (float32)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _stable_hash(payload: str) -> int:
    """
    Stable 32-bit hash of the payload.

    MD5 keeps the output consistent across Python versions / seeds without
    relying on the interpreter's `hash()` (which is salted per run).
    """
    digest = hashlib.md5(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")  # uint32 window


def _extract_trigrams(token: str) -> List[str]:
    """
    Slice `token` into overlapping character trigrams.

    Short tokens (<3 chars) are padded with '_' so every token yields at least
    one trigram.
    """
    token = token.strip()
    if not token:
        return []

    if len(token) < 3:
        token = token.ljust(3, "_")

    return [token[i : i + 3] for i in range(len(token) - 2)]


@dataclass
class RPNEmbeddingEngine:
    """
    Sovereign embedding generator built on RPN-friendly operations.

    Each unique trigram maps to an embedding vector. The CPU prototype keeps the
    mapping in a sparse dictionary keyed by stable trigram hash. Once the PTX
    path is ready, these opcodes translate directly to device operations.
    """

    embedding_dim: int = 128
    dtype: np.dtype = np.float32
    _embeddings: MutableMapping[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    vocab_size: int = field(default=0, init=False)
    hit_count: int = field(default=0, init=False)
    miss_count: int = field(default=0, init=False)
    _pending_consolidation: bool = field(default=False, init=False, repr=False)
    _last_consolidated_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._hash_to_index: Dict[int, int] = {}
        self._embedding_list: List[np.ndarray] = []
        self._gpu_bridge = None
        self._gpu_table_dirty = True

    def hash_trigram(self, trigram: str) -> int:
        """Hash a trigram to an unsigned 32-bit integer."""
        return _stable_hash(trigram)

    # ------------------------------------------------------------------ #
    # Embedding lookups
    # ------------------------------------------------------------------ #
    def _initialise_embedding(self, trigram_hash: int) -> np.ndarray:
        """
        Initialise a new embedding vector for `trigram_hash`.

        Uses a deterministic RNG seeded by the hash so first-look embeddings are
        stable across runs until persisted updates take over.
        """
        rng = np.random.default_rng(seed=trigram_hash)
        # Xavier/Glorot-style scaling
        std = np.sqrt(2.0 / (self.embedding_dim + self.embedding_dim))
        vec = rng.normal(loc=0.0, scale=std, size=self.embedding_dim).astype(self.dtype)
        vec = _normalize(vec)

        index = len(self._embedding_list)
        self._hash_to_index[trigram_hash] = index
        self._embedding_list.append(vec)
        self._embeddings[trigram_hash] = vec
        self.vocab_size = len(self._embedding_list)
        self.miss_count += 1
        self.mark_unconsolidated()
        self._gpu_table_dirty = True
        return vec

    def embed_lookup(self, trigram_hash: int) -> np.ndarray:
        """
        Fetch the embedding for `trigram_hash`, initialising it if needed.
        """
        if trigram_hash not in self._hash_to_index:
            return self._initialise_embedding(trigram_hash)

        self.hit_count += 1
        index = self._hash_to_index[trigram_hash]
        vec = self._embedding_list[index]
        self._embeddings[trigram_hash] = vec
        return vec

    # ------------------------------------------------------------------ #
    # Word / sentence embedding
    # ------------------------------------------------------------------ #
    def embed_trigrams(self, trigrams: Sequence[str]) -> np.ndarray:
        """
        Embed a pre-tokenised sequence of trigrams and return an L2-normalised
        average.
        """
        if not trigrams:
            return np.zeros(self.embedding_dim, dtype=self.dtype)

        embeddings = [self.embed_lookup(self.hash_trigram(tg)) for tg in trigrams]
        stacked = np.vstack(embeddings).astype(self.dtype)
        return _normalize(stacked.mean(axis=0))

    def embed_word(self, word: str) -> np.ndarray:
        """Embed a single word via character trigram averaging."""
        trigrams = _extract_trigrams(word.lower())
        return self.embed_trigrams(trigrams)

    def embed_sentence(self, sentence: str) -> np.ndarray:
        """
        Embed a sentence by averaging word embeddings and normalising the result.
        """
        tokens = [t for t in sentence.strip().split() if t]
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=self.dtype)

        word_embeddings = [self.embed_word(tok) for tok in tokens]
        stacked = np.vstack(word_embeddings).astype(self.dtype)
        return _normalize(stacked.mean(axis=0))

    # ------------------------------------------------------------------ #
    # GPU bridge integration
    # ------------------------------------------------------------------ #
    def attach_gpu_bridge(self, bridge) -> None:
        """Attach a GPU trigram embedding bridge (optional)."""
        self._gpu_bridge = bridge
        self._gpu_table_dirty = True
        self._sync_gpu_table()

    def has_gpu_bridge(self) -> bool:
        return self._gpu_bridge is not None

    def _sync_gpu_table(self) -> None:
        if self._gpu_bridge is None or not self._gpu_table_dirty:
            return
        table = self.get_embedding_table()
        self._gpu_bridge.upload_embedding_table(table)
        self._gpu_table_dirty = False

    def _ensure_trigram_indices(self, trigrams: Sequence[str]) -> List[int]:
        indices: List[int] = []
        for trigram in trigrams:
            trigram_hash = self.hash_trigram(trigram)
            self.embed_lookup(trigram_hash)
            indices.append(self._hash_to_index[trigram_hash])
        return indices

    def get_embedding_table(self) -> np.ndarray:
        if not self._embedding_list:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(self._embedding_list).astype(np.float32)

    def embed_word_gpu(self, word: str) -> np.ndarray:
        """Embed word using GPU trigram bridge (GPU-sovereign, no fallback)."""
        if self._gpu_bridge is None:
            raise RuntimeError(
                "GPU trigram bridge not initialized. "
                "RPN embeddings require GPU sovereignty - no CPU fallback. "
                "Call attach_gpu_bridge() before using embed_word_gpu()."
            )
        trigrams = _extract_trigrams(word.lower())
        if not trigrams:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        indices = self._ensure_trigram_indices(trigrams)
        self._sync_gpu_table()
        return self._gpu_bridge.embed_indices(indices, return_cpu=True)

    def embed_sentence_gpu(self, sentence: str) -> np.ndarray:
        """Embed sentence using GPU trigram bridge (GPU-sovereign, no fallback)."""
        if self._gpu_bridge is None:
            raise RuntimeError(
                "GPU trigram bridge not initialized. "
                "RPN embeddings require GPU sovereignty - no CPU fallback. "
                "Call attach_gpu_bridge() before using embed_sentence_gpu()."
            )
        tokens = [t for t in sentence.strip().split() if t]
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = [self.embed_word_gpu(token) for token in tokens]
        stacked = np.vstack(embeddings).astype(np.float32)
        return _normalize(stacked.mean(axis=0))

    def extract_trigram_indices(self, text: str) -> List[int]:
        trigrams = _extract_trigrams(text.lower())
        if not trigrams:
            return []
        return self._ensure_trigram_indices(trigrams)

    def embed_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        """
        Convenience helper: embed a sequence of tokens (already split words).
        """
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=self.dtype)
        word_embeddings = [self.embed_word(tok) for tok in tokens]
        stacked = np.vstack(word_embeddings).astype(self.dtype)
        return _normalize(stacked.mean(axis=0))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_embeddings(self, path: str | Path) -> None:
        """Serialise the learned embedding table."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_dim": self.embedding_dim,
            "embeddings": dict(self._embeddings),
            "hash_to_index": self._hash_to_index,
            "vocab_size": self.vocab_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "pending_consolidation": self._pending_consolidation,
            "last_consolidated_at": self._last_consolidated_at,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load_embeddings(self, path: str | Path) -> None:
        """Load a previously persisted embedding table."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding map not found: {path}")
        with path.open("rb") as handle:
            payload: Mapping[str, object] = pickle.load(handle)

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, dict):
            raise ValueError(f"Malformed embedding payload in {path}")

        self.embedding_dim = int(payload.get("embedding_dim", self.embedding_dim))
        self._embeddings = {
            int(k): np.asarray(v, dtype=self.dtype) for k, v in embeddings.items()
        }
        mapping = payload.get("hash_to_index")
        if isinstance(mapping, dict):
            self._hash_to_index = {int(k): int(v) for k, v in mapping.items()}
        else:
            # Rebuild mapping in deterministic order
            self._hash_to_index = {}
            for idx, key in enumerate(sorted(self._embeddings.keys())):
                self._hash_to_index[key] = idx

        ordered = sorted(self._hash_to_index.items(), key=lambda kv: kv[1])
        self._embedding_list = [self._embeddings[h].astype(np.float32) for h, _ in ordered]

        self.vocab_size = int(payload.get("vocab_size", len(self._embedding_list)))
        self.hit_count = int(payload.get("hit_count", 0))
        self.miss_count = int(payload.get("miss_count", 0))
        self._pending_consolidation = bool(payload.get("pending_consolidation", False))
        last_ts = payload.get("last_consolidated_at")
        self._last_consolidated_at = float(last_ts) if last_ts is not None else None
        self._gpu_table_dirty = True
        self._sync_gpu_table()

    # ------------------------------------------------------------------ #
    # Table persistence for GPU bridge
    # ------------------------------------------------------------------ #
    def save_embedding_table(self, path: str | Path) -> None:
        """Persist contiguous embedding table with hash ordering."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = self.get_embedding_table()
        hashes = np.array(
            [hash_val for hash_val, _ in sorted(self._hash_to_index.items(), key=lambda kv: kv[1])],
            dtype=np.uint32,
        )
        np.savez(path, embeddings=table, hashes=hashes)

    def load_embedding_table(self, path: str | Path) -> None:
        """Load embedding table + hash mapping from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding table not found: {path}")

        data = np.load(path)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        hashes = np.asarray(data["hashes"], dtype=np.uint32)

        if embeddings.shape[0] != hashes.shape[0]:
            raise ValueError("Hash and embedding table size mismatch.")

        self._embedding_list = [embeddings[i] for i in range(embeddings.shape[0])]
        self._hash_to_index = {int(h): idx for idx, h in enumerate(hashes.tolist())}
        self._embeddings = {
            int(h): embeddings[idx] for idx, h in enumerate(hashes.tolist())
        }
        self.vocab_size = embeddings.shape[0]
        self._gpu_table_dirty = True
        self._sync_gpu_table()

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        """Return basic usage statistics."""
        return {
            "vocab_size": self.vocab_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "pending_consolidation": int(self._pending_consolidation),
        }

    # ------------------------------------------------------------------ #
    # Consolidation helpers
    # ------------------------------------------------------------------ #
    @property
    def embeddings(self) -> MutableMapping[int, np.ndarray]:
        """Expose the raw embedding mapping (for consolidation routines)."""
        return self._embeddings

    @property
    def pending_consolidation(self) -> bool:
        """Whether new data has been ingested since the last consolidation run."""
        return self._pending_consolidation

    @property
    def last_consolidated_at(self) -> float | None:
        """Unix timestamp of the most recent consolidation event."""
        return self._last_consolidated_at

    def mark_unconsolidated(self) -> None:
        """Flag that fresh embeddings should be consolidated."""
        self._pending_consolidation = True

    def mark_consolidated(self) -> None:
        """Mark embeddings as consolidated."""
        self._pending_consolidation = False
        self._last_consolidated_at = time.time()


__all__ = ["RPNEmbeddingEngine"]
