"""
Sovereign RPN embedding engine.

Character trigram hashing, sparse embedding lookup, and sentence aggregation
without NumPy staging. Public embedding vectors remain lightweight sequence
objects so higher layers can still hand them to NumPy-backed code when needed,
but this module itself stays on the sovereign side of the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import pickle
from pathlib import Path
import random
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32

_EPS = 1.1920928955078125e-07


class Float32Vector:
    """Small float32-compatible 1D vector without NumPy ownership."""

    def __init__(self, values: Iterable[float] = (), *, expected_dim: int | None = None):
        self._values = tuple(float(value) for value in values)
        if expected_dim is not None and len(self._values) != int(expected_dim):
            raise ValueError(f"Expected {expected_dim} values, received {len(self._values)}")

    @property
    def shape(self) -> tuple[int]:
        return (len(self._values),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._values)

    @property
    def flat(self) -> list[float]:
        return list(self._values)

    def astype(self, _dtype=None) -> "Float32Vector":
        return self

    def flatten(self) -> "Float32Vector":
        return self

    def copy(self) -> "Float32Vector":
        return Float32Vector(self._values)

    def tolist(self) -> list[float]:
        return list(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __repr__(self) -> str:
        return f"Float32Vector(shape={self.shape})"


def _stable_hash(payload: str) -> int:
    """Stable 32-bit hash of the payload."""
    digest = hashlib.md5(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _extract_trigrams(token: str) -> List[str]:
    """Slice `token` into overlapping character trigrams."""
    token = token.strip()
    if not token:
        return []
    if len(token) < 3:
        token = token.ljust(3, "_")
    return [token[i : i + 3] for i in range(len(token) - 2)]


def _coerce_vector(values: object, embedding_dim: int) -> Float32Vector:
    if isinstance(values, Float32Vector):
        if values.size != embedding_dim:
            raise ValueError(f"Expected {embedding_dim} values, received {values.size}")
        return values
    tensor = HostTensorF32.from_array_like(values)
    flat = tensor.to_flat_list()
    if len(flat) != embedding_dim:
        raise ValueError(f"Expected {embedding_dim} values, received {len(flat)}")
    return Float32Vector(flat, expected_dim=embedding_dim)


def _zero_vector(embedding_dim: int) -> Float32Vector:
    return Float32Vector((0.0 for _ in range(int(embedding_dim))), expected_dim=int(embedding_dim))


def _normalize(values: Sequence[float], embedding_dim: int) -> Float32Vector:
    flat = [float(value) for value in values]
    if len(flat) != int(embedding_dim):
        raise ValueError(f"Expected {embedding_dim} values, received {len(flat)}")
    norm = math.sqrt(sum(value * value for value in flat))
    if norm <= _EPS:
        return _zero_vector(embedding_dim)
    return Float32Vector((value / norm for value in flat), expected_dim=embedding_dim)


def _mean_vectors(vectors: Sequence[Sequence[float]], embedding_dim: int) -> Float32Vector:
    if not vectors:
        return _zero_vector(embedding_dim)
    accum = [0.0] * int(embedding_dim)
    for vector in vectors:
        coerced = _coerce_vector(vector, embedding_dim)
        for idx, value in enumerate(coerced):
            accum[idx] += float(value)
    scale = 1.0 / float(len(vectors))
    return _normalize((value * scale for value in accum), embedding_dim)


def _positional_weighted_vectors(vectors: Sequence[Sequence[float]], embedding_dim: int) -> Float32Vector:
    if not vectors:
        return _zero_vector(embedding_dim)
    accum = [0.0] * int(embedding_dim)
    for index, vector in enumerate(vectors):
        weight = 0.4 * (0.6 ** float(index))
        coerced = _coerce_vector(vector, embedding_dim)
        for dim, value in enumerate(coerced):
            accum[dim] += weight * float(value)
    return _normalize(accum, embedding_dim)


def _vector_table(vectors: Sequence[Sequence[float]], embedding_dim: int) -> HostTensorF32:
    if not vectors:
        return HostTensorF32.zeros(0, embedding_dim)
    rows = [_coerce_vector(vector, embedding_dim).tolist() for vector in vectors]
    return HostTensorF32.from_array_like(rows, rows=len(rows), cols=embedding_dim)


def _float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


@dataclass
class RPNEmbeddingEngine:
    """
    Sovereign embedding generator built on RPN-friendly operations.

    Each unique trigram maps to an embedding vector. The table is sparse on the
    host and can be mirrored into the GPU trigram bridge on demand.
    """

    embedding_dim: int = 128
    dtype: str = "float32"
    _embeddings: MutableMapping[int, object] = field(default_factory=dict, init=False, repr=False)
    vocab_size: int = field(default=0, init=False)
    hit_count: int = field(default=0, init=False)
    miss_count: int = field(default=0, init=False)
    _pending_consolidation: bool = field(default=False, init=False, repr=False)
    _last_consolidated_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._hash_to_index: Dict[int, int] = {}
        self._embedding_list: List[Float32Vector] = []
        self._gpu_bridge = None
        self._gpu_table_dirty = True

    def hash_trigram(self, trigram: str) -> int:
        return _stable_hash(trigram)

    # ------------------------------------------------------------------ #
    # Embedding lookups
    # ------------------------------------------------------------------ #
    def _initialise_embedding(self, trigram_hash: int) -> Float32Vector:
        rng = random.Random(int(trigram_hash))
        std = math.sqrt(2.0 / (self.embedding_dim + self.embedding_dim))
        vec = _normalize(
            (rng.gauss(0.0, std) for _ in range(self.embedding_dim)),
            self.embedding_dim,
        )

        index = len(self._embedding_list)
        self._hash_to_index[trigram_hash] = index
        self._embedding_list.append(vec)
        self._embeddings[trigram_hash] = vec
        self.vocab_size = len(self._embedding_list)
        self.miss_count += 1
        self.mark_unconsolidated()
        self._gpu_table_dirty = True
        return vec

    def embed_lookup(self, trigram_hash: int) -> Float32Vector:
        if trigram_hash not in self._hash_to_index:
            return self._initialise_embedding(trigram_hash)

        self.hit_count += 1
        index = self._hash_to_index[trigram_hash]
        vec = _coerce_vector(self._embedding_list[index], self.embedding_dim)
        self._embedding_list[index] = vec
        self._embeddings[trigram_hash] = vec
        return vec

    # ------------------------------------------------------------------ #
    # Word / sentence embedding
    # ------------------------------------------------------------------ #
    def embed_trigrams(self, trigrams: Sequence[str]) -> Float32Vector:
        if not trigrams:
            return _zero_vector(self.embedding_dim)
        embeddings = [self.embed_lookup(self.hash_trigram(tg)) for tg in trigrams]
        return _mean_vectors(embeddings, self.embedding_dim)

    def embed_word(self, word: str) -> Float32Vector:
        return self.embed_trigrams(_extract_trigrams(word.lower()))

    def embed_sentence(self, sentence: str) -> Float32Vector:
        tokens = [token for token in sentence.strip().split() if token]
        if not tokens:
            return _zero_vector(self.embedding_dim)
        return _positional_weighted_vectors([self.embed_word(token) for token in tokens], self.embedding_dim)

    # ------------------------------------------------------------------ #
    # GPU bridge integration
    # ------------------------------------------------------------------ #
    def attach_gpu_bridge(self, bridge) -> None:
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

    def get_embedding_table(self) -> HostTensorF32:
        return _vector_table(self._embedding_list, self.embedding_dim)

    def embed_word_gpu(self, word: str) -> Float32Vector:
        if self._gpu_bridge is None:
            raise RuntimeError(
                "GPU trigram bridge not initialized. "
                "RPN embeddings require GPU sovereignty - no CPU fallback. "
                "Call attach_gpu_bridge() before using embed_word_gpu()."
            )
        trigrams = _extract_trigrams(word.lower())
        if not trigrams:
            return _zero_vector(self.embedding_dim)
        indices = self._ensure_trigram_indices(trigrams)
        self._sync_gpu_table()
        return _coerce_vector(self._gpu_bridge.embed_indices(indices, return_cpu=True), self.embedding_dim)

    def embed_sentence_gpu(self, sentence: str) -> Float32Vector:
        if self._gpu_bridge is None:
            raise RuntimeError(
                "GPU trigram bridge not initialized. "
                "RPN embeddings require GPU sovereignty - no CPU fallback. "
                "Call attach_gpu_bridge() before using embed_sentence_gpu()."
            )
        tokens = [token for token in sentence.strip().split() if token]
        if not tokens:
            return _zero_vector(self.embedding_dim)
        embeddings = [self.embed_word_gpu(token) for token in tokens]
        return _positional_weighted_vectors(embeddings, self.embedding_dim)

    def embed_sentences_gpu(self, sentences: Sequence[str]) -> List[Float32Vector]:
        if self._gpu_bridge is None:
            raise RuntimeError(
                "GPU trigram bridge not initialized. "
                "RPN embeddings require GPU sovereignty - no CPU fallback. "
                "Call attach_gpu_bridge() before using embed_sentences_gpu()."
            )

        token_rows: List[List[str]] = []
        unique_tokens: Dict[str, None] = {}
        for sentence in sentences:
            tokens = [token for token in str(sentence).strip().split() if token]
            token_rows.append(tokens)
            for token in tokens:
                unique_tokens.setdefault(token, None)

        if not token_rows:
            return []

        token_cache: Dict[str, Float32Vector] = {}
        for token in unique_tokens:
            token_cache[token] = self.embed_word_gpu(token)

        outputs: List[Float32Vector] = []
        for tokens in token_rows:
            if not tokens:
                outputs.append(_zero_vector(self.embedding_dim))
                continue
            outputs.append(_positional_weighted_vectors([token_cache[token] for token in tokens], self.embedding_dim))
        return outputs

    def extract_trigram_indices(self, text: str) -> List[int]:
        trigrams = _extract_trigrams(text.lower())
        if not trigrams:
            return []
        return self._ensure_trigram_indices(trigrams)

    def embed_tokens(self, tokens: Sequence[str]) -> Float32Vector:
        if not tokens:
            return _zero_vector(self.embedding_dim)
        return _positional_weighted_vectors([self.embed_word(token) for token in tokens], self.embedding_dim)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_embeddings(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_dim": self.embedding_dim,
            "embeddings": {int(key): _coerce_vector(value, self.embedding_dim).tolist() for key, value in self._embeddings.items()},
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
            int(key): _coerce_vector(value, self.embedding_dim)
            for key, value in embeddings.items()
        }
        mapping = payload.get("hash_to_index")
        if isinstance(mapping, dict):
            self._hash_to_index = {int(key): int(value) for key, value in mapping.items()}
        else:
            self._hash_to_index = {}
            for idx, key in enumerate(sorted(self._embeddings.keys())):
                self._hash_to_index[key] = idx

        ordered = sorted(self._hash_to_index.items(), key=lambda item: item[1])
        self._embedding_list = [_coerce_vector(self._embeddings[hash_val], self.embedding_dim) for hash_val, _ in ordered]

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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embeddings": [vector.tolist() for vector in self._embedding_list],
            "hashes": [hash_val for hash_val, _ in sorted(self._hash_to_index.items(), key=lambda item: item[1])],
            "embedding_dim": self.embedding_dim,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load_embedding_table(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding table not found: {path}")
        with path.open("rb") as handle:
            payload: Mapping[str, object] = pickle.load(handle)

        embeddings = payload.get("embeddings")
        hashes = payload.get("hashes")
        if not isinstance(embeddings, list) or not isinstance(hashes, list):
            raise ValueError("Malformed embedding table payload.")

        self.embedding_dim = int(payload.get("embedding_dim", self.embedding_dim))
        if len(embeddings) != len(hashes):
            raise ValueError("Hash and embedding table size mismatch.")

        self._embedding_list = [_coerce_vector(row, self.embedding_dim) for row in embeddings]
        self._hash_to_index = {int(hash_val): idx for idx, hash_val in enumerate(hashes)}
        self._embeddings = {
            int(hash_val): self._embedding_list[idx]
            for idx, hash_val in enumerate(hashes)
        }
        self.vocab_size = len(self._embedding_list)
        self._gpu_table_dirty = True
        self._sync_gpu_table()

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
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
    def embeddings(self) -> MutableMapping[int, object]:
        return self._embeddings

    @property
    def pending_consolidation(self) -> bool:
        return self._pending_consolidation

    @property
    def last_consolidated_at(self) -> float | None:
        return self._last_consolidated_at

    def mark_unconsolidated(self) -> None:
        self._pending_consolidation = True

    def mark_consolidated(self) -> None:
        self._pending_consolidation = False
        self._last_consolidated_at = time.time()


__all__ = ["Float32Vector", "RPNEmbeddingEngine"]
