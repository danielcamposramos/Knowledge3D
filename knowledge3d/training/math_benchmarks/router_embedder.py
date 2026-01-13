"""
Router Embedder: Galaxy-anchored embeddings with sovereign n-gram fallback.

This avoids hash-only embeddings by grounding tokens in WordGalaxy/MathGalaxy
metadata where possible, and falls back to character 3-gram vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Optional

from knowledge3d.cranium.math_galaxy import get_math_galaxy
from knowledge3d.cranium.word_galaxy import WordGalaxy, get_word_galaxy


_TOKEN_REGEX = re.compile(r"\\[A-Za-z]+|[A-Za-z]+(?:/[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def tokenize_math_text(text: str) -> List[str]:
    """Tokenize a math string with a lightweight regex split."""
    if not text:
        return []
    return [t for t in _TOKEN_REGEX.findall(text) if t and not t.isspace()]


def _trigrams(token: str) -> List[str]:
    if not token:
        return []
    if len(token) < 3:
        return [token]
    padded = f"^{token}$"
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


def _stable_index(digest: bytes, offset: int, dim: int) -> int:
    chunk = digest[offset : offset + 2]
    if len(chunk) < 2:
        chunk = (digest + b"\x00\x00")[offset : offset + 2]
    return int.from_bytes(chunk, "little", signed=False) % dim


def _trigram_vector(trigram: str, dim: int) -> List[float]:
    digest = hashlib.sha256(trigram.encode("utf-8")).digest()
    vec = [0.0] * dim
    # Add a few signed impulses per trigram to keep locality via shared n-grams.
    for i in range(4):
        idx = _stable_index(digest, i * 2, dim)
        sign = 1.0 if digest[8 + i] % 2 == 0 else -1.0
        vec[idx] += sign
    return vec


def _sum_vectors(vectors: Iterable[List[float]], dim: int) -> List[float]:
    total = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            total[i] += val
    return total


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def _token_seed(
    token: str,
    *,
    word_galaxy: WordGalaxy,
    math_galaxy,
) -> str:
    lower = token.lower()
    if token and len(token) == 1:
        symbol = math_galaxy.get_by_char(token)
        if symbol is not None:
            return f"{symbol.name}|{symbol.domain}|{symbol.latex}"
    word = word_galaxy.get(lower) or word_galaxy.get(token)
    if word is not None:
        return f"{word.word_id}|{word.domain}|{word.rpn_context or ''}"
    return lower


def embed_text(
    text: str,
    *,
    dim: int = 256,
    word_galaxy: Optional[WordGalaxy] = None,
    math_galaxy: Optional[object] = None,
) -> List[float]:
    """
    Embed text into a fixed-size vector using Galaxy anchors + n-gram fallback.

    This uses WordGalaxy/MathGalaxy metadata as seed strings when available;
    otherwise it falls back to character 3-gram bags for locality.
    """
    word_galaxy = word_galaxy or get_word_galaxy()
    math_galaxy = math_galaxy or get_math_galaxy()

    tokens = tokenize_math_text(text)
    if not tokens:
        return [0.0] * dim

    token_vectors: List[List[float]] = []
    for token in tokens:
        seed = _token_seed(token, word_galaxy=word_galaxy, math_galaxy=math_galaxy)
        trigrams = _trigrams(seed)
        gram_vectors = [_trigram_vector(g, dim) for g in trigrams]
        token_vec = _sum_vectors(gram_vectors, dim)
        token_vectors.append(token_vec)

    pooled = _sum_vectors(token_vectors, dim)
    pooled = [v / float(len(token_vectors)) for v in pooled]
    return _normalize(pooled)


def embed_semantic_tags(
    tags: Iterable[str],
    *,
    dim: int = 256,
    word_galaxy: Optional[WordGalaxy] = None,
    math_galaxy: Optional[object] = None,
) -> List[float]:
    text = " ".join(str(t or "") for t in tags if str(t or "").strip())
    return embed_text(text, dim=dim, word_galaxy=word_galaxy, math_galaxy=math_galaxy)


__all__ = ["embed_text", "embed_semantic_tags", "tokenize_math_text"]
