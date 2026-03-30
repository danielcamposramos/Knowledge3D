"""Sovereign text embedding via FNV-1a token hashing."""

from __future__ import annotations

EMBEDDING_DIM = 32
TOKEN_BUCKET_START = 8


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def embed_text_sovereign(text: str) -> list[float]:
    """Hash text into the same 32-float space used by Galaxy stars."""
    raw = str(text or "").strip().lower()
    tokens = [token for token in raw.split() if token]
    embedding = [0.0] * EMBEDDING_DIM
    if not raw:
        return embedding

    digit_count = sum(1 for char in raw if char.isdigit())
    alpha_count = sum(1 for char in raw if char.isalpha())
    embedding[0] = min(1.0, len(tokens) / 16.0)
    embedding[1] = min(1.0, digit_count / 8.0)
    embedding[2] = 1.0 if "?" in raw else 0.0
    embedding[3] = min(1.0, alpha_count / max(1.0, float(len(raw))))
    embedding[4] = 1.0 if any(token in {"why", "how", "what", "which", "who", "when", "where"} for token in tokens) else 0.0
    embedding[5] = 1.0 if any(token in {"move", "left", "right", "up", "down", "grid", "cell", "shape"} for token in tokens) else 0.0
    embedding[6] = 1.0 if any(token in {"sum", "solve", "equation", "number", "math", "calculate"} for token in tokens) else 0.0
    embedding[7] = 1.0 if any(token in {"physics", "biology", "chemistry", "history", "science"} for token in tokens) else 0.0

    bucket_count = EMBEDDING_DIM - TOKEN_BUCKET_START
    for token in tokens:
        token_hash = _fnv1a32(token)
        bucket = TOKEN_BUCKET_START + (token_hash % bucket_count)
        sign = 1.0 if ((token_hash >> 16) & 1) else -1.0
        magnitude = 1.0 + (0.25 * (((token_hash >> 8) & 0xFF) / 255.0))
        embedding[bucket] += sign * magnitude

    norm = sum(value * value for value in embedding) ** 0.5
    if norm > 1.0e-8:
        embedding = [value / norm for value in embedding]
    return embedding


__all__ = ["EMBEDDING_DIM", "embed_text_sovereign"]
