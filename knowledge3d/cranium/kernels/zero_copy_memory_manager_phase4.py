"""Ingestion-only helpers for the lightweight procedural content lane.

The underlying CUDA file provides real kernels, but the current WINE adapter
uses these helpers as a content-analysis boundary before proceduralization.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _content_signature(content_data: bytes) -> dict[str, Any]:
    payload = bytes(content_data or b"")
    digest = hashlib.sha256(payload).digest()
    span = max(1, len(payload))
    vertex_count = max(8, min(4096, span // 12))
    face_count = max(4, vertex_count // 3)
    cache_size = min(4096, max(64, vertex_count))
    stream_count = max(1, min(8, span // 1024 + 1))
    base = [digest[i] / 255.0 for i in range(16)]
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "cache_size": cache_size,
        "stream_count": stream_count,
        "vertices": [[base[i % len(base)], base[(i + 1) % len(base)], base[(i + 2) % len(base)]] for i in range(min(8, vertex_count))],
        "faces": [[i, (i + 1) % max(3, min(8, vertex_count)), (i + 2) % max(3, min(8, vertex_count))] for i in range(min(6, face_count))],
        "cached_vertices": [[base[i % len(base)], base[(i + 5) % len(base)], base[(i + 9) % len(base)]] for i in range(min(4, vertex_count))],
        "cached_faces": [[i, (i + 1) % 4, (i + 2) % 4] for i in range(min(4, face_count))],
        "vertex_streams": [
            [[base[(j + lane) % len(base)], base[(j + lane + 1) % len(base)], base[(j + lane + 2) % len(base)]] for j in range(4)]
            for lane in range(stream_count)
        ],
        "face_streams": [
            [[0, 1, 2], [1, 2, 3]] for _ in range(stream_count)
        ],
        "signature": hashlib.sha256(payload).hexdigest(),
    }


def launch_fine_grained_kernel(content_data: bytes) -> dict[str, Any]:
    return _content_signature(content_data)


def launch_persistent_kernel(content_data: bytes) -> dict[str, Any]:
    return _content_signature(content_data)


def launch_stream_optimized_kernel(content_data: bytes) -> dict[str, Any]:
    return _content_signature(content_data)


__all__ = [
    "launch_fine_grained_kernel",
    "launch_persistent_kernel",
    "launch_stream_optimized_kernel",
]
