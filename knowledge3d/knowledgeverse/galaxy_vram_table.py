"""VRAM-resident Galaxy star table for sovereign ARC3 knowledge lookup."""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table


STAR_RECORD_BYTES = 160
STAR_EMBEDDING_OFFSET = 0
STAR_GALAXY_ID_OFFSET = 128
STAR_TYPE_OFFSET = 132
STAR_N_REFS_OFFSET = 136
STAR_REFS_OFFSET = 140
STAR_FLAGS_OFFSET = 156

STAR_FLAG_ACTIVE = 0x01
STAR_FLAG_LEARNABLE = 0x02
STAR_NULL_REF = 0xFFFFFFFF


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def compose_star_embedding(stars: list[dict[str, Any]], star_index: int, dim: int = 32) -> list[float]:
    if star_index < 0 or star_index >= len(stars):
        return [0.0] * dim
    star = stars[star_index]
    output = [float(value) for value in list(star.get("embedding") or [])[:dim]]
    if len(output) < dim:
        output.extend([0.0] * (dim - len(output)))
    refs = [int(value) for value in list(star.get("component_refs") or [])[:4] if int(value) != STAR_NULL_REF]
    if not refs:
        return output[:dim]
    base_weight = 0.60
    ref_weight = 0.40 / float(len(refs))
    output = [value * base_weight for value in output[:dim]]
    for ref_index in refs:
        if ref_index < 0 or ref_index >= len(stars):
            continue
        ref_embedding = [float(value) for value in list(stars[ref_index].get("embedding") or [])[:dim]]
        if len(ref_embedding) < dim:
            ref_embedding.extend([0.0] * (dim - len(ref_embedding)))
        for index in range(dim):
            output[index] += ref_weight * ref_embedding[index]
    norm = sum(value * value for value in output) ** 0.5
    if norm > 1.0e-6:
        output = [value / norm for value in output]
    return output[:dim]


class GalaxyVRAMTable:
    """VRAM star table for sovereign kernel-side knowledge access."""

    def __init__(self, max_stars: int = 300_000) -> None:
        self.max_stars = max(1, int(max_stars))
        self.star_count = 0
        self.gpu_ptr = None
        self._allocate(self.max_stars)

    def close(self) -> None:
        if getattr(self, "gpu_ptr", None):
            loader.gpu_free(self.gpu_ptr)
            self.gpu_ptr = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _allocate(self, max_stars: int) -> None:
        self.max_stars = max(1, int(max_stars))
        self.gpu_ptr = loader.gpu_malloc(self.max_stars * STAR_RECORD_BYTES)
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))

    def _ensure_capacity(self, required_stars: int) -> None:
        required = max(1, int(required_stars))
        if required <= self.max_stars:
            return
        old_ptr = self.gpu_ptr
        self._allocate(required)
        if old_ptr:
            loader.gpu_free(old_ptr)

    def load_stars(self, stars: list[dict[str, Any]]) -> int:
        self._ensure_capacity(len(stars))
        count = min(len(stars), self.max_stars)
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        for index, star in enumerate(stars[:count]):
            self._pack_star(payload, index, star)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))
        self.star_count = count
        return count

    def read_stars(self, count: int | None = None) -> list[dict[str, Any]]:
        read_count = self.star_count if count is None else min(max(0, int(count)), self.star_count)
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.gpu_ptr, len(payload))
        return [self._unpack_star(payload, index) for index in range(read_count)]

    def _pack_star(self, payload: bytearray, star_index: int, star: dict[str, Any]) -> None:
        base = int(star_index) * STAR_RECORD_BYTES
        embedding = [float(value) for value in list(star.get("embedding") or [])[:32]]
        if len(embedding) < 32:
            embedding.extend([0.0] * (32 - len(embedding)))
        refs = [int(value) for value in list(star.get("component_refs") or [])[:4]]
        refs.extend([STAR_NULL_REF] * (4 - len(refs)))
        galaxy_id = star.get("galaxy_id", _fnv1a32("reality"))
        if isinstance(galaxy_id, str):
            galaxy_id = _fnv1a32(galaxy_id)
        struct.pack_into("<32f", payload, base + STAR_EMBEDDING_OFFSET, *embedding)
        struct.pack_into("<I", payload, base + STAR_GALAXY_ID_OFFSET, int(galaxy_id) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + STAR_TYPE_OFFSET, int(star.get("star_type", 0)))
        struct.pack_into("<I", payload, base + STAR_N_REFS_OFFSET, min(4, len(list(star.get("component_refs") or []))))
        struct.pack_into("<4I", payload, base + STAR_REFS_OFFSET, *[int(value) & 0xFFFFFFFF for value in refs[:4]])
        struct.pack_into("<I", payload, base + STAR_FLAGS_OFFSET, int(star.get("flags", STAR_FLAG_ACTIVE)))

    def _unpack_star(self, payload: bytearray, star_index: int) -> dict[str, Any]:
        base = int(star_index) * STAR_RECORD_BYTES
        embedding = list(struct.unpack_from("<32f", payload, base + STAR_EMBEDDING_OFFSET))
        galaxy_id = struct.unpack_from("<I", payload, base + STAR_GALAXY_ID_OFFSET)[0]
        star_type = struct.unpack_from("<I", payload, base + STAR_TYPE_OFFSET)[0]
        n_refs = min(4, struct.unpack_from("<I", payload, base + STAR_N_REFS_OFFSET)[0])
        refs_raw = list(struct.unpack_from("<4I", payload, base + STAR_REFS_OFFSET))
        flags = struct.unpack_from("<I", payload, base + STAR_FLAGS_OFFSET)[0]
        component_refs = [int(value) for value in refs_raw[:n_refs] if value != STAR_NULL_REF]
        return {
            "embedding": embedding,
            "galaxy_id": int(galaxy_id),
            "star_type": int(star_type),
            "component_refs": component_refs,
            "flags": int(flags),
        }


__all__ = [
    "STAR_FLAG_ACTIVE",
    "STAR_FLAG_LEARNABLE",
    "STAR_NULL_REF",
    "STAR_RECORD_BYTES",
    "GalaxyVRAMTable",
    "build_foundational_galaxy_table",
    "compose_star_embedding",
]
