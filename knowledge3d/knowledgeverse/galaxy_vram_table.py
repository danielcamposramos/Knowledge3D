"""VRAM-resident Galaxy star table for the sovereign hot path."""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table


STAR_RECORD_BYTES = 256
STAR_EMBEDDING_OFFSET = 0
STAR_GALAXY_ID_OFFSET = 128
STAR_TYPE_OFFSET = 132
STAR_SELECTION_ROLE_OFFSET = 136
STAR_LAYER_ID_OFFSET = 140
STAR_FLAGS_OFFSET = 144
STAR_ANSWER_ELIGIBLE_OFFSET = 148
STAR_SEMANTIC_POLARITY_OFFSET = 152
STAR_SEMANTIC_FOCUS_OFFSET = 156
STAR_SEMANTIC_MASS_OFFSET = 160
STAR_ATTRACTIVE_PRIOR_OFFSET = 164
STAR_REPULSIVE_PRIOR_OFFSET = 168
STAR_ROUTE_POLICY_OFFSET = 172
STAR_STAR_HASH_OFFSET = 176
STAR_ROUTER_REF_COUNT_OFFSET = 184
STAR_ROUTER_REFS_OFFSET = 188
STAR_EXECUTOR_REF_COUNT_OFFSET = 196
STAR_EXECUTOR_REFS_OFFSET = 200
STAR_VALIDATOR_REF_COUNT_OFFSET = 208
STAR_VALIDATOR_REFS_OFFSET = 212
STAR_ANTI_PATTERN_REF_COUNT_OFFSET = 220
STAR_ANTI_PATTERN_REFS_OFFSET = 224
STAR_POSITION_OFFSET = 232
STAR_VELOCITY_OFFSET = 244

STAR_FLAG_ACTIVE = 0x01
STAR_FLAG_LEARNABLE = 0x02
STAR_NULL_REF = 0xFFFFFFFF

ROLE_UNKNOWN = 0
ROLE_ROUTER = 1
ROLE_EXECUTOR = 2
ROLE_VALIDATOR = 3
ROLE_ANSWER = 4
ROLE_ANTI_PATTERN = 5

ROLE_IDS = {
    "unknown": ROLE_UNKNOWN,
    "router": ROLE_ROUTER,
    "executor": ROLE_EXECUTOR,
    "validator": ROLE_VALIDATOR,
    "answer": ROLE_ANSWER,
    "anti_pattern": ROLE_ANTI_PATTERN,
}
ROLE_NAMES = {value: key for key, value in ROLE_IDS.items()}

ROLE_REF_LIMIT = 2
ROLE_KEYS = (
    "router_refs",
    "executor_refs",
    "validator_refs",
    "anti_pattern_refs",
)

ROUTE_POLICY_DECOMPOSE_ON_FAIL = 0x01
ROUTE_POLICY_REQUIRES_EXECUTOR = 0x02
ROUTE_POLICY_REQUIRES_VALIDATOR = 0x04
ROUTE_POLICY_ANSWER_GATE = 0x08


def encode_route_policy(
    *,
    decompose_on_fail: bool = False,
    requires_executor: bool = False,
    requires_validator: bool = False,
    answer_gate: bool = False,
    branch_topk: int = 0,
) -> int:
    flags = 0
    if decompose_on_fail:
        flags |= ROUTE_POLICY_DECOMPOSE_ON_FAIL
    if requires_executor:
        flags |= ROUTE_POLICY_REQUIRES_EXECUTOR
    if requires_validator:
        flags |= ROUTE_POLICY_REQUIRES_VALIDATOR
    if answer_gate:
        flags |= ROUTE_POLICY_ANSWER_GATE
    topk = max(0, min(255, int(branch_topk or 0)))
    return int(flags | (topk << 8))


def decode_route_policy(route_policy_id: int) -> dict[str, int | bool]:
    value = int(route_policy_id or 0)
    return {
        "decompose_on_fail": bool(value & ROUTE_POLICY_DECOMPOSE_ON_FAIL),
        "requires_executor": bool(value & ROUTE_POLICY_REQUIRES_EXECUTOR),
        "requires_validator": bool(value & ROUTE_POLICY_REQUIRES_VALIDATOR),
        "answer_gate": bool(value & ROUTE_POLICY_ANSWER_GATE),
        "branch_topk": int((value >> 8) & 0xFF),
    }


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value)


def _role_id(value: Any) -> int:
    if isinstance(value, str):
        return int(ROLE_IDS.get(str(value).strip().lower(), ROLE_UNKNOWN))
    try:
        numeric = int(value)
    except Exception:
        return ROLE_UNKNOWN
    return numeric if numeric in ROLE_NAMES else ROLE_UNKNOWN


def _ref_slots(star: dict[str, Any], key: str) -> tuple[int, list[int]]:
    refs = [int(value) for value in list(star.get(key) or []) if int(value) != STAR_NULL_REF][:ROLE_REF_LIMIT]
    refs.extend([STAR_NULL_REF] * (ROLE_REF_LIMIT - len(refs)))
    return min(ROLE_REF_LIMIT, len(list(star.get(key) or []))), refs[:ROLE_REF_LIMIT]


def _embedding32(values: list[float] | tuple[float, ...] | Any) -> list[float]:
    row = [float(value) for value in list(values or [])[:32]]
    if len(row) < 32:
        row.extend([0.0] * (32 - len(row)))
    row = row[:32]
    norm = sum(value * value for value in row) ** 0.5
    if norm > 1.0e-6:
        row = [value / norm for value in row]
    return row[:32]


def _vector3(values: list[float] | tuple[float, ...] | Any) -> list[float]:
    row = [float(value) for value in list(values or [])[:3]]
    if len(row) < 3:
        row.extend([0.0] * (3 - len(row)))
    return row[:3]


def _role_refs_for_embedding(star: dict[str, Any]) -> list[int]:
    refs: list[int] = []
    for key in ("router_refs", "executor_refs", "validator_refs", "anti_pattern_refs"):
        for raw in list(star.get(key) or []):
            value = int(raw)
            if value != STAR_NULL_REF and value not in refs:
                refs.append(value)
    return refs[:16]


def compose_star_embedding(stars: list[dict[str, Any]], star_index: int, dim: int = 32) -> list[float]:
    if star_index < 0 or star_index >= len(stars):
        return [0.0] * dim
    star = stars[star_index]
    output = [float(value) for value in list(star.get("embedding") or [])[:dim]]
    if len(output) < dim:
        output.extend([0.0] * (dim - len(output)))
    refs = _role_refs_for_embedding(star)
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
        self.ref_capacity = 1
        self.ref_indices_ptr = None
        self.router_offsets_ptr = None
        self.router_counts_ptr = None
        self.executor_offsets_ptr = None
        self.executor_counts_ptr = None
        self.validator_offsets_ptr = None
        self.validator_counts_ptr = None
        self.anti_pattern_offsets_ptr = None
        self.anti_pattern_counts_ptr = None
        self._host_stars: list[dict[str, Any]] = []
        self._allocate(self.max_stars)
        self._allocate_role_buffers(self.max_stars)
        self._allocate_ref_indices(self.ref_capacity)

    @staticmethod
    def _read_u32_array(ptr, count: int) -> list[int]:
        total = max(0, int(count))
        if ptr is None or total <= 0:
            return []
        payload = bytearray(total * 4)
        loader.memcpy_dtoh(_bytes_ptr(payload), ptr, len(payload))
        return list(struct.unpack_from(f"<{total}I", payload, 0))

    def close(self) -> None:
        for attr in (
            "gpu_ptr",
            "ref_indices_ptr",
            "router_offsets_ptr",
            "router_counts_ptr",
            "executor_offsets_ptr",
            "executor_counts_ptr",
            "validator_offsets_ptr",
            "validator_counts_ptr",
            "anti_pattern_offsets_ptr",
            "anti_pattern_counts_ptr",
        ):
            ptr = getattr(self, attr, None)
            if ptr:
                loader.gpu_free(ptr)
                setattr(self, attr, None)

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

    def _allocate_role_buffers(self, max_stars: int) -> None:
        buffer_bytes = max(1, int(max_stars)) * 4
        for attr in (
            "router_offsets_ptr",
            "router_counts_ptr",
            "executor_offsets_ptr",
            "executor_counts_ptr",
            "validator_offsets_ptr",
            "validator_counts_ptr",
            "anti_pattern_offsets_ptr",
            "anti_pattern_counts_ptr",
        ):
            old_ptr = getattr(self, attr, None)
            new_ptr = loader.gpu_malloc(buffer_bytes)
            zero_payload = bytearray(buffer_bytes)
            loader.memcpy_htod(new_ptr, _bytes_ptr(zero_payload), len(zero_payload))
            setattr(self, attr, new_ptr)
            if old_ptr:
                loader.gpu_free(old_ptr)

    def _allocate_ref_indices(self, capacity: int) -> None:
        self.ref_capacity = max(1, int(capacity))
        new_ptr = loader.gpu_malloc(self.ref_capacity * 4)
        zero_payload = bytearray(self.ref_capacity * 4)
        loader.memcpy_htod(new_ptr, _bytes_ptr(zero_payload), len(zero_payload))
        if getattr(self, "ref_indices_ptr", None):
            loader.gpu_free(self.ref_indices_ptr)
        self.ref_indices_ptr = new_ptr

    def _ensure_capacity(self, required_stars: int) -> None:
        required = max(1, int(required_stars))
        if required <= self.max_stars:
            return
        old_ptr = self.gpu_ptr
        self._allocate(required)
        self._allocate_role_buffers(required)
        if old_ptr:
            loader.gpu_free(old_ptr)

    def _ensure_ref_capacity(self, required_refs: int) -> None:
        required = max(1, int(required_refs))
        if required <= self.ref_capacity:
            return
        self._allocate_ref_indices(required)

    def prepare_gpu_build(self, *, star_count: int, ref_capacity: int) -> None:
        required_stars = max(1, int(star_count))
        required_refs = max(1, int(ref_capacity))
        self._ensure_capacity(required_stars)
        self._ensure_ref_capacity(required_refs)
        record_payload = bytearray(required_stars * STAR_RECORD_BYTES)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(record_payload), len(record_payload))
        zero_role_payload = bytearray(required_stars * 4)
        for ptr in (
            self.router_offsets_ptr,
            self.router_counts_ptr,
            self.executor_offsets_ptr,
            self.executor_counts_ptr,
            self.validator_offsets_ptr,
            self.validator_counts_ptr,
            self.anti_pattern_offsets_ptr,
            self.anti_pattern_counts_ptr,
        ):
            if ptr is not None:
                loader.memcpy_htod(ptr, _bytes_ptr(zero_role_payload), len(zero_role_payload))
        ref_payload = bytearray(required_refs * 4)
        for index in range(required_refs):
            struct.pack_into("<I", ref_payload, index * 4, STAR_NULL_REF)
        loader.memcpy_htod(self.ref_indices_ptr, _bytes_ptr(ref_payload), len(ref_payload))
        self.star_count = 0
        self._host_stars = []

    def load_stars(self, stars: list[dict[str, Any]]) -> int:
        self._ensure_capacity(len(stars))
        count = min(len(stars), self.max_stars)
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        role_offsets: dict[str, list[int]] = {key: [0] * self.max_stars for key in ROLE_KEYS}
        role_counts: dict[str, list[int]] = {key: [0] * self.max_stars for key in ROLE_KEYS}
        ref_indices: list[int] = []
        for index, star in enumerate(stars[:count]):
            self._pack_star(payload, index, star)
            for key in ROLE_KEYS:
                refs = [int(value) for value in list(star.get(key) or []) if int(value) != STAR_NULL_REF]
                role_offsets[key][index] = len(ref_indices)
                role_counts[key][index] = len(refs)
                ref_indices.extend(refs)
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))
        self._ensure_ref_capacity(len(ref_indices) or 1)
        ref_payload = bytearray(self.ref_capacity * 4)
        for index, ref_value in enumerate(ref_indices):
            struct.pack_into("<I", ref_payload, index * 4, int(ref_value) & 0xFFFFFFFF)
        loader.memcpy_htod(self.ref_indices_ptr, _bytes_ptr(ref_payload), len(ref_payload))
        self._load_role_array(self.router_offsets_ptr, role_offsets["router_refs"])
        self._load_role_array(self.router_counts_ptr, role_counts["router_refs"])
        self._load_role_array(self.executor_offsets_ptr, role_offsets["executor_refs"])
        self._load_role_array(self.executor_counts_ptr, role_counts["executor_refs"])
        self._load_role_array(self.validator_offsets_ptr, role_offsets["validator_refs"])
        self._load_role_array(self.validator_counts_ptr, role_counts["validator_refs"])
        self._load_role_array(self.anti_pattern_offsets_ptr, role_offsets["anti_pattern_refs"])
        self._load_role_array(self.anti_pattern_counts_ptr, role_counts["anti_pattern_refs"])
        self.star_count = count
        self._host_stars = [dict(star) for star in stars[:count]]
        return count

    def _load_role_array(self, ptr, values: list[int]) -> None:
        payload = bytearray(self.max_stars * 4)
        for index, value in enumerate(values[: self.max_stars]):
            struct.pack_into("<I", payload, index * 4, int(value) & 0xFFFFFFFF)
        loader.memcpy_htod(ptr, _bytes_ptr(payload), len(payload))

    def read_stars(self, count: int | None = None) -> list[dict[str, Any]]:
        read_count = self.star_count if count is None else min(max(0, int(count)), self.star_count)
        if self._host_stars:
            return [dict(self._host_stars[index]) for index in range(read_count)]
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.gpu_ptr, len(payload))
        return [self._unpack_star(payload, index) for index in range(read_count)]

    def export_artifact_bundle(self) -> dict[str, Any]:
        count = int(self.star_count)
        record_bytes = max(0, count) * STAR_RECORD_BYTES
        record_payload = bytearray(record_bytes)
        if count > 0:
            loader.memcpy_dtoh(_bytes_ptr(record_payload), self.gpu_ptr, len(record_payload))
        router_offsets = self._read_u32_array(self.router_offsets_ptr, count)
        router_counts = self._read_u32_array(self.router_counts_ptr, count)
        executor_offsets = self._read_u32_array(self.executor_offsets_ptr, count)
        executor_counts = self._read_u32_array(self.executor_counts_ptr, count)
        validator_offsets = self._read_u32_array(self.validator_offsets_ptr, count)
        validator_counts = self._read_u32_array(self.validator_counts_ptr, count)
        anti_offsets = self._read_u32_array(self.anti_pattern_offsets_ptr, count)
        anti_counts = self._read_u32_array(self.anti_pattern_counts_ptr, count)
        ref_count = 0
        for offsets, counts in (
            (router_offsets, router_counts),
            (executor_offsets, executor_counts),
            (validator_offsets, validator_counts),
            (anti_offsets, anti_counts),
        ):
            for offset, size in zip(offsets, counts):
                end = int(offset) + int(size)
                if end > ref_count:
                    ref_count = end
        ref_payload = bytearray(ref_count * 4)
        if ref_count > 0:
            loader.memcpy_dtoh(_bytes_ptr(ref_payload), self.ref_indices_ptr, len(ref_payload))
        return {
            "version": 1,
            "star_count": count,
            "record_bytes": bytes(record_payload),
            "router_offsets": router_offsets,
            "router_counts": router_counts,
            "executor_offsets": executor_offsets,
            "executor_counts": executor_counts,
            "validator_offsets": validator_offsets,
            "validator_counts": validator_counts,
            "anti_pattern_offsets": anti_offsets,
            "anti_pattern_counts": anti_counts,
            "ref_indices": list(struct.unpack_from(f"<{ref_count}I", ref_payload, 0)) if ref_count > 0 else [],
        }

    def load_artifact_bundle(self, bundle: dict[str, Any], *, host_stars: list[dict[str, Any]] | None = None) -> int:
        count = max(0, int(bundle.get("star_count", 0) or 0))
        self._ensure_capacity(count or 1)
        record_bytes = bytes(bundle.get("record_bytes") or b"")
        required_bytes = count * STAR_RECORD_BYTES
        if len(record_bytes) != required_bytes:
            raise ValueError("invalid_sovereign_star_record_bundle")
        payload = bytearray(self.max_stars * STAR_RECORD_BYTES)
        if record_bytes:
            payload[:required_bytes] = record_bytes
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))

        def _u32_list(key: str) -> list[int]:
            values = [int(value) for value in list(bundle.get(key) or [])]
            if len(values) < self.max_stars:
                values.extend([0] * (self.max_stars - len(values)))
            return values[: self.max_stars]

        router_offsets = _u32_list("router_offsets")
        router_counts = _u32_list("router_counts")
        executor_offsets = _u32_list("executor_offsets")
        executor_counts = _u32_list("executor_counts")
        validator_offsets = _u32_list("validator_offsets")
        validator_counts = _u32_list("validator_counts")
        anti_offsets = _u32_list("anti_pattern_offsets")
        anti_counts = _u32_list("anti_pattern_counts")

        ref_indices = [int(value) for value in list(bundle.get("ref_indices") or [])]
        self._ensure_ref_capacity(len(ref_indices) or 1)
        ref_payload = bytearray(self.ref_capacity * 4)
        for index, value in enumerate(ref_indices):
            struct.pack_into("<I", ref_payload, index * 4, int(value) & 0xFFFFFFFF)
        loader.memcpy_htod(self.ref_indices_ptr, _bytes_ptr(ref_payload), len(ref_payload))
        self._load_role_array(self.router_offsets_ptr, router_offsets)
        self._load_role_array(self.router_counts_ptr, router_counts)
        self._load_role_array(self.executor_offsets_ptr, executor_offsets)
        self._load_role_array(self.executor_counts_ptr, executor_counts)
        self._load_role_array(self.validator_offsets_ptr, validator_offsets)
        self._load_role_array(self.validator_counts_ptr, validator_counts)
        self._load_role_array(self.anti_pattern_offsets_ptr, anti_offsets)
        self._load_role_array(self.anti_pattern_counts_ptr, anti_counts)
        self.star_count = count
        self._host_stars = [dict(star) for star in list(host_stars or [])[:count]]
        return count

    def _pack_star(self, payload: bytearray, star_index: int, star: dict[str, Any]) -> None:
        base = int(star_index) * STAR_RECORD_BYTES
        embedding = _embedding32(star.get("embedding") or [])
        router_count, router_refs = _ref_slots(star, "router_refs")
        executor_count, executor_refs = _ref_slots(star, "executor_refs")
        validator_count, validator_refs = _ref_slots(star, "validator_refs")
        anti_count, anti_refs = _ref_slots(star, "anti_pattern_refs")
        galaxy_id = star.get("galaxy_id", _fnv1a32("reality"))
        if isinstance(galaxy_id, str):
            galaxy_id = _fnv1a32(galaxy_id)
        star_hash = int(star.get("star_hash") or _fnv1a64(str(star.get("id") or star.get("name") or star_index)))
        position = _vector3(star.get("semantic_position") or [])
        velocity = _vector3(star.get("semantic_velocity") or [])
        struct.pack_into("<32f", payload, base + STAR_EMBEDDING_OFFSET, *embedding)
        struct.pack_into("<I", payload, base + STAR_GALAXY_ID_OFFSET, int(galaxy_id) & 0xFFFFFFFF)
        struct.pack_into("<I", payload, base + STAR_TYPE_OFFSET, int(star.get("star_type", 0)))
        struct.pack_into("<I", payload, base + STAR_SELECTION_ROLE_OFFSET, _role_id(star.get("selection_role", 0)))
        struct.pack_into("<I", payload, base + STAR_LAYER_ID_OFFSET, int(star.get("layer_id", 0)))
        struct.pack_into("<I", payload, base + STAR_FLAGS_OFFSET, int(star.get("flags", STAR_FLAG_ACTIVE)))
        struct.pack_into("<I", payload, base + STAR_ANSWER_ELIGIBLE_OFFSET, 1 if star.get("answer_eligible") else 0)
        struct.pack_into("<i", payload, base + STAR_SEMANTIC_POLARITY_OFFSET, int(star.get("semantic_polarity", 0)))
        struct.pack_into("<f", payload, base + STAR_SEMANTIC_FOCUS_OFFSET, float(star.get("semantic_focus", 0.0)))
        struct.pack_into("<f", payload, base + STAR_SEMANTIC_MASS_OFFSET, float(star.get("semantic_mass", 1.0)))
        struct.pack_into("<f", payload, base + STAR_ATTRACTIVE_PRIOR_OFFSET, float(star.get("attractive_prior", 0.0)))
        struct.pack_into("<f", payload, base + STAR_REPULSIVE_PRIOR_OFFSET, float(star.get("repulsive_prior", 0.0)))
        struct.pack_into("<I", payload, base + STAR_ROUTE_POLICY_OFFSET, int(star.get("route_policy_id", 0)))
        struct.pack_into("<Q", payload, base + STAR_STAR_HASH_OFFSET, star_hash & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<I", payload, base + STAR_ROUTER_REF_COUNT_OFFSET, int(router_count))
        struct.pack_into("<2I", payload, base + STAR_ROUTER_REFS_OFFSET, *[int(value) & 0xFFFFFFFF for value in router_refs])
        struct.pack_into("<I", payload, base + STAR_EXECUTOR_REF_COUNT_OFFSET, int(executor_count))
        struct.pack_into("<2I", payload, base + STAR_EXECUTOR_REFS_OFFSET, *[int(value) & 0xFFFFFFFF for value in executor_refs])
        struct.pack_into("<I", payload, base + STAR_VALIDATOR_REF_COUNT_OFFSET, int(validator_count))
        struct.pack_into("<2I", payload, base + STAR_VALIDATOR_REFS_OFFSET, *[int(value) & 0xFFFFFFFF for value in validator_refs])
        struct.pack_into("<I", payload, base + STAR_ANTI_PATTERN_REF_COUNT_OFFSET, int(anti_count))
        struct.pack_into("<2I", payload, base + STAR_ANTI_PATTERN_REFS_OFFSET, *[int(value) & 0xFFFFFFFF for value in anti_refs])
        struct.pack_into("<3f", payload, base + STAR_POSITION_OFFSET, *position)
        struct.pack_into("<3f", payload, base + STAR_VELOCITY_OFFSET, *velocity)

    def _unpack_star(self, payload: bytearray, star_index: int) -> dict[str, Any]:
        base = int(star_index) * STAR_RECORD_BYTES
        embedding = list(struct.unpack_from("<32f", payload, base + STAR_EMBEDDING_OFFSET))
        galaxy_id = struct.unpack_from("<I", payload, base + STAR_GALAXY_ID_OFFSET)[0]
        star_type = struct.unpack_from("<I", payload, base + STAR_TYPE_OFFSET)[0]
        selection_role_id = struct.unpack_from("<I", payload, base + STAR_SELECTION_ROLE_OFFSET)[0]
        layer_id = struct.unpack_from("<I", payload, base + STAR_LAYER_ID_OFFSET)[0]
        flags = struct.unpack_from("<I", payload, base + STAR_FLAGS_OFFSET)[0]
        answer_eligible = struct.unpack_from("<I", payload, base + STAR_ANSWER_ELIGIBLE_OFFSET)[0]
        semantic_polarity = struct.unpack_from("<i", payload, base + STAR_SEMANTIC_POLARITY_OFFSET)[0]
        semantic_focus = struct.unpack_from("<f", payload, base + STAR_SEMANTIC_FOCUS_OFFSET)[0]
        semantic_mass = struct.unpack_from("<f", payload, base + STAR_SEMANTIC_MASS_OFFSET)[0]
        attractive_prior = struct.unpack_from("<f", payload, base + STAR_ATTRACTIVE_PRIOR_OFFSET)[0]
        repulsive_prior = struct.unpack_from("<f", payload, base + STAR_REPULSIVE_PRIOR_OFFSET)[0]
        route_policy_id = struct.unpack_from("<I", payload, base + STAR_ROUTE_POLICY_OFFSET)[0]
        star_hash = struct.unpack_from("<Q", payload, base + STAR_STAR_HASH_OFFSET)[0]
        router_count = min(ROLE_REF_LIMIT, struct.unpack_from("<I", payload, base + STAR_ROUTER_REF_COUNT_OFFSET)[0])
        router_refs_raw = list(struct.unpack_from("<2I", payload, base + STAR_ROUTER_REFS_OFFSET))
        executor_count = min(ROLE_REF_LIMIT, struct.unpack_from("<I", payload, base + STAR_EXECUTOR_REF_COUNT_OFFSET)[0])
        executor_refs_raw = list(struct.unpack_from("<2I", payload, base + STAR_EXECUTOR_REFS_OFFSET))
        validator_count = min(ROLE_REF_LIMIT, struct.unpack_from("<I", payload, base + STAR_VALIDATOR_REF_COUNT_OFFSET)[0])
        validator_refs_raw = list(struct.unpack_from("<2I", payload, base + STAR_VALIDATOR_REFS_OFFSET))
        anti_count = min(ROLE_REF_LIMIT, struct.unpack_from("<I", payload, base + STAR_ANTI_PATTERN_REF_COUNT_OFFSET)[0])
        anti_refs_raw = list(struct.unpack_from("<2I", payload, base + STAR_ANTI_PATTERN_REFS_OFFSET))
        semantic_position = list(struct.unpack_from("<3f", payload, base + STAR_POSITION_OFFSET))
        semantic_velocity = list(struct.unpack_from("<3f", payload, base + STAR_VELOCITY_OFFSET))
        router_refs = [int(value) for value in router_refs_raw[:router_count] if value != STAR_NULL_REF]
        executor_refs = [int(value) for value in executor_refs_raw[:executor_count] if value != STAR_NULL_REF]
        validator_refs = [int(value) for value in validator_refs_raw[:validator_count] if value != STAR_NULL_REF]
        anti_refs = [int(value) for value in anti_refs_raw[:anti_count] if value != STAR_NULL_REF]
        component_refs = [ref for ref in router_refs + executor_refs + validator_refs + anti_refs if ref != STAR_NULL_REF]
        return {
            "embedding": embedding,
            "galaxy_id": int(galaxy_id),
            "star_type": int(star_type),
            "selection_role": ROLE_NAMES.get(int(selection_role_id), "unknown"),
            "selection_role_id": int(selection_role_id),
            "layer_id": int(layer_id),
            "flags": int(flags),
            "answer_eligible": bool(answer_eligible),
            "semantic_polarity": int(semantic_polarity),
            "semantic_focus": float(semantic_focus),
            "semantic_mass": float(semantic_mass),
            "attractive_prior": float(attractive_prior),
            "repulsive_prior": float(repulsive_prior),
            "route_policy_id": int(route_policy_id),
            "route_policy": decode_route_policy(route_policy_id),
            "star_hash": int(star_hash),
            "router_refs": router_refs,
            "executor_refs": executor_refs,
            "validator_refs": validator_refs,
            "anti_pattern_refs": anti_refs,
            "component_refs": component_refs,
            "semantic_position": semantic_position,
            "semantic_velocity": semantic_velocity,
        }


__all__ = [
    "ROLE_ANTI_PATTERN",
    "ROLE_ANSWER",
    "ROLE_EXECUTOR",
    "ROLE_IDS",
    "ROLE_NAMES",
    "ROLE_ROUTER",
    "ROLE_UNKNOWN",
    "ROLE_VALIDATOR",
    "STAR_ANSWER_ELIGIBLE_OFFSET",
    "STAR_ATTRACTIVE_PRIOR_OFFSET",
    "STAR_EMBEDDING_OFFSET",
    "STAR_FLAG_ACTIVE",
    "STAR_FLAG_LEARNABLE",
    "STAR_GALAXY_ID_OFFSET",
    "STAR_LAYER_ID_OFFSET",
    "STAR_NULL_REF",
    "STAR_RECORD_BYTES",
    "STAR_ROUTE_POLICY_OFFSET",
    "STAR_REPULSIVE_PRIOR_OFFSET",
    "STAR_ROUTER_REF_COUNT_OFFSET",
    "STAR_ROUTER_REFS_OFFSET",
    "STAR_EXECUTOR_REF_COUNT_OFFSET",
    "STAR_EXECUTOR_REFS_OFFSET",
    "STAR_VALIDATOR_REF_COUNT_OFFSET",
    "STAR_VALIDATOR_REFS_OFFSET",
    "STAR_ANTI_PATTERN_REF_COUNT_OFFSET",
    "STAR_ANTI_PATTERN_REFS_OFFSET",
    "STAR_SELECTION_ROLE_OFFSET",
    "STAR_SEMANTIC_MASS_OFFSET",
    "STAR_SEMANTIC_POLARITY_OFFSET",
    "STAR_POSITION_OFFSET",
    "STAR_VELOCITY_OFFSET",
    "STAR_STAR_HASH_OFFSET",
    "GalaxyVRAMTable",
    "build_foundational_galaxy_table",
    "compose_star_embedding",
]
