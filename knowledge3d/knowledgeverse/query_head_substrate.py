"""Bind-time GPU substrate for the unified sovereign query head."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.spatial_sovereign.frustum import (
    FrustumCuller,
    UInt32Vector,
    create_perspective_matrix,
    create_view_matrix,
    matmul_4x4,
)
from knowledge3d.cranium.spatial_sovereign.morton_octree import MortonOctreeSovereign


class DynamicLodDriverBridge:
    """Thin ctypes bridge for the dynamic LOD PTX kernel."""

    NODE_STRIDE_BYTES = 4096
    NODE_STRIDE_FLOATS = NODE_STRIDE_BYTES // 4
    EMBEDDING_DIM = 512
    MORTON_OFFSET_BYTES = 3084
    MORTON_OFFSET_U32 = MORTON_OFFSET_BYTES // 4

    def __init__(self) -> None:
        ptx_path = Path(__file__).resolve().parents[1] / "cranium" / "ptx" / "dynamic_lod_tune.ptx"
        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernel = loader.get_function(self._module, "dynamic_lod_tune")
        self._d_unified = None
        self._d_query = None
        self._d_saliency = None
        self._node_capacity = 0
        self._query_capacity = self.EMBEDDING_DIM
        self._host_saliency = HostTensorF32.zeros(0, 2)

    def bind_unified_buffer(self, host_buffer: object, node_count: int) -> None:
        byte_buffer, required_bytes = _byte_buffer_view(host_buffer)
        if self._d_unified is not None:
            loader.gpu_free(self._d_unified)
        self._d_unified = loader.gpu_malloc(required_bytes)
        loader.memcpy_htod(
            self._d_unified,
            ctypes.c_void_p(ctypes.addressof(byte_buffer)),
            required_bytes,
        )
        self._ensure_query_buffer()
        self._ensure_saliency_buffer(int(node_count))

    def tune(self, query_embedding16: Iterable[float], node_count: int, saliency_threshold: float = 0.62) -> HostTensorF32:
        self.tune_device(
            query_embedding16,
            node_count=node_count,
            saliency_threshold=saliency_threshold,
        )
        return self.read_saliency(int(node_count))

    def tune_device(
        self,
        query_embedding16: Iterable[float],
        node_count: int,
        saliency_threshold: float = 0.62,
    ) -> tuple[loader.CUdeviceptr, int]:
        if self._d_unified is None:
            raise RuntimeError("Dynamic LOD buffer not bound")
        query512 = self._project_embedding16_to512(query_embedding16)
        query_tensor = HostTensorF32.from_array_like(query512, rows=self.EMBEDDING_DIM, cols=1)
        loader.memcpy_htod(
            self._d_query,
            ctypes.c_void_p(query_tensor.data_ptr),
            query_tensor.nbytes,
        )
        self._ensure_saliency_buffer(int(node_count))
        block_size = 128
        grid_size = (int(node_count) + block_size - 1) // block_size
        loader.launch(
            self._kernel,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
            params=[
                self._d_unified,
                self._d_query,
                ctypes.c_uint32(int(node_count)),
                ctypes.c_float(float(saliency_threshold)),
                self._d_saliency,
            ],
        )
        loader.synchronize()
        return self._d_saliency, int(node_count)

    def read_saliency(self, node_count: int) -> HostTensorF32:
        output = self._host_saliency
        loader.memcpy_dtoh(
            ctypes.c_void_p(output.data_ptr),
            self._d_saliency,
            output.nbytes,
        )
        if int(node_count) < output.rows:
            rows = [output[row] for row in range(int(node_count))]
            return HostTensorF32.from_array_like(rows, rows=int(node_count), cols=2)
        return output.copy()

    def close(self) -> None:
        for attr in ("_d_unified", "_d_query", "_d_saliency"):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                try:
                    loader.gpu_free(ptr)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._node_capacity = 0
        self._host_saliency = HostTensorF32.zeros(0, 2)

    def _ensure_query_buffer(self) -> None:
        if self._d_query is not None:
            return
        self._d_query = loader.gpu_malloc(self.EMBEDDING_DIM * 4)

    def _ensure_saliency_buffer(self, node_count: int) -> None:
        if node_count <= self._node_capacity and self._d_saliency is not None:
            return
        if self._d_saliency is not None:
            loader.gpu_free(self._d_saliency)
        self._d_saliency = loader.gpu_malloc(int(node_count) * 2 * 4)
        self._node_capacity = int(node_count)
        self._host_saliency = HostTensorF32.zeros(int(node_count), 2)

    @classmethod
    def build_unified_host_buffer(
        cls,
        embeddings16: object,
        morton_levels: object,
    ) -> ctypes.Array:
        embedding_rows = [list(float(value) for value in row) for row in embeddings16]
        morton_values = [int(value) for value in morton_levels]
        node_count = int(len(embedding_rows))
        byte_buffer = (ctypes.c_uint8 * (node_count * cls.NODE_STRIDE_BYTES))()
        float_ptr = ctypes.cast(ctypes.addressof(byte_buffer), ctypes.POINTER(ctypes.c_float))
        u32_ptr = ctypes.cast(ctypes.addressof(byte_buffer), ctypes.POINTER(ctypes.c_uint32))
        for row_index, embedding in enumerate(embedding_rows):
            projected = cls._project_embedding16_to512(embedding)
            base = row_index * cls.NODE_STRIDE_FLOATS
            for col, value in enumerate(projected):
                float_ptr[base + col] = float(value)
            u32_ptr[base + cls.MORTON_OFFSET_U32] = int(morton_values[row_index])
        return byte_buffer

    @classmethod
    def _project_embedding16_to512(cls, embedding16: Iterable[float]) -> list[float]:
        values = list(float(value) for value in embedding16)
        if not values:
            return [0.0] * cls.EMBEDDING_DIM
        tiled = (values * ((cls.EMBEDDING_DIM + len(values) - 1) // len(values)))[: cls.EMBEDDING_DIM]
        return [float(value) for value in tiled]


@dataclass
class QueryHeadSubstrate:
    """GPU-resident spatial substrate for the composed query head."""

    signature: str
    positions: HostTensorF32
    galaxy_indexes: UInt32Vector
    morton_levels: UInt32Vector
    morton_octree: MortonOctreeSovereign
    frustum_culler: FrustumCuller
    dynamic_lod: DynamicLodDriverBridge

    @classmethod
    def build(cls, *, signature: str, catalog: list[dict]) -> "QueryHeadSubstrate":
        embeddings = [
            [float(value) for value in entry.get("embedding16", [0.0] * 16)]
            for entry in catalog
        ]
        galaxy_indexes = UInt32Vector(
            int(round(float(entry.get("gpu_galaxy_index", 0.0))))
            for entry in catalog
        )
        positions = HostTensorF32.from_array_like(
            [
                _semantic_position3(
                    embedding=row,
                    galaxy_index=float(entry.get("gpu_galaxy_index", 0.0)),
                    domain_hash=float(entry.get("domain_hash", 0.0)) if "domain_hash" in entry else 0.0,
                    subject_hash=float(entry.get("subject_hash", 0.0)) if "subject_hash" in entry else 0.0,
                )
                for row, entry in zip(embeddings, catalog)
            ]
        )
        morton_octree = MortonOctreeSovereign()
        morton_tree = morton_octree.build_tree(positions)
        morton_codes = morton_tree["codes"]
        morton_levels = UInt32Vector(_morton_level(code) for code in morton_codes)
        level_by_index = [0 for _ in catalog]
        sorted_indices = [int(value) for value in morton_tree["indices"]]
        for sorted_index, morton_level in zip(sorted_indices, morton_levels):
            level_by_index[int(sorted_index)] = int(morton_level)

        dynamic_lod = DynamicLodDriverBridge()
        unified_host = DynamicLodDriverBridge.build_unified_host_buffer(
            embeddings16=embeddings,
            morton_levels=level_by_index,
        )
        dynamic_lod.bind_unified_buffer(unified_host, len(catalog))

        frustum_culler = FrustumCuller(enable_profiling=False)
        substrate = cls(
            signature=str(signature),
            positions=positions,
            galaxy_indexes=galaxy_indexes,
            morton_levels=UInt32Vector(level_by_index),
            morton_octree=morton_octree,
            frustum_culler=frustum_culler,
            dynamic_lod=dynamic_lod,
        )
        substrate._device_allowed_galaxy_indexes = None
        substrate._device_frustum_flags_ptr = None
        substrate._device_frustum_count = 0
        substrate._device_saliency_ptr = None
        return substrate

    def morton_locate(
        self,
        *,
        query_embedding16: Iterable[float],
        allowed_galaxy_indexes: set[int] | None,
        max_results: int,
        morton_radius: int,
        euclidean_radius: float,
    ) -> UInt32Vector:
        query_position = _semantic_position3(list(query_embedding16), 0.0, 0.0, 0.0)
        result = self.morton_octree.query_radius(
            query_position,
            morton_radius=int(morton_radius),
            euclidean_radius=float(euclidean_radius),
            max_results=int(max_results),
        )
        if allowed_galaxy_indexes:
            allowed = {int(value) for value in allowed_galaxy_indexes}
            filtered = [
                int(index)
                for index in result
                if int(self.galaxy_indexes[int(index)]) in allowed
            ]
            return UInt32Vector(filtered)
        return result

    def morton_locate_device(
        self,
        *,
        query_embedding16: Iterable[float],
        allowed_galaxy_indexes: set[int] | None,
        max_results: int,
        morton_radius: int,
        euclidean_radius: float,
    ) -> tuple[int, int]:
        query_position = _semantic_position3(list(query_embedding16), 0.0, 0.0, 0.0)
        query_limit = int(max_results)
        if allowed_galaxy_indexes:
            query_limit = min(
                int(self.positions.rows),
                max(int(max_results), min(int(self.positions.rows), int(max_results) * 4)),
            )
        device_ptr, count = self.morton_octree.query_radius_device(
            query_position,
            morton_radius=int(morton_radius),
            euclidean_radius=float(euclidean_radius),
            max_results=query_limit,
        )
        self._device_allowed_galaxy_indexes = (
            {int(value) for value in allowed_galaxy_indexes}
            if allowed_galaxy_indexes
            else None
        )
        self._device_frustum_flags_ptr = None
        self._device_frustum_count = 0
        self._device_saliency_ptr = None
        return int(device_ptr.value), int(count)

    def frustum_visible(
        self,
        *,
        query_embedding16: Iterable[float],
        candidate_indices: object,
    ) -> UInt32Vector:
        indices = [int(value) for value in candidate_indices]
        if not indices:
            return UInt32Vector()
        query_position = _semantic_position3(list(query_embedding16), 0.0, 0.0, 0.0)
        target = _mean_selected_rows(self.positions, indices)
        if target is None:
            return UInt32Vector(indices)
        direction = [target[axis] - query_position[axis] for axis in range(3)]
        norm = math.sqrt(sum(float(value) * float(value) for value in direction))
        if norm <= 1e-6:
            return UInt32Vector(indices)
        direction_unit = [value / norm for value in direction]
        eye = [query_position[axis] - (direction_unit[axis] * 0.35) for axis in range(3)]
        up = (0.0, 1.0, 0.0)
        if abs(sum(direction_unit[axis] * up[axis] for axis in range(3))) > 0.95:
            up = (1.0, 0.0, 0.0)
        view = create_view_matrix(eye, target, up)
        proj = create_perspective_matrix(72.0, 1.0, 0.01, 8.0)
        view_proj = matmul_4x4(proj, view)
        visible = self.frustum_culler.cull_nodes(
            self.positions,
            indices,
            view_proj=view_proj,
            view=view,
        )
        return visible

    def frustum_visible_device(
        self,
        *,
        query_embedding16: Iterable[float],
        d_candidate_indices: int,
        candidate_count: int,
    ) -> tuple[int, int]:
        count = int(candidate_count)
        if count <= 0:
            self._device_frustum_flags_ptr = None
            self._device_frustum_count = 0
            return int(d_candidate_indices), 0
        seed_indices = self.morton_octree.read_indices(
            d_candidate_indices,
            count,
            limit=min(count, 32),
        ).tolist()
        query_position = _semantic_position3(list(query_embedding16), 0.0, 0.0, 0.0)
        target = _mean_selected_rows(self.positions, seed_indices)
        if target is None:
            self._device_frustum_flags_ptr = None
            self._device_frustum_count = count
            return int(d_candidate_indices), count
        direction = [target[axis] - query_position[axis] for axis in range(3)]
        norm = math.sqrt(sum(float(value) * float(value) for value in direction))
        if norm <= 1e-6:
            self._device_frustum_flags_ptr = None
            self._device_frustum_count = count
            return int(d_candidate_indices), count
        direction_unit = [value / norm for value in direction]
        eye = [query_position[axis] - (direction_unit[axis] * 0.35) for axis in range(3)]
        up = (0.0, 1.0, 0.0)
        if abs(sum(direction_unit[axis] * up[axis] for axis in range(3))) > 0.95:
            up = (1.0, 0.0, 0.0)
        view = create_view_matrix(eye, target, up)
        proj = create_perspective_matrix(72.0, 1.0, 0.01, 8.0)
        view_proj = matmul_4x4(proj, view)
        flags_ptr, retained = self.frustum_culler.cull_nodes_device(
            self.morton_octree.positions_device_ptr,
            d_candidate_indices,
            count,
            view_proj=view_proj,
            view=view,
        )
        self._device_frustum_flags_ptr = flags_ptr
        self._device_frustum_count = int(retained)
        return int(d_candidate_indices), int(retained)

    def lod_metrics(
        self,
        *,
        query_embedding16: Iterable[float],
        candidate_indices: object,
        saliency_threshold: float = 0.62,
    ) -> dict[int, tuple[float, int]]:
        indices = [int(value) for value in candidate_indices]
        if not indices:
            return {}
        saliency_map = self.dynamic_lod.tune(
            list(query_embedding16),
            len(self.positions),
            saliency_threshold=saliency_threshold,
        )
        metrics: dict[int, tuple[float, int]] = {}
        for node_index in indices:
            cosine = float(saliency_map[node_index, 0])
            lod_level = int(round(float(saliency_map[node_index, 1])))
            metrics[int(node_index)] = (cosine, lod_level)
        return metrics

    def lod_metrics_device(
        self,
        *,
        query_embedding16: Iterable[float],
        d_candidate_indices: int,
        candidate_count: int,
        saliency_threshold: float = 0.62,
    ) -> tuple[int, int]:
        if int(candidate_count) <= 0:
            self._device_saliency_ptr = None
            return int(d_candidate_indices), 0
        saliency_ptr, _ = self.dynamic_lod.tune_device(
            list(query_embedding16),
            len(self.positions),
            saliency_threshold=saliency_threshold,
        )
        self._device_saliency_ptr = saliency_ptr
        return int(d_candidate_indices), int(candidate_count)

    def read_top_candidates(
        self,
        *,
        d_indices: int,
        count: int,
        top_k: int,
        focus_level: int | None = None,
    ) -> tuple[list[int], dict[int, tuple[float, int]], dict[str, int]]:
        raw_indices = self.morton_octree.read_indices(d_indices, count).tolist()
        if self._device_frustum_flags_ptr is not None:
            flags = self.frustum_culler.read_flags(
                int(self._device_frustum_count or count),
                flags_ptr=self._device_frustum_flags_ptr,
            )
        else:
            flags = [1 for _ in raw_indices]
        visible_indices = [
            int(candidate_index)
            for candidate_index, flag in zip(raw_indices, flags)
            if int(flag) != 0
        ]
        allowed = (
            {int(value) for value in self._device_allowed_galaxy_indexes}
            if self._device_allowed_galaxy_indexes
            else None
        )
        if allowed:
            visible_indices = [
                int(candidate_index)
                for candidate_index in visible_indices
                if int(self.galaxy_indexes[int(candidate_index)]) in allowed
            ]
        visible_indices = list(dict.fromkeys(int(value) for value in visible_indices))
        metrics: dict[int, tuple[float, int]] = {}
        if self._device_saliency_ptr is not None:
            saliency_map = self.dynamic_lod.read_saliency(len(self.positions))
            for node_index in visible_indices:
                cosine = float(saliency_map[int(node_index), 0])
                lod_level = int(round(float(saliency_map[int(node_index), 1])))
                metrics[int(node_index)] = (cosine, lod_level)
        ranked = list(visible_indices)
        if metrics:
            ranked.sort(
                key=lambda node_index: (
                    1 if focus_level is not None and int(metrics[node_index][1]) <= int(focus_level) else 0,
                    float(metrics[node_index][0]),
                    -int(metrics[node_index][1]),
                ),
                reverse=True,
            )
        top_candidates = ranked[: max(0, int(top_k))]
        return (
            top_candidates,
            {int(node_index): metrics.get(int(node_index), (0.0, 0)) for node_index in top_candidates},
            {
                "raw_count": int(len(raw_indices)),
                "visible_count": int(len(visible_indices)),
                "top_count": int(len(top_candidates)),
            },
        )

    def close(self) -> None:
        try:
            self.dynamic_lod.close()
        finally:
            try:
                self.frustum_culler.close()
            finally:
                self.morton_octree.close()


def expand_embedding16_to128(embedding16: Iterable[float]) -> list[float]:
    values = list(float(value) for value in embedding16)
    if not values:
        return [0.0] * 128
    tiled = (values * ((128 + len(values) - 1) // len(values)))[:128]
    return [float(value) for value in tiled]


def halting_inputs(
    scores: Iterable[float],
    *,
    candidate_ids: Iterable[str] = (),
    minimum_threshold: float = 0.3,
    gap_threshold: float = 0.1,
    agreement_threshold: float = 3.0,
) -> tuple[HostTensorF32, UInt32Vector]:
    values = [float(score) for score in scores]
    if not values:
        return HostTensorF32.zeros(4, 1), UInt32Vector([1, 1, 1, 1])

    ids = [str(value).strip() for value in candidate_ids if str(value).strip()]
    grouped_scores: list[tuple[float, int]] = []
    agreement_count = 1.0
    if ids and len(ids) == len(values):
        grouped: dict[str, list[float]] = {}
        for candidate_id, score in zip(ids, values):
            grouped.setdefault(candidate_id, []).append(float(score))
        grouped_scores = sorted(
            (
                (max(group_scores) + (0.02 * float(len(group_scores) - 1)), len(group_scores))
                for group_scores in grouped.values()
            ),
            reverse=True,
        )
        agreement_count = float(max(size for _, size in grouped_scores))
    else:
        ordered = sorted(values, reverse=True)
        grouped_scores = [(float(score), 1) for score in ordered]

    top_score = float(grouped_scores[0][0]) if grouped_scores else 0.0
    second_score = float(grouped_scores[1][0]) if len(grouped_scores) > 1 else 0.0
    score_gap = top_score - second_score
    min_norm = top_score / max(float(minimum_threshold), 1e-6)
    gap_norm = score_gap / max(float(gap_threshold), 1e-6)
    agree_norm = agreement_count / max(float(agreement_threshold), 1e-6)
    converged_norm = min(min_norm, gap_norm, agree_norm)
    return (
        HostTensorF32.from_array_like([min_norm, gap_norm, agree_norm, converged_norm], rows=4, cols=1),
        UInt32Vector([1, 1, 1, 1]),
    )


def relative_halting_inputs(
    scores: Iterable[float],
    *,
    gap_threshold: float = 0.15,
) -> tuple[HostTensorF32, UInt32Vector]:
    values = sorted((float(score) for score in scores), reverse=True)
    if not values:
        return HostTensorF32.zeros(4, 1), UInt32Vector([1, 1, 1, 1])
    top_score = float(values[0])
    second_score = float(values[1]) if len(values) > 1 else 0.0
    score_gap = top_score - second_score
    gap_norm = score_gap / max(float(gap_threshold), 1e-6)
    return (
        HostTensorF32.from_array_like([1.0, gap_norm, 1.0, gap_norm], rows=4, cols=1),
        UInt32Vector([1, 1, 1, 1]),
    )


def _semantic_position3(
    embedding: object,
    galaxy_index: float,
    domain_hash: float,
    subject_hash: float,
) -> list[float]:
    values = list(float(value) for value in (embedding[:3] if hasattr(embedding, "__getitem__") else list(embedding)))
    while len(values) < 3:
        values.append(0.0)
    galaxy_bias = (float(galaxy_index) / 10.0) - 0.5
    x = values[0] + (0.08 * galaxy_bias)
    y = values[1] + (0.08 * (float(domain_hash) - 0.5))
    z = values[2] + (0.08 * (float(subject_hash) - 0.5))
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    if norm <= 1e-8:
        return [0.0, 0.0, 1.0]
    return [float(x / norm), float(y / norm), float(z / norm)]


def _byte_buffer_view(host_buffer: object) -> tuple[ctypes.Array, int]:
    if isinstance(host_buffer, HostTensorF32):
        raw = ctypes.string_at(host_buffer.data_ptr, host_buffer.nbytes)
        buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        return buf, len(raw)
    if isinstance(host_buffer, ctypes.Array):
        return host_buffer, ctypes.sizeof(host_buffer)
    payload = bytes(host_buffer)
    buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    return buf, len(payload)


def _mean_selected_rows(matrix: HostTensorF32, indices: Sequence[int]) -> list[float] | None:
    if not indices:
        return None
    accum = [0.0, 0.0, 0.0]
    for index in indices:
        row = matrix[int(index)]
        if len(row) < 3:
            continue
        for axis in range(3):
            accum[axis] += float(row[axis])
    count = float(len(indices))
    if count <= 0.0:
        return None
    mean = [value / count for value in accum]
    if not all(math.isfinite(value) for value in mean):
        return None
    return mean


def _morton_level(code: int) -> int:
    return max(0, min(int(bin(int(code)).count("1") // 3), 12))
