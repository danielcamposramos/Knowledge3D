"""Bind-time GPU substrate for the unified sovereign query head."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.spatial_sovereign.frustum import (
    FrustumCuller,
    create_perspective_matrix,
    create_view_matrix,
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
        self._host_saliency = np.zeros((0, 2), dtype=np.float32)

    def bind_unified_buffer(self, host_buffer: np.ndarray, node_count: int) -> None:
        byte_buffer = np.ascontiguousarray(host_buffer, dtype=np.uint8).reshape(-1)
        required_bytes = int(byte_buffer.nbytes)
        if self._d_unified is not None:
            loader.gpu_free(self._d_unified)
        self._d_unified = loader.gpu_malloc(required_bytes)
        loader.memcpy_htod(
            self._d_unified,
            ctypes.c_void_p(byte_buffer.ctypes.data),
            required_bytes,
        )
        self._ensure_query_buffer()
        self._ensure_saliency_buffer(int(node_count))

    def tune(self, query_embedding16: Iterable[float], node_count: int, saliency_threshold: float = 0.62) -> np.ndarray:
        if self._d_unified is None:
            raise RuntimeError("Dynamic LOD buffer not bound")
        query512 = self._project_embedding16_to512(query_embedding16)
        loader.memcpy_htod(
            self._d_query,
            ctypes.c_void_p(query512.ctypes.data),
            query512.nbytes,
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
        output = self._host_saliency[: int(node_count)]
        loader.memcpy_dtoh(
            ctypes.c_void_p(output.ctypes.data),
            self._d_saliency,
            output.nbytes,
        )
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
        self._host_saliency = np.zeros((0, 2), dtype=np.float32)

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
        self._host_saliency = np.zeros((int(node_count), 2), dtype=np.float32)

    @classmethod
    def build_unified_host_buffer(
        cls,
        embeddings16: np.ndarray,
        morton_levels: np.ndarray,
    ) -> np.ndarray:
        node_count = int(embeddings16.shape[0])
        byte_buffer = np.zeros(node_count * cls.NODE_STRIDE_BYTES, dtype=np.uint8)
        float_view = byte_buffer.view(np.float32).reshape(node_count, cls.NODE_STRIDE_FLOATS)
        u32_view = byte_buffer.view(np.uint32).reshape(node_count, cls.NODE_STRIDE_FLOATS)
        projected = np.stack(
            [cls._project_embedding16_to512(row) for row in embeddings16],
            axis=0,
        )
        float_view[:, : cls.EMBEDDING_DIM] = projected
        u32_view[:, cls.MORTON_OFFSET_U32] = morton_levels.astype(np.uint32, copy=False)
        return byte_buffer

    @classmethod
    def _project_embedding16_to512(cls, embedding16: Iterable[float]) -> np.ndarray:
        values = list(float(value) for value in embedding16)
        if not values:
            return np.zeros(cls.EMBEDDING_DIM, dtype=np.float32)
        tiled = (values * ((cls.EMBEDDING_DIM + len(values) - 1) // len(values)))[: cls.EMBEDDING_DIM]
        return np.asarray(tiled, dtype=np.float32)


@dataclass
class QueryHeadSubstrate:
    """GPU-resident spatial substrate for the composed query head."""

    signature: str
    positions: np.ndarray
    galaxy_indexes: np.ndarray
    morton_levels: np.ndarray
    morton_octree: MortonOctreeSovereign
    frustum_culler: FrustumCuller
    dynamic_lod: DynamicLodDriverBridge

    @classmethod
    def build(cls, *, signature: str, catalog: list[dict]) -> "QueryHeadSubstrate":
        embeddings = np.asarray(
            [entry.get("embedding16", [0.0] * 16) for entry in catalog],
            dtype=np.float32,
        )
        galaxy_indexes = np.asarray(
            [int(round(float(entry.get("gpu_galaxy_index", 0.0)))) for entry in catalog],
            dtype=np.int32,
        )
        positions = np.asarray(
            [
                _semantic_position3(
                    embedding=row,
                    galaxy_index=float(entry.get("gpu_galaxy_index", 0.0)),
                    domain_hash=float(entry.get("domain_hash", 0.0)) if "domain_hash" in entry else 0.0,
                    subject_hash=float(entry.get("subject_hash", 0.0)) if "subject_hash" in entry else 0.0,
                )
                for row, entry in zip(embeddings, catalog)
            ],
            dtype=np.float32,
        )
        morton_octree = MortonOctreeSovereign()
        morton_tree = morton_octree.build_tree(positions)
        morton_codes = morton_tree["codes"]
        morton_levels = np.asarray([_morton_level(code) for code in morton_codes], dtype=np.uint32)
        level_by_index = np.zeros(len(catalog), dtype=np.uint32)
        sorted_indices = np.asarray(morton_tree["indices"], dtype=np.uint32)
        level_by_index[sorted_indices] = morton_levels

        dynamic_lod = DynamicLodDriverBridge()
        unified_host = DynamicLodDriverBridge.build_unified_host_buffer(
            embeddings16=embeddings,
            morton_levels=level_by_index,
        )
        dynamic_lod.bind_unified_buffer(unified_host, len(catalog))

        frustum_culler = FrustumCuller(enable_profiling=False)
        return cls(
            signature=str(signature),
            positions=positions,
            galaxy_indexes=galaxy_indexes,
            morton_levels=level_by_index,
            morton_octree=morton_octree,
            frustum_culler=frustum_culler,
            dynamic_lod=dynamic_lod,
        )

    def morton_locate(
        self,
        *,
        query_embedding16: Iterable[float],
        allowed_galaxy_indexes: set[int] | None,
        max_results: int,
        morton_radius: int,
        euclidean_radius: float,
    ) -> np.ndarray:
        query_position = _semantic_position3(np.asarray(list(query_embedding16), dtype=np.float32), 0.0, 0.0, 0.0)
        result = self.morton_octree.query_radius(
            np.asarray(query_position, dtype=np.float32),
            morton_radius=int(morton_radius),
            euclidean_radius=float(euclidean_radius),
            max_results=int(max_results),
        )
        if allowed_galaxy_indexes:
            mask = np.isin(
                self.galaxy_indexes[result.astype(np.int32, copy=False)],
                np.asarray(sorted(allowed_galaxy_indexes), dtype=np.int32),
            )
            result = result[mask]
        return result

    def frustum_visible(
        self,
        *,
        query_embedding16: Iterable[float],
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(candidate_indices, dtype=np.uint32).reshape(-1)
        if indices.size == 0:
            return indices
        query_position = np.asarray(
            _semantic_position3(np.asarray(list(query_embedding16), dtype=np.float32), 0.0, 0.0, 0.0),
            dtype=np.float32,
        )
        target = np.mean(self.positions[indices.astype(np.int32, copy=False)], axis=0)
        if not np.isfinite(target).all():
            return indices
        direction = target - query_position
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return indices
        eye = query_position - (direction / norm) * 0.35
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(direction / norm, up))) > 0.95:
            up = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        view = create_view_matrix(eye, target, up)
        proj = create_perspective_matrix(72.0, 1.0, 0.01, 8.0)
        view_proj = proj @ view
        return self.frustum_culler.cull_nodes(
            self.positions,
            indices,
            view_proj=view_proj,
            view=view,
        )

    def lod_metrics(
        self,
        *,
        query_embedding16: Iterable[float],
        candidate_indices: np.ndarray,
        saliency_threshold: float = 0.62,
    ) -> dict[int, tuple[float, int]]:
        indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        if indices.size == 0:
            return {}
        saliency_map = self.dynamic_lod.tune(
            list(query_embedding16),
            len(self.positions),
            saliency_threshold=saliency_threshold,
        )
        metrics: dict[int, tuple[float, int]] = {}
        for node_index in indices.tolist():
            cosine = float(saliency_map[node_index, 0])
            lod_level = int(round(float(saliency_map[node_index, 1])))
            metrics[int(node_index)] = (cosine, lod_level)
        return metrics

    def close(self) -> None:
        try:
            self.dynamic_lod.close()
        finally:
            try:
                self.frustum_culler.close()
            finally:
                self.morton_octree.close()


def expand_embedding16_to128(embedding16: Iterable[float]) -> np.ndarray:
    values = list(float(value) for value in embedding16)
    if not values:
        return np.zeros(128, dtype=np.float32)
    tiled = (values * ((128 + len(values) - 1) // len(values)))[:128]
    return np.asarray(tiled, dtype=np.float32)


def halting_inputs(
    scores: Iterable[float],
    *,
    candidate_ids: Iterable[str] = (),
    minimum_threshold: float = 0.3,
    gap_threshold: float = 0.1,
    agreement_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = [float(score) for score in scores]
    if not values:
        return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.uint32)

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
        np.asarray([min_norm, gap_norm, agree_norm, converged_norm], dtype=np.float32),
        np.ones(4, dtype=np.uint32),
    )


def relative_halting_inputs(
    scores: Iterable[float],
    *,
    gap_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    values = sorted((float(score) for score in scores), reverse=True)
    if not values:
        return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.uint32)
    top_score = float(values[0])
    second_score = float(values[1]) if len(values) > 1 else 0.0
    score_gap = top_score - second_score
    gap_norm = score_gap / max(float(gap_threshold), 1e-6)
    return (
        np.asarray([1.0, gap_norm, 1.0, gap_norm], dtype=np.float32),
        np.ones(4, dtype=np.uint32),
    )


def _semantic_position3(
    embedding: np.ndarray,
    galaxy_index: float,
    domain_hash: float,
    subject_hash: float,
) -> list[float]:
    values = list(float(value) for value in embedding[:3])
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


def _morton_level(code: int) -> int:
    return max(0, min(int(bin(int(code)).count("1") // 3), 12))
