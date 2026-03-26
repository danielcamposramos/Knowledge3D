"""Bind-time semantic CSR graph construction for Knowledgeverse LED navigation.

This module is intentionally build-time / query-support code. It can use NumPy
because it is not part of the sovereign PTX hot path itself; it prepares sparse
graph structures that the LED bridge consumes during query execution.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import Any

import numpy as np

from knowledge3d.cranium.sovereign import loader


def _fnv1a_u32(parts: list[str]) -> str:
    acc = 2166136261
    for part in parts:
        for ch in part:
            acc ^= ord(ch)
            acc = (acc * 16777619) & 0xFFFFFFFF
    return f"{acc:08x}"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def _normalized_subject_aliases(subject_hint: str) -> list[str]:
    hint = str(subject_hint).strip().lower()
    if not hint:
        return []
    normalized = hint.replace("-", "_").replace(" ", "_")
    aliases = {hint, normalized}
    compact = normalized.replace("_", "")
    if compact:
        aliases.add(compact)
    synonym_map = {
        "math": {"mathematics", "algebra"},
        "mathematics": {"math", "algebra"},
        "logic": {"formal_logic", "philosophy"},
        "computer_science": {"cs", "computerscience"},
        "cs": {"computer_science", "computerscience"},
        "computerscience": {"computer_science", "cs"},
        "cyber_security": {"cybersecurity"},
        "cybersecurity": {"cyber_security"},
    }
    for alias in tuple(aliases):
        aliases.update(synonym_map.get(alias, set()))
    return [alias for alias in aliases if alias]


def _entry_subject_aliases(entry: dict[str, Any]) -> list[str]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    aliases: list[str] = []
    explicit = metadata.get("mmlu_subjects") if isinstance(metadata.get("mmlu_subjects"), list) else []
    for item in explicit:
        aliases.extend(_normalized_subject_aliases(str(item)))
    for raw in (
        entry.get("subject"),
        metadata.get("subject"),
        metadata.get("subfield"),
        entry.get("domain"),
        metadata.get("domain"),
    ):
        if str(raw or "").strip():
            aliases.extend(_normalized_subject_aliases(str(raw)))
    metadata_aliases = metadata.get("aliases") if isinstance(metadata.get("aliases"), list) else []
    for item in metadata_aliases:
        aliases.extend(_normalized_subject_aliases(str(item)))
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        clean = str(alias).strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _build_subject_clusters(
    catalog: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    alias_to_cluster: dict[str, int] = {}
    clusters = np.zeros(len(catalog), dtype=np.uint16)
    next_cluster = 1
    for index, entry in enumerate(catalog):
        aliases = _entry_subject_aliases(entry)
        cluster_id = 0
        for alias in aliases:
            existing = alias_to_cluster.get(alias)
            if existing is not None:
                cluster_id = int(existing)
                break
        if cluster_id == 0 and aliases:
            cluster_id = int(next_cluster)
            next_cluster += 1
        for alias in aliases:
            alias_to_cluster.setdefault(alias, int(cluster_id))
        clusters[index] = np.uint16(cluster_id)
    return clusters, alias_to_cluster


def _semantic_cost_from_similarity(similarity: float) -> int:
    sim = max(-1.0, min(float(similarity), 1.0))
    normalized = 1.0 - ((sim + 1.0) * 0.5)
    return int(round(normalized * 65535.0))


def _pack_led_cost(semantic_cost: int, geometric_cost: int) -> int:
    sem = max(0, min(int(semantic_cost), 0xFFFF))
    geo = max(0, min(int(geometric_cost), 0xFFFF))
    return int((sem << 16) | geo)


def _geometric_cost(left: dict[str, Any], right: dict[str, Any]) -> int:
    same_template = bool(
        str(left.get("template_ref", "")).strip()
        and str(left.get("template_ref", "")).strip() == str(right.get("template_ref", "")).strip()
    )
    same_subject = bool(
        str(left.get("subject", "")).strip()
        and str(left.get("subject", "")).strip().lower() == str(right.get("subject", "")).strip().lower()
    )
    same_galaxy = str(left.get("galaxy", "")).strip() == str(right.get("galaxy", "")).strip()
    same_category = str(left.get("category", "")).strip().lower() == str(right.get("category", "")).strip().lower()
    if same_template:
        return 1
    if same_subject:
        return 2
    if same_galaxy:
        return 4
    if same_category:
        return 6
    return 9


@dataclass
class SemanticCSRGraph:
    signature: str
    embeddings: np.ndarray
    galaxy_indexes: np.ndarray
    row_offsets: np.ndarray
    col_indices: np.ndarray
    packed_costs: np.ndarray
    subject_clusters: np.ndarray
    knn_k: int
    similarity_threshold: float
    subject_alias_clusters: dict[str, int] = field(default_factory=dict, repr=False)
    _seed_module: Any | None = field(default=None, init=False, repr=False)
    _seed_kernel: Any | None = field(default=None, init=False, repr=False)
    _graph_module: Any | None = field(default=None, init=False, repr=False)
    _graph_kernel: Any | None = field(default=None, init=False, repr=False)
    _d_embeddings: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_galaxy_indexes: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_row_offsets: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_col_indices: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_packed_costs: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_subject_clusters: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_allowed_galaxies: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _allowed_capacity: int = field(default=0, init=False, repr=False)
    _d_seed_indices: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_seed_scores: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_seed_count: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _seed_capacity: int = field(default=0, init=False, repr=False)
    _d_selected_nodes: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_selected_count: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_local_row_offsets: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_local_col_indices: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_local_packed_costs: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _d_local_edge_count: loader.CUdeviceptr | None = field(default=None, init=False, repr=False)
    _selected_capacity: int = field(default=0, init=False, repr=False)
    _local_edge_capacity: int = field(default=0, init=False, repr=False)

    def select_seed_nodes(
        self,
        *,
        query_embedding: list[float],
        allowed_galaxy_indexes: set[int] | None = None,
        top_k: int = 8,
        similarity_threshold: float = 0.18,
    ) -> list[tuple[int, float]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embeddings.shape[1]:
            padded = np.zeros((1, self.embeddings.shape[1]), dtype=np.float32)
            width = min(query.shape[1], self.embeddings.shape[1])
            padded[0, :width] = query[0, :width]
            query = padded
        query = _normalize_rows(query)
        similarities = (self.embeddings @ query[0]).astype(np.float32)
        if allowed_galaxy_indexes:
            mask = np.isin(self.galaxy_indexes, np.asarray(sorted(allowed_galaxy_indexes), dtype=np.int32))
            similarities = np.where(mask, similarities, -np.inf)
        candidates: list[tuple[int, float]] = []
        limit = min(int(top_k), int(similarities.shape[0]))
        if limit <= 0:
            return candidates
        top_idx = np.argpartition(similarities, -limit)[-limit:]
        ordered = top_idx[np.argsort(similarities[top_idx])[::-1]]
        for raw_index in ordered.tolist():
            similarity = float(similarities[raw_index])
            if similarity < float(similarity_threshold):
                continue
            candidates.append((int(raw_index), similarity))
        return candidates

    def subject_cluster_id(self, subject_hint: str) -> int:
        for alias in _normalized_subject_aliases(subject_hint):
            cluster_id = self.subject_alias_clusters.get(alias)
            if cluster_id is not None:
                return int(cluster_id)
        return 0

    def subject_cluster_for_index(self, index: int) -> int:
        if not (0 <= int(index) < int(self.subject_clusters.shape[0])):
            return 0
        return int(self.subject_clusters[int(index)])

    def ensure_device_buffers(self) -> None:
        if self._d_embeddings is not None:
            return
        ptx_dir = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
        seed_ptx = ptx_dir / "seed_select_top_k.ptx"
        graph_ptx = ptx_dir / "graph_expand_bfs.ptx"
        if not seed_ptx.exists():
            raise FileNotFoundError(f"Seed selection PTX missing: {seed_ptx}")
        if not graph_ptx.exists():
            raise FileNotFoundError(f"Graph expansion PTX missing: {graph_ptx}")
        self._seed_module = loader.load_module_from_file(str(seed_ptx))
        self._seed_kernel = loader.get_function(self._seed_module, "seed_select_top_k")
        self._graph_module = loader.load_module_from_file(str(graph_ptx))
        self._graph_kernel = loader.get_function(self._graph_module, "graph_expand_bfs")

        self._d_embeddings = loader.gpu_malloc(int(self.embeddings.nbytes))
        self._d_galaxy_indexes = loader.gpu_malloc(int(self.galaxy_indexes.nbytes))
        self._d_row_offsets = loader.gpu_malloc(int(self.row_offsets.nbytes))
        self._d_col_indices = loader.gpu_malloc(int(self.col_indices.nbytes))
        self._d_packed_costs = loader.gpu_malloc(int(self.packed_costs.nbytes))
        self._d_subject_clusters = loader.gpu_malloc(int(self.subject_clusters.nbytes))
        loader.memcpy_htod(self._d_embeddings, self.embeddings.ctypes.data_as(ctypes.c_void_p), int(self.embeddings.nbytes))
        loader.memcpy_htod(self._d_galaxy_indexes, self.galaxy_indexes.ctypes.data_as(ctypes.c_void_p), int(self.galaxy_indexes.nbytes))
        loader.memcpy_htod(self._d_row_offsets, self.row_offsets.ctypes.data_as(ctypes.c_void_p), int(self.row_offsets.nbytes))
        loader.memcpy_htod(self._d_col_indices, self.col_indices.ctypes.data_as(ctypes.c_void_p), int(self.col_indices.nbytes))
        loader.memcpy_htod(self._d_packed_costs, self.packed_costs.ctypes.data_as(ctypes.c_void_p), int(self.packed_costs.nbytes))
        loader.memcpy_htod(self._d_subject_clusters, self.subject_clusters.ctypes.data_as(ctypes.c_void_p), int(self.subject_clusters.nbytes))

    def close(self) -> None:
        for attr in (
            "_d_embeddings",
            "_d_galaxy_indexes",
            "_d_row_offsets",
            "_d_col_indices",
            "_d_packed_costs",
            "_d_subject_clusters",
            "_d_allowed_galaxies",
            "_d_seed_indices",
            "_d_seed_scores",
            "_d_seed_count",
            "_d_selected_nodes",
            "_d_selected_count",
            "_d_local_row_offsets",
            "_d_local_col_indices",
            "_d_local_packed_costs",
            "_d_local_edge_count",
        ):
            ptr = getattr(self, attr, None)
            if ptr is None:
                continue
            try:
                loader.gpu_free(ptr)
            except Exception:
                pass
            setattr(self, attr, None)
        self._allowed_capacity = 0
        self._seed_capacity = 0
        self._selected_capacity = 0
        self._local_edge_capacity = 0

    def reset_traversal_state(self) -> None:
        # Keep the immutable graph in VRAM; clear only transient traversal outputs.
        for attr in (
            "_d_allowed_galaxies",
            "_d_seed_indices",
            "_d_seed_scores",
            "_d_seed_count",
            "_d_selected_nodes",
            "_d_selected_count",
            "_d_local_row_offsets",
            "_d_local_col_indices",
            "_d_local_packed_costs",
            "_d_local_edge_count",
        ):
            ptr = getattr(self, attr, None)
            if ptr is None:
                continue
            try:
                loader.gpu_free(ptr)
            except Exception:
                pass
            setattr(self, attr, None)
        self._allowed_capacity = 0
        self._seed_capacity = 0
        self._selected_capacity = 0
        self._local_edge_capacity = 0

    def _ensure_allowed_capacity(self, count: int) -> None:
        required = max(0, int(count))
        if required <= 0:
            return
        if self._d_allowed_galaxies is not None and required <= self._allowed_capacity:
            return
        if self._d_allowed_galaxies is not None:
            loader.gpu_free(self._d_allowed_galaxies)
        self._d_allowed_galaxies = loader.gpu_malloc(max(required * 4, 4))
        self._allowed_capacity = required

    def _ensure_seed_capacity(self, top_k: int) -> None:
        required = max(1, int(top_k))
        if required <= self._seed_capacity and self._d_seed_indices is not None:
            return
        for attr in ("_d_seed_indices", "_d_seed_scores", "_d_seed_count"):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                loader.gpu_free(ptr)
                setattr(self, attr, None)
        self._d_seed_indices = loader.gpu_malloc(required * 4)
        self._d_seed_scores = loader.gpu_malloc(required * 4)
        self._d_seed_count = loader.gpu_malloc(4)
        self._seed_capacity = required

    def _ensure_local_kernel_capacity(self, max_nodes: int, max_edges: int) -> None:
        nodes = max(1, int(max_nodes))
        edges = max(1, int(max_edges))
        needs_nodes = nodes > self._selected_capacity or self._d_selected_nodes is None
        needs_edges = edges > self._local_edge_capacity or self._d_local_col_indices is None
        if needs_nodes:
            for attr in ("_d_selected_nodes", "_d_selected_count", "_d_local_row_offsets"):
                ptr = getattr(self, attr, None)
                if ptr is not None:
                    loader.gpu_free(ptr)
                    setattr(self, attr, None)
            self._d_selected_nodes = loader.gpu_malloc(nodes * 4)
            self._d_selected_count = loader.gpu_malloc(4)
            self._d_local_row_offsets = loader.gpu_malloc((nodes + 1) * 4)
            self._selected_capacity = nodes
        if needs_edges:
            for attr in ("_d_local_col_indices", "_d_local_packed_costs", "_d_local_edge_count"):
                ptr = getattr(self, attr, None)
                if ptr is not None:
                    loader.gpu_free(ptr)
                    setattr(self, attr, None)
            self._d_local_col_indices = loader.gpu_malloc(edges * 4)
            self._d_local_packed_costs = loader.gpu_malloc(edges * 4)
            self._d_local_edge_count = loader.gpu_malloc(4)
            self._local_edge_capacity = edges

    def select_seed_nodes_device(
        self,
        *,
        query_embedding: list[float],
        allowed_galaxy_indexes: set[int] | None = None,
        top_k: int = 8,
        similarity_threshold: float = 0.18,
        target_cluster_id: int = 0,
        cluster_bias: float = 0.0,
    ) -> tuple[int, int, int]:
        self.ensure_device_buffers()
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embeddings.shape[1]:
            padded = np.zeros((1, self.embeddings.shape[1]), dtype=np.float32)
            width = min(query.shape[1], self.embeddings.shape[1])
            padded[0, :width] = query[0, :width]
            query = padded
        query = _normalize_rows(query)
        query_host = np.asarray(query[0], dtype=np.float32)
        d_query = loader.gpu_malloc(int(query_host.nbytes))
        try:
            loader.memcpy_htod(d_query, query_host.ctypes.data_as(ctypes.c_void_p), int(query_host.nbytes))
            allowed_count = 0
            allowed_ptr = loader.CUdeviceptr(0)
            if allowed_galaxy_indexes:
                allowed_values = sorted(int(value) for value in allowed_galaxy_indexes)
                self._ensure_allowed_capacity(len(allowed_values))
                allowed_buf = (ctypes.c_int32 * len(allowed_values))(*allowed_values)
                loader.memcpy_htod(
                    self._d_allowed_galaxies,
                    ctypes.c_void_p(ctypes.addressof(allowed_buf)),
                    ctypes.sizeof(allowed_buf),
                )
                allowed_count = len(allowed_values)
                allowed_ptr = self._d_allowed_galaxies
            self._ensure_seed_capacity(int(top_k))
            zero_i32 = (ctypes.c_int32 * 1)(0)
            loader.memcpy_htod(self._d_seed_count, ctypes.c_void_p(ctypes.addressof(zero_i32)), ctypes.sizeof(zero_i32))
            loader.launch(
                self._seed_kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    self._d_embeddings,
                    self._d_galaxy_indexes,
                    d_query,
                    allowed_ptr,
                    ctypes.c_int32(int(allowed_count)),
                    ctypes.c_int32(int(self.embeddings.shape[0])),
                    ctypes.c_int32(int(self.embeddings.shape[1])),
                    ctypes.c_int32(int(top_k)),
                    ctypes.c_float(float(similarity_threshold)),
                    ctypes.c_int32(int(target_cluster_id)),
                    ctypes.c_float(float(cluster_bias)),
                    self._d_subject_clusters,
                    self._d_seed_indices,
                    self._d_seed_scores,
                    self._d_seed_count,
                ],
            )
            loader.synchronize()
            count_buf = (ctypes.c_int32 * 1)(0)
            loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(count_buf)), self._d_seed_count, ctypes.sizeof(count_buf))
            actual_count = max(0, min(int(count_buf[0]), int(top_k)))
            return int(self._d_seed_indices.value), int(self._d_seed_scores.value), actual_count
        finally:
            loader.gpu_free(d_query)

    def read_seed_pairs(
        self,
        indices_ptr: int | loader.CUdeviceptr,
        scores_ptr: int | loader.CUdeviceptr,
        count: int,
    ) -> list[tuple[int, float]]:
        actual = max(0, int(count))
        if actual <= 0:
            return []
        idx_buf = (ctypes.c_int32 * actual)()
        score_buf = (ctypes.c_float * actual)()
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(idx_buf)),
            loader.CUdeviceptr(int(indices_ptr)),
            ctypes.sizeof(idx_buf),
        )
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(score_buf)),
            loader.CUdeviceptr(int(scores_ptr)),
            ctypes.sizeof(score_buf),
        )
        pairs: list[tuple[int, float]] = []
        for idx in range(actual):
            node = int(idx_buf[idx])
            similarity = float(score_buf[idx])
            if node < 0:
                continue
            pairs.append((node, similarity))
        return pairs

    def extract_local_kernel(
        self,
        *,
        seed_nodes: list[int],
        max_nodes: int = 2048,
        max_edge_expansions: int = 24576,
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        if not seed_nodes:
            return [], np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32)
        frontier: list[tuple[float, int]] = [(0.0, int(node)) for node in seed_nodes]
        best_cost: dict[int, float] = {int(node): 0.0 for node in seed_nodes}
        selected: list[int] = []
        selected_set: set[int] = set()
        expansions = 0
        while frontier and len(selected) < int(max_nodes) and expansions < int(max_edge_expansions):
            cost, node = heapq.heappop(frontier)
            if node in selected_set:
                continue
            selected.append(node)
            selected_set.add(node)
            row_start = int(self.row_offsets[node])
            row_end = int(self.row_offsets[node + 1])
            for edge_idx in range(row_start, row_end):
                neighbor = int(self.col_indices[edge_idx])
                packed = int(self.packed_costs[edge_idx])
                geo = float(packed & 0xFFFF)
                sem = float((packed >> 16) & 0xFFFF)
                edge_cost = (0.35 * geo) + (0.65 * sem)
                tentative = cost + edge_cost
                if tentative + 1e-6 < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = tentative
                    heapq.heappush(frontier, (tentative, neighbor))
                expansions += 1
                if expansions >= int(max_edge_expansions):
                    break
        mapping = {global_index: local_index for local_index, global_index in enumerate(selected)}
        local_rows = [0]
        local_cols: list[int] = []
        local_costs: list[int] = []
        for global_index in selected:
            row_start = int(self.row_offsets[global_index])
            row_end = int(self.row_offsets[global_index + 1])
            for edge_idx in range(row_start, row_end):
                neighbor = int(self.col_indices[edge_idx])
                local_neighbor = mapping.get(neighbor)
                if local_neighbor is None:
                    continue
                local_cols.append(int(local_neighbor))
                local_costs.append(int(self.packed_costs[edge_idx]))
            local_rows.append(len(local_cols))
        return (
            selected,
            np.asarray(local_rows, dtype=np.uint32),
            np.asarray(local_cols, dtype=np.uint32),
            np.asarray(local_costs, dtype=np.uint32),
        )

    def extract_local_kernel_device(
        self,
        *,
        seed_indices_ptr: int | loader.CUdeviceptr,
        seed_count: int,
        max_nodes: int = 2048,
        max_edge_expansions: int = 24576,
        alpha: float = 0.35,
        beta: float = 0.65,
    ) -> dict[str, int]:
        self.ensure_device_buffers()
        actual_seed_count = max(0, int(seed_count))
        if actual_seed_count <= 0:
            return {
                "selected_nodes_ptr": 0,
                "selected_count": 0,
                "local_row_offsets_ptr": 0,
                "local_col_indices_ptr": 0,
                "local_packed_costs_ptr": 0,
                "local_edge_count": 0,
            }
        self._ensure_local_kernel_capacity(int(max_nodes), int(max_edge_expansions))
        zero_i32 = (ctypes.c_int32 * 1)(0)
        loader.memcpy_htod(self._d_selected_count, ctypes.c_void_p(ctypes.addressof(zero_i32)), ctypes.sizeof(zero_i32))
        loader.memcpy_htod(self._d_local_edge_count, ctypes.c_void_p(ctypes.addressof(zero_i32)), ctypes.sizeof(zero_i32))
        loader.launch(
            self._graph_kernel,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            params=[
                self._d_row_offsets,
                self._d_col_indices,
                self._d_packed_costs,
                loader.CUdeviceptr(int(seed_indices_ptr)),
                ctypes.c_int32(int(actual_seed_count)),
                ctypes.c_int32(int(max_nodes)),
                ctypes.c_int32(int(max_edge_expansions)),
                ctypes.c_float(float(alpha)),
                ctypes.c_float(float(beta)),
                self._d_selected_nodes,
                self._d_selected_count,
                self._d_local_row_offsets,
                self._d_local_col_indices,
                self._d_local_packed_costs,
                self._d_local_edge_count,
            ],
        )
        loader.synchronize()
        selected_buf = (ctypes.c_int32 * 1)(0)
        edge_buf = (ctypes.c_int32 * 1)(0)
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(selected_buf)), self._d_selected_count, ctypes.sizeof(selected_buf))
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(edge_buf)), self._d_local_edge_count, ctypes.sizeof(edge_buf))
        selected_count = max(0, min(int(selected_buf[0]), int(max_nodes)))
        edge_count = max(0, min(int(edge_buf[0]), int(max_edge_expansions)))
        return {
            "selected_nodes_ptr": int(self._d_selected_nodes.value),
            "selected_count": selected_count,
            "local_row_offsets_ptr": int(self._d_local_row_offsets.value),
            "local_col_indices_ptr": int(self._d_local_col_indices.value),
            "local_packed_costs_ptr": int(self._d_local_packed_costs.value),
            "local_edge_count": edge_count,
        }

    def read_selected_nodes(self, device_ptr: int | loader.CUdeviceptr, count: int) -> list[int]:
        actual = max(0, int(count))
        if actual <= 0:
            return []
        buf = (ctypes.c_int32 * actual)()
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(buf)),
            loader.CUdeviceptr(int(device_ptr)),
            ctypes.sizeof(buf),
        )
        return [int(buf[idx]) for idx in range(actual) if int(buf[idx]) >= 0]

    def read_local_csr(
        self,
        *,
        row_offsets_ptr: int | loader.CUdeviceptr,
        col_indices_ptr: int | loader.CUdeviceptr,
        packed_costs_ptr: int | loader.CUdeviceptr,
        node_count: int,
        edge_count: int,
    ) -> tuple[list[int], list[int], list[int]]:
        rows_count = max(0, int(node_count)) + 1
        cols_count = max(0, int(edge_count))
        row_buf = (ctypes.c_uint32 * rows_count)()
        col_buf = (ctypes.c_uint32 * cols_count)()
        cost_buf = (ctypes.c_uint32 * cols_count)()
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(row_buf)),
            loader.CUdeviceptr(int(row_offsets_ptr)),
            ctypes.sizeof(row_buf),
        )
        if cols_count > 0:
            loader.memcpy_dtoh(
                ctypes.c_void_p(ctypes.addressof(col_buf)),
                loader.CUdeviceptr(int(col_indices_ptr)),
                ctypes.sizeof(col_buf),
            )
            loader.memcpy_dtoh(
                ctypes.c_void_p(ctypes.addressof(cost_buf)),
                loader.CUdeviceptr(int(packed_costs_ptr)),
                ctypes.sizeof(cost_buf),
            )
        return (
            [int(row_buf[idx]) for idx in range(rows_count)],
            [int(col_buf[idx]) for idx in range(cols_count)],
            [int(cost_buf[idx]) for idx in range(cols_count)],
        )


def _catalog_signature(catalog: list[dict[str, Any]]) -> str:
    parts: list[str] = [str(len(catalog))]
    for entry in catalog:
        parts.append(str(entry.get("id", "")))
        parts.append(str(entry.get("galaxy", "")))
        parts.append(str(entry.get("category", "")))
        parts.append(str(entry.get("template_ref", "")))
        embedding = entry.get("embedding16")
        if isinstance(embedding, list):
            parts.extend(f"{float(value):.4f}" for value in embedding[:4])
    return _fnv1a_u32(parts)


def _cache_path(cache_root: Path, signature: str, knn_k: int, similarity_threshold: float) -> Path:
    threshold_key = int(round(float(similarity_threshold) * 1000.0))
    return cache_root / f"semantic_csr_{signature}_k{int(knn_k)}_t{threshold_key}.npz"


def load_or_build_semantic_csr_graph(
    *,
    catalog: list[dict[str, Any]],
    cache_root: str | Path,
    knn_k: int = 12,
    similarity_threshold: float = 0.3,
    batch_size: int = 512,
) -> SemanticCSRGraph:
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _catalog_signature(catalog)
    path = _cache_path(cache_dir, signature, knn_k, similarity_threshold)
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        return SemanticCSRGraph(
            signature=signature,
            embeddings=payload["embeddings"].astype(np.float32, copy=False),
            galaxy_indexes=payload["galaxy_indexes"].astype(np.int32, copy=False),
            row_offsets=payload["row_offsets"].astype(np.uint32, copy=False),
            col_indices=payload["col_indices"].astype(np.uint32, copy=False),
            packed_costs=payload["packed_costs"].astype(np.uint32, copy=False),
            subject_clusters=(
                payload["subject_clusters"].astype(np.uint16, copy=False)
                if "subject_clusters" in payload.files
                else _build_subject_clusters(catalog)[0]
            ),
            knn_k=int(payload["knn_k"][0]),
            similarity_threshold=float(payload["similarity_threshold"][0]),
            subject_alias_clusters=_build_subject_clusters(catalog)[1],
        )

    embeddings = np.asarray(
        [entry.get("embedding16", [0.0] * 16) for entry in catalog],
        dtype=np.float32,
    )
    galaxy_indexes = np.asarray(
        [int(round(float(entry.get("gpu_galaxy_index", 0.0)))) for entry in catalog],
        dtype=np.int32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("semantic CSR graph requires at least one embedding row")
    embeddings = _normalize_rows(embeddings)
    subject_clusters, subject_alias_clusters = _build_subject_clusters(catalog)

    node_count = int(embeddings.shape[0])
    neighbors: list[dict[int, int]] = [dict() for _ in range(node_count)]
    k_eff = max(1, min(int(knn_k), max(1, node_count - 1)))

    for start in range(0, node_count, int(batch_size)):
        end = min(node_count, start + int(batch_size))
        sims = (embeddings[start:end] @ embeddings.T).astype(np.float32)
        row_ids = np.arange(start, end, dtype=np.int64)
        sims[np.arange(end - start), row_ids - start] = -np.inf
        top_idx = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
        for local_row, candidate_indexes in enumerate(top_idx):
            source_index = start + local_row
            ordered = candidate_indexes[np.argsort(sims[local_row, candidate_indexes])[::-1]]
            source_entry = catalog[source_index]
            for target_index in ordered.tolist():
                similarity = float(sims[local_row, target_index])
                if similarity < float(similarity_threshold):
                    continue
                target_entry = catalog[int(target_index)]
                packed_cost = _pack_led_cost(
                    _semantic_cost_from_similarity(similarity),
                    _geometric_cost(source_entry, target_entry),
                )
                current = neighbors[source_index].get(int(target_index))
                if current is None or packed_cost < current:
                    neighbors[source_index][int(target_index)] = packed_cost
                reverse = neighbors[int(target_index)].get(source_index)
                if reverse is None or packed_cost < reverse:
                    neighbors[int(target_index)][source_index] = packed_cost

    row_offsets = [0]
    col_indices: list[int] = []
    packed_costs: list[int] = []
    for adjacency in neighbors:
        for target_index, packed_cost in sorted(adjacency.items()):
            col_indices.append(int(target_index))
            packed_costs.append(int(packed_cost))
        row_offsets.append(len(col_indices))

    np.savez_compressed(
        path,
        embeddings=embeddings,
        galaxy_indexes=galaxy_indexes,
        row_offsets=np.asarray(row_offsets, dtype=np.uint32),
        col_indices=np.asarray(col_indices, dtype=np.uint32),
        packed_costs=np.asarray(packed_costs, dtype=np.uint32),
        subject_clusters=subject_clusters,
        knn_k=np.asarray([int(knn_k)], dtype=np.int32),
        similarity_threshold=np.asarray([float(similarity_threshold)], dtype=np.float32),
    )

    return SemanticCSRGraph(
        signature=signature,
        embeddings=embeddings,
        galaxy_indexes=galaxy_indexes,
        row_offsets=np.asarray(row_offsets, dtype=np.uint32),
        col_indices=np.asarray(col_indices, dtype=np.uint32),
        packed_costs=np.asarray(packed_costs, dtype=np.uint32),
        subject_clusters=subject_clusters,
        knn_k=int(knn_k),
        similarity_threshold=float(similarity_threshold),
        subject_alias_clusters=subject_alias_clusters,
    )


__all__ = ["SemanticCSRGraph", "load_or_build_semantic_csr_graph"]
