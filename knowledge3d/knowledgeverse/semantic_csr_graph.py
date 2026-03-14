"""Bind-time semantic CSR graph construction for Knowledgeverse LED navigation.

This module is intentionally build-time / query-support code. It can use NumPy
because it is not part of the sovereign PTX hot path itself; it prepares sparse
graph structures that the LED bridge consumes during query execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Any

import numpy as np


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
    knn_k: int
    similarity_threshold: float

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
            knn_k=int(payload["knn_k"][0]),
            similarity_threshold=float(payload["similarity_threshold"][0]),
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
        knn_k=int(knn_k),
        similarity_threshold=float(similarity_threshold),
    )


__all__ = ["SemanticCSRGraph", "load_or_build_semantic_csr_graph"]
