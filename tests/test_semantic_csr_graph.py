from __future__ import annotations

from knowledge3d.knowledgeverse.semantic_csr_graph import load_or_build_semantic_csr_graph


def _entry(entry_id: str, embedding: list[float], *, galaxy: str = "Reality", category: str = "concept") -> dict[str, object]:
    return {
        "id": entry_id,
        "galaxy": galaxy,
        "category": category,
        "gpu_galaxy_index": 1,
        "embedding16": embedding + [0.0] * (16 - len(embedding)),
    }


def _row_neighbors(graph, row_index: int) -> list[int]:
    start = int(graph.row_offsets[row_index])
    end = int(graph.row_offsets[row_index + 1])
    return [int(value) for value in graph.col_indices[start:end].tolist()]


def test_semantic_csr_graph_gpu_knn_builds_expected_edges(tmp_path):
    catalog = [
        _entry("anchor", [1.0, 0.0]),
        _entry("near_anchor", [0.9, 0.1]),
        _entry("opposite", [-1.0, 0.0]),
        _entry("orthogonal", [0.0, 1.0]),
    ]

    graph = load_or_build_semantic_csr_graph(
        catalog=catalog,
        cache_root=tmp_path / "graph_cache",
        knn_k=2,
        similarity_threshold=0.2,
        batch_size=2,
    )

    assert graph.cache_hit is False
    assert graph.build_backend == "gpu_knn"
    assert _row_neighbors(graph, 0) == [1]
    assert _row_neighbors(graph, 1) == [0]
    assert _row_neighbors(graph, 2) == []
    assert _row_neighbors(graph, 3) == []

    pairs = graph.select_seed_nodes(
        query_embedding=[1.0, 0.0] + [0.0] * 14,
        top_k=2,
        similarity_threshold=0.2,
    )
    assert [index for index, _ in pairs] == [0, 1]
    assert pairs[0][1] > pairs[1][1] > 0.9
    graph.close()


def test_semantic_csr_graph_cache_round_trip_marks_cache_hit(tmp_path):
    catalog = [
        _entry("left", [1.0, 0.0]),
        _entry("right", [0.0, 1.0]),
    ]

    first = load_or_build_semantic_csr_graph(
        catalog=catalog,
        cache_root=tmp_path / "graph_cache",
        knn_k=1,
        similarity_threshold=0.0,
        batch_size=1,
    )
    second = load_or_build_semantic_csr_graph(
        catalog=catalog,
        cache_root=tmp_path / "graph_cache",
        knn_k=1,
        similarity_threshold=0.0,
        batch_size=1,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.build_backend == "cache"
    assert second.row_offsets.tolist() == first.row_offsets.tolist()
    assert second.col_indices.tolist() == first.col_indices.tolist()
    assert second.packed_costs.tolist() == first.packed_costs.tolist()
    first.close()
    second.close()
