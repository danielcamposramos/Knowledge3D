from __future__ import annotations

import ctypes

import pytest

from knowledge3d.knowledgeverse import query_head_substrate as qhs
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32


class _FakeMorton:
    def __init__(self, *args, **kwargs) -> None:
        self.bound = None

    def build_tree(self, positions):
        self.bound = positions
        return {
            "codes": qhs.UInt32Vector([3, 1, 2]),
            "indices": qhs.UInt32Vector([1, 0, 2]),
        }

    def query_radius(self, query_center, *, morton_radius, euclidean_radius, max_results):
        return qhs.UInt32Vector([2, 0, 1])

    def close(self) -> None:
        pass


class _FakeFrustum:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def cull_nodes(self, positions, candidate_indices, view_proj=None, view=None):
        return qhs.UInt32Vector(index for index in candidate_indices if int(index) != 1)

    def close(self) -> None:
        pass


class _FakeDynamicLod:
    def __init__(self, *args, **kwargs) -> None:
        self.bound = None

    def bind_unified_buffer(self, host_buffer, node_count: int) -> None:
        self.bound = (host_buffer, int(node_count))

    @classmethod
    def build_unified_host_buffer(cls, embeddings16, morton_levels):
        rows = len(list(embeddings16))
        buf = (ctypes.c_uint8 * (rows * 4096))()
        return buf

    def tune(self, query_embedding16, node_count: int, saliency_threshold: float = 0.62):
        rows = []
        for idx in range(int(node_count)):
            rows.append([0.1 * float(idx + 1), float(idx)])
        return HostTensorF32.from_array_like(rows, rows=int(node_count), cols=2)

    def close(self) -> None:
        pass


def test_expand_embedding16_to128_and_halting_helpers():
    expanded = qhs.expand_embedding16_to128([1.0, 2.0, 3.0])
    assert len(expanded) == 128
    assert expanded[:6] == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    scores, flags = qhs.halting_inputs([0.2, 0.4, 0.1])
    assert scores.shape == (4, 1)
    assert flags.tolist() == [1, 1, 1, 1]

    rel_scores, rel_flags = qhs.relative_halting_inputs([0.5, 0.2])
    assert rel_scores.shape == (4, 1)
    assert rel_flags.tolist() == [1, 1, 1, 1]


def test_query_head_substrate_build_and_queries(monkeypatch):
    monkeypatch.setattr(qhs, "MortonOctreeSovereign", _FakeMorton)
    monkeypatch.setattr(qhs, "FrustumCuller", _FakeFrustum)
    monkeypatch.setattr(qhs, "DynamicLodDriverBridge", _FakeDynamicLod)

    catalog = [
        {
            "embedding16": [1.0] * 16,
            "gpu_galaxy_index": 1,
            "domain_hash": 0.25,
            "subject_hash": 0.5,
        },
        {
            "embedding16": [0.5] * 16,
            "gpu_galaxy_index": 2,
            "domain_hash": 0.5,
            "subject_hash": 0.75,
        },
        {
            "embedding16": [0.25] * 16,
            "gpu_galaxy_index": 1,
        },
    ]

    substrate = qhs.QueryHeadSubstrate.build(signature="demo", catalog=catalog)

    assert substrate.signature == "demo"
    assert substrate.positions.shape == (3, 3)
    assert substrate.galaxy_indexes.tolist() == [1, 2, 1]
    assert substrate.morton_levels.tolist() == [0, 0, 0]

    located = substrate.morton_locate(
        query_embedding16=[0.1] * 16,
        allowed_galaxy_indexes={1},
        max_results=8,
        morton_radius=4,
        euclidean_radius=1.0,
    )
    assert located.tolist() == [2, 0]

    visible = substrate.frustum_visible(
        query_embedding16=[0.1] * 16,
        candidate_indices=[0, 1, 2],
    )
    assert visible.tolist() == [0, 2]

    metrics = substrate.lod_metrics(
        query_embedding16=[0.1] * 16,
        candidate_indices=[0, 2],
        saliency_threshold=0.62,
    )
    assert metrics[0][0] == pytest.approx(0.1)
    assert metrics[0][1] == 0
    assert metrics[2][0] == pytest.approx(0.3)
    assert metrics[2][1] == 2
