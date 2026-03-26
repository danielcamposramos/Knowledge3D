from __future__ import annotations

from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse import query_head_substrate as qhs


def test_device_navigation_restores_led_focus_and_subject_cluster(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_phase_d3b")

    catalog = [
        {
            "id": "generic_math",
            "galaxy": "Math",
            "embedding16": [1.0] + [0.0] * 15,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Math"),
            "confidence": 0.4,
        },
        {
            "id": "subject_match",
            "galaxy": "Reality",
            "embedding16": [0.0, 1.0] + [0.0] * 14,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Reality"),
            "confidence": 0.7,
        },
    ]

    class _FakeSubstrate:
        def frustum_visible_device(self, **kwargs):
            return int(kwargs["d_candidate_indices"]), int(kwargs["candidate_count"])

        def lod_metrics_device(self, **kwargs):
            return int(kwargs["d_candidate_indices"]), int(kwargs["candidate_count"])

        def read_top_candidates(self, **kwargs):
            return [0, 1], {0: (0.2, 5), 1: (0.8, 2)}, {"raw_count": 2, "visible_count": 2, "top_count": 2}

    class _FakeGraph:
        def subject_cluster_id(self, subject_hint):
            return 7 if str(subject_hint) == "college_physics" else 0

        def subject_cluster_for_index(self, index):
            return 7 if int(index) == 1 else 0

        def select_seed_nodes_device(self, **kwargs):
            return 111, 222, 2

        def read_seed_pairs(self, indices_ptr, scores_ptr, count):
            return [(0, 0.55), (1, 0.54)]

        def extract_local_kernel_device(self, **kwargs):
            return {
                "selected_nodes_ptr": 333,
                "selected_count": 2,
                "local_row_offsets_ptr": 444,
                "local_col_indices_ptr": 555,
                "local_packed_costs_ptr": 666,
                "local_edge_count": 1,
            }

        def read_selected_nodes(self, device_ptr, count):
            return [0, 1]

        def read_local_csr(self, **kwargs):
            return [0, 0, 0], [], []

    class _FakePathfinder:
        def navigate_csr_device(self, *args, **kwargs):
            return 999, 2

        def read_device_path(self, device_ptr, path_length):
            return qhs.UInt32Vector([0, 1])

    monkeypatch.setattr(kv, "get_query_head_substrate", lambda: _FakeSubstrate())
    monkeypatch.setattr(kv, "get_gpu_galaxy_catalog", lambda: list(catalog))
    monkeypatch.setattr(kv, "get_semantic_csr_graph", lambda: _FakeGraph())
    monkeypatch.setattr(kv, "get_led_pathfinder", lambda: _FakePathfinder())
    monkeypatch.setattr(kv, "_embedding_similarities", lambda reference, candidates: [0.2, 0.95])
    monkeypatch.setattr(
        kv,
        "_goal_edge_cost",
        lambda match, **kwargs: 0.9 if match.get("id") == "generic_math" else 0.1,
    )

    selection_steps: list[str] = []
    candidates = kv._compose_head_navigation_candidates_device(
        binding={"galaxies": list(kv.GPU_MMLU_TARGET_GALAXIES)},
        target_galaxies=list(kv.GPU_MMLU_TARGET_GALAXIES),
        galaxy_weights=None,
        reasoning_program_id=kv.GPU_CHAT_REASONING_PROGRAM_ID,
        query_embedding=[0.0, 1.0] + [0.0] * 14,
        task_type="MMLU_TASK",
        selection_steps=selection_steps,
        task={"type": "MMLU_TASK", "task_id": "phase_d3b_device"},
        query_text="college physics question",
        domain_hint="college_physics",
    )

    assert candidates
    assert candidates[0]["match"]["id"] == "subject_match"
    assert candidates[0]["led_focus"] == 1.0
    assert candidates[0]["subject_anchor_focus"] == 1.0
    assert any("Seed select/device" in step for step in selection_steps)
    assert any("Graph expand/device" in step for step in selection_steps)
    assert any("LED-A device local graph" in step for step in selection_steps)
    assert any("MMLU subject cluster bias/device" in step for step in selection_steps)


def test_build_candidate_adjacency_clamps_truncated_local_csr(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_phase_d3b_bounds")

    adjacency = kv._build_candidate_adjacency(
        visible_indices=[100, 200],
        local_nodes=[100, 200],
        local_rows=[0, 5, 7],
        local_cols=[1, 0],
    )

    assert adjacency[100] == [200]
    assert adjacency[200] == []
