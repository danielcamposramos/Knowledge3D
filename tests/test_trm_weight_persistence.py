from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import DefeasibleResolver
from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.foundational_operations_bootstrap import (
    _foundational_reality_entries,
    foundational_reasoning_entries,
)
from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import NavigatorSpecialist
from knowledge3d.knowledgeverse.specialist_base import SpecialistBase
from knowledge3d.training.trm_galaxy_nav import (
    apply_trm_weights_to_traces,
    build_trace_balance_weights,
    compute_galaxy_idf,
    DEFAULT_GALAXY_ORDER,
    evaluate_decoder_on_traces,
    evaluate_trm_weights_on_traces,
    fit_galaxy_decoder_from_traces,
    initialize_trm_weight_matrices,
    save_galaxy_decoder_checkpoint,
    save_trm_weight_checkpoint,
    summarize_trace_top1_predictions,
    trace_target_logits,
    train_trm_weights_from_traces,
)


def test_phase_d_boot_wires_trm_launcher_without_enabling_it(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_boot")

    assert kv._trm is not None
    assert kv._trm_backend == "fused"
    assert kv._trm_ready is True
    assert kv._trm_init_error == ""
    assert set(kv._trm_weight_buffers.keys()) == {"W1", "W2", "W3", "W4"}
    assert kv._trm_weight_bytes > 0
    assert kv._matryoshka_bridge is not None
    assert kv._trm_matryoshka_host_weights is not None
    assert kv._trm_matryoshka_weight_buffer is not None
    assert kv._trm_galaxy_decoder is None
    assert kv._gpu_galaxy_binding is not None
    assert tuple(kv._gpu_galaxy_binding.get("galaxies", ())) == kv._live_galaxy_order
    assert kv._pinned_all_default_binding is True


def test_phase_d_state_buffers_and_stimulus_encoding_are_bootstrapped(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_state")

    assert set(kv._trm_state_buffers.keys()) == {
        "d_q_input",
        "d_q",
        "d_y",
        "d_z",
        "d_z_new",
        "d_y_new",
        "d_workspace",
    }
    assert all(ptr is not None for ptr in kv._trm_state_buffers.values())
    assert kv._trm_state_buffer_bytes == (
        ((6 * kv.TRM_STATE_VECTOR_DIM) + kv.TRM_WORKSPACE_FLOATS) * 4
    )
    assert kv._trm_weight_bytes >= kv._trm_state_buffer_bytes

    embedding16 = np.arange(1, 17, dtype=np.float32)
    q_host = kv._encode_stimulus(embedding16, readback=True)
    assert q_host is not None
    padded = np.zeros(kv.TRM_STATE_VECTOR_DIM, dtype=np.float32)
    padded[: embedding16.size] = embedding16
    np.testing.assert_allclose(
        q_host,
        np.clip(kv._trm_matryoshka_host_weights @ padded, -10.0, 10.0),
        atol=1e-4,
    )

    non_zero = np.ones(kv.TRM_STATE_VECTOR_DIM, dtype=np.float32)
    non_zero_workspace = np.ones(kv.TRM_WORKSPACE_FLOATS, dtype=np.float32)
    for key in ("d_y", "d_z", "d_z_new", "d_y_new"):
        loader.memcpy_htod(
            kv._trm_state_buffers[key],
            ctypes.c_void_p(non_zero.ctypes.data),
            non_zero.nbytes,
        )
    loader.memcpy_htod(
        kv._trm_state_buffers["d_workspace"],
        ctypes.c_void_p(non_zero_workspace.ctypes.data),
        non_zero_workspace.nbytes,
    )

    kv._reset_trm_state()

    for key in ("d_q_input", "d_y", "d_z", "d_z_new", "d_y_new"):
        host = np.empty(kv.TRM_STATE_VECTOR_DIM, dtype=np.float32)
        loader.memcpy_dtoh(
            ctypes.c_void_p(host.ctypes.data),
            kv._trm_state_buffers[key],
            host.nbytes,
        )
        np.testing.assert_array_equal(host, np.zeros_like(host))

    workspace_host = np.empty(kv.TRM_WORKSPACE_FLOATS, dtype=np.float32)
    loader.memcpy_dtoh(
        ctypes.c_void_p(workspace_host.ctypes.data),
        kv._trm_state_buffers["d_workspace"],
        workspace_host.nbytes,
    )
    np.testing.assert_array_equal(workspace_host, np.zeros_like(workspace_host))


def test_phase_d_trm_shadow_probe_returns_expected_diagnostics(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_shadow")

    shadow = kv._trm_shadow_probe(
        np.arange(1, 17, dtype=np.float32),
        target_galaxies=["Math", "Reality"],
        reasoning_program_id=kv.GPU_MATH_REASONING_PROGRAM_ID,
    )

    assert set(shadow.keys()) == {
        "y_new_top3_galaxies",
        "y_new_entropy",
        "trm_latency_us",
        "python_galaxies",
        "python_program",
        "query_embedding_512",
        "y_new_vector_512",
        "decoder_source",
        "decoder_checkpoint",
        "trm_recursion_steps",
        "trm_drift",
    }
    assert shadow["python_galaxies"] == ["Math", "Reality"]
    assert shadow["python_program"] == kv.GPU_MATH_REASONING_PROGRAM_ID
    assert float(shadow["trm_latency_us"]) > 0.0
    assert len(shadow["y_new_top3_galaxies"]) == 3
    assert all("galaxy" in item and "weight" in item for item in shadow["y_new_top3_galaxies"])
    assert len(shadow["query_embedding_512"]) == 512
    assert len(shadow["y_new_vector_512"]) == 512
    assert shadow["decoder_source"] == "raw_head"


def test_phase_d_trm_select_galaxies_uses_decoder_and_keeps_python_program(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_nav")
    decoder = {
        "W_galaxy": np.zeros((10, 512), dtype=np.float32),
        "b_galaxy": np.zeros(10, dtype=np.float32),
    }
    decoder["W_galaxy"][5, 0] = 2.0  # Math
    decoder["W_galaxy"][4, 1] = 1.5  # Grammar
    decoder["W_galaxy"][2, 2] = 1.0  # Word
    kv._trm_galaxy_decoder = decoder
    kv._trm_galaxy_decoder_path = str(tmp_path / "decoder.npz")

    y_new = np.zeros(512, dtype=np.float32)
    y_new[0] = 3.0
    y_new[1] = 2.2
    y_new[2] = 1.7
    monkeypatch.setattr(
        kv,
        "_run_single_trm_tick",
        lambda query_embedding: {
            "query_embedding_512": np.zeros(512, dtype=np.float32).tolist(),
            "y_new_vector_512": y_new.tolist(),
            "trm_latency_us": 100.0,
        },
    )

    galaxy_weights, program_id, meta = kv._trm_select_galaxies(
        np.arange(1, 17, dtype=np.float32),
        task_type="MATH_TASK",
        fallback_galaxies=["Reality", "Grammar"],
        reasoning_program_id=kv.GPU_MATH_REASONING_PROGRAM_ID,
    )

    assert galaxy_weights["Math"] > galaxy_weights["Grammar"] > galaxy_weights["Word"]
    assert program_id == kv.GPU_MATH_REASONING_PROGRAM_ID
    assert meta["status"] == "ok"
    assert meta["selected_galaxies"] == ["Grammar", "Math"]


def test_phase_d_trm_influence_strength_biases_around_uniform(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_TRM_INFLUENCE_STRENGTH", "1.0")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_influence")

    normalized = kv._normalize_galaxy_weights(
        {
            "Math": 0.99997,
            "Grammar": 0.00003,
            "Reality": 0.0,
        }
    )

    uniform = 1.0 / float(len(kv._live_galaxy_order))
    assert normalized["Math"] == pytest.approx(1.0 + (0.99997 - uniform))
    assert normalized["Grammar"] == pytest.approx(1.0 + (0.00003 - uniform))
    assert normalized["Reality"] == pytest.approx(1.0 - uniform)


def test_phase_d_zero_trm_influence_strength_is_neutral(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_TRM_INFLUENCE_STRENGTH", "0.0")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_neutral")

    normalized = kv._normalize_galaxy_weights(
        {
            "Math": 0.99997,
            "Grammar": 0.00003,
            "Reality": 0.0,
        }
    )

    for galaxy_name in kv._live_galaxy_order:
        assert normalized[galaxy_name] == pytest.approx(1.0)


def test_phase_d_seed_budget_concentrates_slots_on_target_and_positive_bias(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_seed_budget")

    budget = kv._allocate_galaxy_seed_budget(
        task_type="MATH_TASK",
        target_galaxies=["Math"],
        normalized_galaxy_weights={
            "Math": 1.45,
            "Reality": 1.15,
            "Grammar": 0.95,
            "Word": 0.95,
        },
    )

    assert sum(budget.values()) == 3
    assert budget["Math"] >= 1
    assert budget["Math"] >= budget.get("Reality", 0)
    assert "Grammar" not in budget


def test_phase_d_candidate_adjacency_threads_visible_csr_neighbors(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_candidate_adjacency")

    adjacency = kv._build_candidate_adjacency(
        visible_indices=[10, 12, 14],
        local_nodes=[9, 10, 11, 12, 14],
        local_rows=[0, 0, 2, 3, 5, 6],
        local_cols=[2, 3, 3, 2, 4, 3],
    )

    assert adjacency[10] == [12]
    assert adjacency[12] == [14]
    assert adjacency[14] == [12]


def test_phase_d_specialist_swarm_features_use_real_candidate_graph(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_specialist_graph")

    class _FakeGraphCrystallizer:
        def __init__(self):
            self.calls = []

        def crystallize_graph(self, node_features, adjacency, neighbor_counts, rounds, self_weight, neighbor_weight):
            self.calls.append(
                {
                    "shape": tuple(node_features.shape),
                    "adjacency": adjacency.tolist(),
                    "neighbor_counts": neighbor_counts.tolist(),
                    "rounds": rounds,
                    "self_weight": self_weight,
                    "neighbor_weight": neighbor_weight,
                }
            )
            return np.asarray(node_features, dtype=np.float32)

        def crystallize_list(self, *_args, **_kwargs):
            raise AssertionError("compatibility_path_should_not_run")

    fake_graph = _FakeGraphCrystallizer()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: fake_graph)

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Reality"},
            "candidate_global_idx": 10,
            "graph_neighbors": [11],
            "led_focus": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Reality"},
            "candidate_global_idx": 11,
            "graph_neighbors": [10],
            "led_focus": 0.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="LHE_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_graph.calls) == 1
    call = fake_graph.calls[0]
    assert call["shape"] == (2, 16)
    assert call["adjacency"] == [[1], [0]]
    assert call["neighbor_counts"] == [1, 1]
    assert call["rounds"] == 3
    assert call["self_weight"] == pytest.approx(0.5)
    assert call["neighbor_weight"] == pytest.approx(0.5)
    assert any("GRE specialist dispatch:" in step for step in selection_steps)


def test_phase_d_specialist_swarm_features_fall_back_to_semantic_knn(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_semantic_knn")

    class _FakeGraphCrystallizer:
        def __init__(self):
            self.calls = []

        def crystallize_graph(self, node_features, adjacency, neighbor_counts, rounds, self_weight, neighbor_weight):
            self.calls.append(
                {
                    "shape": tuple(node_features.shape),
                    "adjacency": adjacency.tolist(),
                    "neighbor_counts": neighbor_counts.tolist(),
                    "rounds": rounds,
                    "self_weight": self_weight,
                    "neighbor_weight": neighbor_weight,
                }
            )
            return np.asarray(node_features, dtype=np.float32)

        def crystallize_list(self, *_args, **_kwargs):
            raise AssertionError("compatibility_path_should_not_run")

    fake_graph = _FakeGraphCrystallizer()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: fake_graph)
    monkeypatch.setattr(
        kv,
        "_embedding_similarity_matrix",
        lambda _sources, _targets: [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ],
    )

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Reality"},
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "led_focus": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Reality"},
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "led_focus": 0.0,
        },
        {
            "match": {"embedding16": [0.0, 0.0, 1.0] + [0.0] * 13, "galaxy": "Grammar"},
            "candidate_global_idx": 12,
            "graph_neighbors": [],
            "led_focus": 0.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="LHE_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_graph.calls) == 1
    call = fake_graph.calls[0]
    assert call["shape"] == (3, 16)
    assert call["adjacency"] == [[1, 2], [0, 2], [0, 1]]
    assert call["neighbor_counts"] == [2, 2, 2]
    assert call["rounds"] == 3
    assert call["self_weight"] == pytest.approx(0.5)
    assert call["neighbor_weight"] == pytest.approx(0.5)
    assert any("mode=semantic_knn" in step for step in selection_steps)


def test_phase_a1_resonance_field_adjusts_specialist_coherence(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_resonance_field")

    class _FakeGraphCrystallizer:
        def crystallize_graph(self, node_features, adjacency, neighbor_counts, rounds, self_weight, neighbor_weight):
            return np.asarray(node_features, dtype=np.float32)

        def crystallize_list(self, *_args, **_kwargs):
            raise AssertionError("compatibility_path_should_not_run")

    class _FakeResonanceField:
        def __init__(self):
            self.calls = []

        def compute_resonance(self, candidate_embeddings, galaxy_ids, base_scores):
            self.calls.append(
                {
                    "shape": tuple(np.asarray(candidate_embeddings).shape),
                    "galaxy_ids": list(galaxy_ids),
                    "base_scores": list(base_scores),
                }
            )
            return np.asarray(base_scores, dtype=np.float32) + 0.25

    fake_resonance = _FakeResonanceField()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: _FakeGraphCrystallizer())
    monkeypatch.setattr(kv, "get_resonance_field", lambda: fake_resonance)

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Reality"},
            "candidate_global_idx": 10,
            "graph_neighbors": [11],
            "led_focus": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Math"},
            "candidate_global_idx": 11,
            "graph_neighbors": [10],
            "led_focus": 0.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="MMLU_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_resonance.calls) == 1
    call = fake_resonance.calls[0]
    assert call["shape"] == (2, 16)
    assert call["galaxy_ids"] == [6, 5]
    assert local_candidates[0]["cross_galaxy_resonance"] == pytest.approx(call["base_scores"][0] + 0.25)
    assert local_candidates[0]["specialist_coherence"] == pytest.approx(call["base_scores"][0] + 0.25)
    assert "gre_resonance_field" in local_candidates[0]["specialist_worker"]


def test_phase_b4_geometry_router_threads_spatial_signal_into_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_geometry_router")

    class _FakeGeometryRouter:
        def __init__(self):
            self.calls = []

        def compute_relations(self, embeddings_a, embeddings_b):
            arr_a = np.asarray(embeddings_a, dtype=np.float32)
            arr_b = np.asarray(embeddings_b, dtype=np.float32)
            self.calls.append({"shape_a": tuple(arr_a.shape), "shape_b": tuple(arr_b.shape)})
            return np.asarray(
                [
                    [0.8, 0.1, 0.7, 0.7, 0.7, 0.7, 1.0, 0.0, 1.0, 1.0, 0.6, 0.0, 0.9, 0.8, 0.1, 0.01],
                    [0.2, 0.4, 0.1, 0.1, 0.1, 0.1, 1.0, 0.0, 1.0, 1.0, 0.2, 0.0, 0.4, 0.3, 0.8, 0.64],
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    fake_geometry = _FakeGeometryRouter()
    monkeypatch.setattr(kv, "get_geometry_router", lambda: fake_geometry)

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Math", "category": "template", "confidence": 0.8},
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "similarity": 0.6,
            "option_similarity": 0.6,
            "program": {"id": "reasoning_math_template_match_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Grammar", "category": "rule", "confidence": 0.7},
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "similarity": 0.4,
            "option_similarity": 0.4,
            "program": {"id": "reasoning_math_template_match_top1"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="MATH_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_geometry.calls) == 1
    assert fake_geometry.calls[0]["shape_a"] == (2, 16)
    assert fake_geometry.calls[0]["shape_b"] == (2, 16)
    assert local_candidates[0]["specialist_geometry"] > local_candidates[1]["specialist_geometry"]
    assert "gre_geometry_router" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[0],
        primary_program_id="reasoning_math_template_match_top1",
        target_galaxies=["Math", "Grammar"],
        task_type="MATH_TASK",
        domain_hint="math",
    )
    assert kv._gpu_scalar_literal(local_candidates[0]["specialist_geometry"]) in expr


def test_phase_b5_temporal_reasoning_threads_path_signal_into_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_temporal_reasoning")

    class _FakeTemporalReasoning:
        def __init__(self):
            self.calls = []

        def compute_patterns(self, sequence):
            arr = np.asarray(sequence, dtype=np.float32)
            self.calls.append({"shape": tuple(arr.shape)})
            patterns = np.zeros((24,), dtype=np.float32)
            patterns[8:12] = 0.9
            patterns[12] = 1.0
            patterns[13] = 1.0
            patterns[14:18] = 0.5
            patterns[18] = 0.8
            patterns[19] = 0.6
            patterns[20] = 1.0
            patterns[21] = 0.75
            patterns[22] = 0.1
            patterns[23] = 0.2
            return patterns

    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    monkeypatch.setattr(kv, "get_geometry_router", lambda: None)
    fake_temporal = _FakeTemporalReasoning()
    monkeypatch.setattr(kv, "get_temporal_reasoning", lambda: fake_temporal)

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Math", "category": "template", "confidence": 0.8},
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "led_path_position": 0,
            "similarity": 0.6,
            "option_similarity": 0.6,
            "program": {"id": "reasoning_math_template_match_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {"embedding16": [0.5, 0.5] + [0.0] * 14, "galaxy": "Math", "category": "template", "confidence": 0.85},
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "led_path_position": 1,
            "similarity": 0.7,
            "option_similarity": 0.7,
            "program": {"id": "reasoning_math_template_match_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Grammar", "category": "rule", "confidence": 0.7},
            "candidate_global_idx": 12,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "led_path_position": -1,
            "similarity": 0.4,
            "option_similarity": 0.4,
            "program": {"id": "reasoning_math_template_match_top1"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="MATH_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_temporal.calls) == 1
    assert fake_temporal.calls[0]["shape"] == (2, 16)
    assert local_candidates[1]["specialist_temporal"] > local_candidates[0]["specialist_temporal"] > 0.0
    assert local_candidates[2]["specialist_temporal"] == pytest.approx(0.0)
    assert "gre_temporal_reasoning" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[1],
        primary_program_id="reasoning_math_template_match_top1",
        target_galaxies=["Math", "Grammar"],
        task_type="MATH_TASK",
        domain_hint="math",
    )
    assert kv._gpu_scalar_literal(local_candidates[1]["specialist_temporal"]) in expr


def test_phase_b6_fractal_emitter_threads_self_similarity_into_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_fractal_emitter")

    class _FakeFractalEmitter:
        def __init__(self):
            self.calls = []

        def compute_self_similarity(self, features, num_scales=3):
            arr = np.asarray(features, dtype=np.float32)
            self.calls.append({"shape": tuple(arr.shape), "num_scales": int(num_scales)})
            return np.asarray([0.9, 0.1], dtype=np.float32)

    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    monkeypatch.setattr(kv, "get_geometry_router", lambda: None)
    monkeypatch.setattr(kv, "get_temporal_reasoning", lambda: None)
    fake_fractal = _FakeFractalEmitter()
    monkeypatch.setattr(kv, "get_fractal_emitter", lambda: fake_fractal)

    local_candidates = [
        {
            "match": {"embedding16": [1.0] + [0.0] * 15, "galaxy": "Drawing", "category": "template", "confidence": 0.8},
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "led_path_position": 0,
            "similarity": 0.6,
            "option_similarity": 0.6,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {"embedding16": [0.0, 1.0] + [0.0] * 14, "galaxy": "Grammar", "category": "rule", "confidence": 0.7},
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "led_path_position": -1,
            "similarity": 0.4,
            "option_similarity": 0.4,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="ARC_TASK",
        path={"label": "primary"},
        selection_steps=selection_steps,
    )

    assert len(fake_fractal.calls) == 1
    assert fake_fractal.calls[0]["shape"] == (2, 16)
    assert fake_fractal.calls[0]["num_scales"] == 4
    assert local_candidates[0]["specialist_fractal"] > local_candidates[1]["specialist_fractal"]
    assert "gre_fractal_emitter" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[0],
        primary_program_id="reasoning_arc_grid_transform_top1",
        target_galaxies=["Drawing", "Grammar"],
        task_type="ARC_TASK",
        domain_hint="visual",
    )
    assert kv._gpu_scalar_literal(local_candidates[0]["specialist_fractal"]) in expr


def test_phase_b7_cognitive_executive_blends_swarm_trust_weights(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_cognitive_executive")

    class _Diag:
        def __init__(self):
            self.resonance_matrix = np.asarray(
                [
                    [1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6],
                    [0.9, 1.0, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5],
                    [0.8, 0.7, 1.0, 0.6, 0.5, 0.5, 0.4, 0.4],
                    [0.8, 0.7, 0.6, 1.0, 0.5, 0.5, 0.4, 0.4],
                    [0.7, 0.6, 0.5, 0.5, 1.0, 0.4, 0.3, 0.3],
                    [0.7, 0.6, 0.5, 0.5, 0.4, 1.0, 0.3, 0.3],
                    [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 1.0, 0.2],
                    [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 1.0],
                ],
                dtype=np.float32,
            )
            self.chain_norms = np.asarray([2.5, 2.0, 1.7, 1.6, 1.3, 1.2, 1.0, 0.9, 0.5], dtype=np.float32)

    class _FakeSwarm:
        def __init__(self):
            self.exec_calls = 0
            self.diag = _Diag()

        def execute_swarm(self, *_args, **_kwargs):
            self.exec_calls += 1
            return np.zeros((128,), dtype=np.float32), None, np.asarray([0.30, 0.20, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03], dtype=np.float32)

        def get_chain_diagnostics(self):
            return self.diag

    class _FakeExecutive:
        def __init__(self):
            self.calls = []

        def compute_trust_weights(self, resonance_matrix, chain_norms):
            self.calls.append(
                {
                    "matrix_shape": tuple(np.asarray(resonance_matrix).shape),
                    "norms_shape": tuple(np.asarray(chain_norms).shape),
                }
            )
            return np.asarray([0.40, 0.18, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03], dtype=np.float32), 0.5

    fake_swarm = _FakeSwarm()
    fake_executive = _FakeExecutive()
    monkeypatch.setattr(kv, "get_swarm_bridge", lambda: fake_swarm)
    monkeypatch.setattr(kv, "get_cognitive_executive", lambda: fake_executive)

    selection_steps: list[str] = []
    weights = kv._dispatch_swarm_weights(
        query_embedding=[1.0] + [0.0] * 15,
        paths=[
            {"program_id": "p0"},
            {"program_id": "p1"},
            {"program_id": "p2"},
        ],
        selection_steps=selection_steps,
    )

    assert fake_swarm.exec_calls == 1
    assert len(fake_executive.calls) == 1
    assert fake_executive.calls[0]["matrix_shape"] == (8, 8)
    assert fake_executive.calls[0]["norms_shape"] == (8,)
    assert len(weights) == 3
    assert weights[0] > weights[1] > weights[2]
    assert any("GRE cognitive executive:" in step for step in selection_steps)


def test_phase_a3_atomic_fission_threads_compositional_consistency_into_math_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_atomic_fission")

    class _FakeAtomicFissionFusion:
        def __init__(self):
            self.calls = []

        def decompose(self, compound, atoms):
            self.calls.append(
                {
                    "compound": list(compound),
                    "shape": tuple(np.asarray(atoms).shape),
                }
            )
            return np.asarray(compound, dtype=np.float32), 0.75

    fake_bridge = _FakeAtomicFissionFusion()
    monkeypatch.setattr(kv, "get_atomic_fission_fusion", lambda: fake_bridge)

    candidate = {
        "match": {
            "embedding16": [1.0] + [0.0] * 15,
            "galaxy": "Math",
            "category": "template",
            "confidence": 0.9,
            "gpu_has_template_ref": 1.0,
        },
        "program": {"id": "reasoning_word_problem_fission"},
        "gsm8k_mode": 1.0,
        "gsm8k_template_focus": 1.0,
        "gsm8k_context": {
            "pattern_rows": [
                {"id": "grammar_pattern", "embedding16": [0.0, 1.0] + [0.0] * 14},
            ],
            "quantity_role_candidates": [
                {"id": "number_anchor", "embedding16": [0.0, 0.0, 1.0] + [0.0] * 13},
            ],
            "number_ids": [],
        },
        "similarity": 0.5,
        "option_similarity": 0.5,
        "specialist_resonance": 0.5,
        "specialist_coherence": 0.5,
        "galaxy_weight": 1.0,
    }
    selection_steps: list[str] = []

    kv._apply_atomic_compositional_consistency(
        local_candidates=[candidate],
        task_type="MATH_TASK",
        selection_steps=selection_steps,
    )

    assert len(fake_bridge.calls) == 1
    assert fake_bridge.calls[0]["shape"] == (2, 16)
    assert candidate["compositional_consistency"] == pytest.approx(0.75)
    assert candidate["compositional_atom_count"] == 2
    assert any("Atomic fission/fusion:" in step for step in selection_steps)

    expr = kv._build_gpu_candidate_score_expression(
        candidate=candidate,
        primary_program_id="reasoning_word_problem_fission",
        target_galaxies=["Math", "Grammar"],
        task_type="MATH_TASK",
        domain_hint="math",
    )
    assert kv._gpu_scalar_literal(0.75) in expr


def test_track_a_semantic_entities_capture_frequency_rate_and_goal():
    navigator = NavigatorSpecialist()

    route = navigator._fusion_reading_path(
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        {
            "specialist": "math",
            "domain": "math",
            "galaxy_names": ["Math", "Grammar"],
            "goal_type_family": "gsm8k",
        },
    )

    fusion = route["fusion_parse"]
    semantic_entities = fusion["semantic_entities"]
    assert any(
        float(entity.get("value", 0.0)) == pytest.approx(3.0)
        and str(entity.get("role", "")).strip() == "count"
        and str(entity.get("unit", "")).strip() == "sprint"
        and str(entity.get("scope", "")).strip() == "per_session"
        for entity in semantic_entities
    )
    assert any(
        float(entity.get("value", 0.0)) == pytest.approx(3.0)
        and str(entity.get("role", "")).strip() == "frequency"
        and str(entity.get("unit", "")).strip() == "session"
        and str(entity.get("scope", "")).strip() == "per_week"
        for entity in semantic_entities
    )
    assert any(
        float(entity.get("value", 0.0)) == pytest.approx(60.0)
        and str(entity.get("role", "")).strip() == "rate"
        and str(entity.get("unit", "")).strip() == "meter"
        and str(entity.get("scope", "")).strip() == "per_sprint"
        for entity in semantic_entities
    )
    assert fusion["goal_entity"]["role"] == "goal"
    assert fusion["goal_entity"]["unit"] == "meter"
    assert fusion["goal_entity"]["scope"] == "per_week"


def test_track_a_reference_resolution_recovers_half_that_much():
    navigator = NavigatorSpecialist()

    route = navigator._fusion_reading_path(
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of white fiber are needed?",
        {
            "specialist": "math",
            "domain": "math",
            "galaxy_names": ["Math", "Grammar"],
            "goal_type_family": "gsm8k",
        },
    )

    semantic_entities = route["fusion_parse"]["semantic_entities"]
    reference_entity = next(
        entity
        for entity in semantic_entities
        if str(entity.get("reference", "")).strip() == "half"
    )

    assert float(reference_entity["resolved_value"]) == pytest.approx(1.0)
    assert float(reference_entity["reference_source"]["value"]) == pytest.approx(2.0)
    assert str(reference_entity.get("unit", "")).strip() == "bolt"


def test_track_a_dimensional_consistency_boosts_existing_compositional_signal(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_track_a_dimensional")

    class _FakeAtomicFissionFusion:
        def decompose(self, compound, atoms):
            return np.asarray(compound, dtype=np.float32), 0.5

    monkeypatch.setattr(kv, "get_atomic_fission_fusion", lambda: _FakeAtomicFissionFusion())

    candidate = {
        "match": {
            "embedding16": [1.0] + [0.0] * 15,
            "galaxy": "Math",
            "category": "template",
        },
        "gsm8k_mode": 1.0,
        "gsm8k_template_focus": 1.0,
        "gsm8k_context": {
            "semantic_entities": [
                {"value": 3.0, "role": "count", "unit": "sprints", "scope": "per_session"},
                {"value": 3.0, "role": "frequency", "unit": "sessions", "scope": "per_week"},
                {"value": 60.0, "role": "rate", "unit": "meters", "scope": "per_sprint"},
            ],
            "goal_entity": {"role": "goal", "unit": "meters", "scope": "per_week"},
        },
    }

    kv._apply_atomic_compositional_consistency(
        local_candidates=[candidate],
        task_type="MATH_TASK",
        selection_steps=[],
    )

    assert candidate["compositional_consistency"] == pytest.approx(0.65)
    assert candidate["compositional_dimensional_consistency"] == pytest.approx(1.0)


def test_math_structural_override_promotes_numeric_answer_over_non_numeric_consensus(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_structural_override_non_numeric")

    records = [
        {
            "option_text": "operation_pattern_multiplication",
            "path_score": 1.94,
            "weighted_support": 1.34,
            "support_count": 1,
            "best_structural_score": 0.0,
            "candidate": {
                "path_score": 1.94,
                "gsm8k_consensus_weight": 1.34,
                "gsm8k_consensus_support": 1,
                "gsm8k_best_structural_score": 0.0,
                "compositional_consistency": 0.30,
            },
        },
        {
            "option_text": "3",
            "path_score": 1.90,
            "weighted_support": 1.30,
            "support_count": 1,
            "best_structural_score": 0.71,
            "candidate": {
                "path_score": 1.90,
                "gsm8k_consensus_weight": 1.30,
                "gsm8k_consensus_support": 1,
                "gsm8k_best_structural_score": 0.71,
                "compositional_consistency": 0.30,
            },
        },
        {
            "option_text": "0.75",
            "path_score": 1.33,
            "weighted_support": 0.92,
            "support_count": 1,
            "best_structural_score": 0.75,
            "candidate": {
                "path_score": 1.33,
                "gsm8k_consensus_weight": 0.92,
                "gsm8k_consensus_support": 1,
                "gsm8k_best_structural_score": 0.75,
                "compositional_consistency": 0.30,
            },
        },
    ]

    override = kv._math_structural_override_record(records)

    assert override is not None
    assert override["option_text"] == "3"


def test_math_structural_override_promotes_better_numeric_answer_over_noisy_consensus(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_structural_override_numeric")

    records = [
        {
            "option_text": "30",
            "path_score": 1.34,
            "weighted_support": 3.26,
            "support_count": 3,
            "best_structural_score": 0.86,
            "candidate": {
                "path_score": 1.34,
                "gsm8k_consensus_weight": 3.26,
                "gsm8k_consensus_support": 3,
                "gsm8k_best_structural_score": 0.86,
                "compositional_consistency": 1.0,
                "compositional_dimensional_consistency": 1.0,
            },
        },
        {
            "option_text": "260",
            "path_score": 1.62,
            "weighted_support": 1.34,
            "support_count": 1,
            "best_structural_score": 0.93,
            "candidate": {
                "path_score": 1.62,
                "gsm8k_consensus_weight": 1.34,
                "gsm8k_consensus_support": 1,
                "gsm8k_best_structural_score": 0.93,
                "compositional_consistency": 1.0,
                "compositional_dimensional_consistency": 1.0,
            },
        },
    ]

    override = kv._math_structural_override_record(records)

    assert override is not None
    assert override["option_text"] == "260"


def test_math_halting_structural_override_halts_single_worker_numeric_answer(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_halting_override")

    class _FakeHaltingGate:
        def analyze_scores(self, ordered_scores, candidate_hashes, **kwargs):
            return (
                np.asarray([1, 0, 1, 0], dtype=np.uint32),
                np.asarray([2.01, 0.04, 1.0], dtype=np.float32),
            )

    monkeypatch.setattr(kv, "get_halting_gate", lambda: _FakeHaltingGate())

    override = {
        "option_text": "3",
        "path_score": 1.90,
        "candidate": {
            "path_score": 1.90,
            "gsm8k_best_structural_score": 0.71,
            "compositional_consistency": 0.30,
        },
    }
    selection_steps: list[str] = []

    converged = kv._halting_gate_converged(
        task_type="MATH_TASK",
        task={"type": "MATH_TASK", "competition": "GSM8K"},
        path_scores=[2.01, 1.90],
        candidate_ids=["operation_pattern_multiplication", "3"],
        selection_steps=selection_steps,
        gsm8k_structural_override=override,
    )

    assert converged is True
    assert any("GSM8K structural override:" in step for step in selection_steps)


def test_phase_track1_strict_rules_produce_definite_proof_tags():
    resolver = DefeasibleResolver()

    verdicts, proof_tags = resolver.resolve(
        conclusions=np.asarray([[0.8]], dtype=np.float32),
        rule_strengths=np.asarray([1], dtype=np.int8),
        superiority=np.asarray([[np.uint32(0xFFFFFFFF)]], dtype=np.uint32),
        num_workers=1,
        num_candidates=1,
        max_superiors=1,
    )

    assert verdicts.shape == (1,)
    assert verdicts[0] == pytest.approx(0.8, abs=1e-5)
    assert int(proof_tags[0]) == 10  # D=+1, d=+1


def test_phase_track1_defeater_blocks_defeasible_verdict():
    resolver = DefeasibleResolver()

    verdicts, proof_tags = resolver.resolve(
        conclusions=np.asarray([[0.9], [0.8]], dtype=np.float32),
        rule_strengths=np.asarray([-1, 0], dtype=np.int8),
        superiority=np.asarray(
            [
                [np.uint32(0xFFFFFFFF)],
                [np.uint32(0xFFFFFFFF)],
            ],
            dtype=np.uint32,
        ),
        num_workers=2,
        num_candidates=1,
        max_superiors=1,
    )

    assert verdicts[0] == pytest.approx(0.0, abs=1e-6)
    assert int(proof_tags[0]) == 5  # D=0, d=0


def test_phase_track1_defeasible_verdict_threads_into_scoring_expression(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_defeasible_expr")

    candidate = {
        "match": {
            "embedding16": [1.0] + [0.0] * 15,
            "galaxy": "Grammar",
            "category": "rule",
            "confidence": 0.8,
        },
        "program": {"id": "reasoning_word_problem_chain"},
        "similarity": 0.5,
        "option_similarity": 0.5,
        "galaxy_weight": 1.0,
        "specialist_intra_defeasible": 0.3,
        "specialist_defeasible_verdict": 0.6,
    }

    expr = kv._build_gpu_candidate_score_expression(
        candidate=candidate,
        primary_program_id="reasoning_word_problem_chain",
        target_galaxies=["Grammar", "Math"],
        task_type="MATH_TASK",
        domain_hint="math",
    )

    assert kv._gpu_scalar_literal(0.3) in expr
    assert kv._gpu_scalar_literal(0.6) in expr


def test_triple_defeasible_stage1_reduces_defeated_path_weight(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_triple_def_stage1")

    def _metadata(rule_id: str):
        if rule_id == "strict_rule":
            return {"rule_strength": 1, "superior_to": ["defeasible_rule"], "trust_weight": 1.0}
        if rule_id == "defeasible_rule":
            return {"rule_strength": 0, "superior_to": [], "trust_weight": 1.0}
        return {}

    monkeypatch.setattr(kv, "_grammar_rule_metadata", _metadata)

    paths = [
        {"program_id": "strict_rule", "label": "strict"},
        {"program_id": "defeasible_rule", "label": "defeasible"},
    ]
    swarm_weights = [1.4, 1.1]
    selection_steps: list[str] = []

    kv._apply_early_defeasible_gate(
        paths=paths,
        swarm_weights=swarm_weights,
        selection_steps=selection_steps,
    )

    assert paths[0]["path_defeasible_tag"] == 1
    assert paths[1]["path_defeasible_tag"] == -1
    assert swarm_weights[0] == pytest.approx(1.4)
    assert swarm_weights[1] == pytest.approx(0.0)
    assert any("stage1" in step for step in selection_steps)


def test_triple_defeasible_stage2_intra_path_strict_beats_defeasible(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_triple_def_stage2")

    def _metadata(rule_id: str):
        if rule_id == "strict_rule":
            return {"rule_strength": 1, "superior_to": ["defeasible_rule"], "trust_weight": 1.0}
        if rule_id == "defeasible_rule":
            return {"rule_strength": 0, "superior_to": [], "trust_weight": 1.0}
        return {}

    monkeypatch.setattr(kv, "_grammar_rule_metadata", _metadata)

    path = {"program_id": "strict_rule", "label": "primary", "path_defeasible_tag": 1}
    local_candidates = [
        {
            "match": {
                "id": "strict_rule",
                "galaxy": "Grammar",
                "embedding16": [1.0] + [0.0] * 15,
            },
            "program": {"id": "strict_rule"},
            "specialist_coherence": 0.8,
        },
        {
            "match": {
                "id": "defeasible_rule",
                "galaxy": "Grammar",
                "embedding16": [0.0, 1.0] + [0.0] * 14,
            },
            "program": {"id": "defeasible_rule"},
            "specialist_coherence": 0.8,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_intra_path_defeasible(
        local_candidates=local_candidates,
        path=path,
        task_type="MATH_TASK",
        selection_steps=selection_steps,
    )

    assert local_candidates[0]["specialist_intra_defeasible"] > 0.0
    assert local_candidates[1]["specialist_intra_defeasible"] < 0.0
    assert local_candidates[0]["specialist_intra_defeasible"] > local_candidates[1]["specialist_intra_defeasible"]
    assert any("stage2" in step for step in selection_steps)


def test_triple_defeasible_stage3_honors_upstream_defeat(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_triple_def_stage3")
    neutral_proof_tag = kv._pack_defeasible_proof_tag(0, 0)

    record = {
        "path_score": 0.9,
        "candidate": {
            "match": {"id": "candidate_a", "galaxy": "Math"},
            "program": {"id": "gsm_consume_from_total"},
            "path": {"path_defeasible_tag": -1},
        },
    }
    selection_steps: list[str] = []

    kv._apply_defeasible_specialist_resolution(
        records=[record],
        task_type="MATH_TASK",
        gsm8k_mode=False,
        selection_steps=selection_steps,
    )

    assert record["specialist_defeasible_verdict"] == pytest.approx(0.0)
    assert int(record["specialist_proof_tag"]) == int(neutral_proof_tag)
    assert record["path_score"] == pytest.approx(0.9)


def test_triple_defeasible_backward_compat(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_triple_def_compat")

    paths = [
        {"program_id": "gsm_consume_from_total", "label": "a"},
        {"program_id": "gsm_answer_final_stack", "label": "b"},
    ]
    swarm_weights = [1.3, 1.1]
    original_weights = list(swarm_weights)
    selection_steps: list[str] = []

    kv._apply_early_defeasible_gate(
        paths=paths,
        swarm_weights=swarm_weights,
        selection_steps=selection_steps,
    )

    assert swarm_weights == pytest.approx(original_weights)
    assert [path["path_defeasible_tag"] for path in paths] == [1, 1]

    local_candidates = [
        {
            "match": {"id": "entry_a", "galaxy": "Math", "embedding16": [1.0] + [0.0] * 15},
            "program": {"id": "gsm_consume_from_total"},
            "specialist_coherence": 0.4,
        },
        {
            "match": {"id": "entry_b", "galaxy": "Grammar", "embedding16": [0.0, 1.0] + [0.0] * 14},
            "program": {"id": "gsm_answer_final_stack"},
            "specialist_coherence": 0.2,
        },
    ]

    kv._apply_intra_path_defeasible(
        local_candidates=local_candidates,
        path=paths[0],
        task_type="MATH_TASK",
        selection_steps=selection_steps,
    )

    assert all(candidate["specialist_intra_defeasible"] == pytest.approx(0.0) for candidate in local_candidates)

    record_a = {
        "option_text": "A",
        "path_score": 0.7,
        "candidate": {
            "match": {"id": "a", "galaxy": "Math"},
            "program": {"id": "gsm_consume_from_total"},
            "path": {"path_defeasible_tag": 1},
        },
    }
    record_b = {
        "option_text": "B",
        "path_score": 0.2,
        "candidate": {
            "match": {"id": "b", "galaxy": "Grammar"},
            "program": {"id": "gsm_answer_final_stack"},
            "path": {"path_defeasible_tag": 1},
        },
    }

    kv._apply_defeasible_specialist_resolution(
        records=[record_a, record_b],
        task_type="MMLU_TASK",
        gsm8k_mode=False,
        selection_steps=selection_steps,
    )

    assert record_a["specialist_defeasible_verdict"] == pytest.approx(0.7)
    assert record_b["specialist_defeasible_verdict"] == pytest.approx(0.2)


def test_triple_defeasible_defers_stage1_and_stage2_for_mmlu(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_triple_def_mmlu_defer")

    paths = [{"program_id": "strict_rule", "label": "mmlu_path"}]
    swarm_weights = [1.25]
    selection_steps: list[str] = []

    kv._apply_early_defeasible_gate(
        task_type="MMLU_TASK",
        paths=paths,
        swarm_weights=swarm_weights,
        selection_steps=selection_steps,
    )

    assert swarm_weights == pytest.approx([1.25])
    assert "path_defeasible_tag" not in paths[0]

    local_candidates = [
        {
            "match": {"id": "entry_a", "galaxy": "Reality", "embedding16": [1.0] + [0.0] * 15},
            "program": {"id": "strict_rule"},
            "specialist_coherence": 0.4,
        }
    ]
    kv._apply_intra_path_defeasible(
        local_candidates=local_candidates,
        path={"program_id": "strict_rule", "label": "mmlu_path"},
        task_type="MMLU_TASK",
        selection_steps=selection_steps,
    )

    assert local_candidates[0]["specialist_intra_defeasible"] == pytest.approx(0.0)
    assert any("deferred_for_mmlu" in step for step in selection_steps)


def test_mmlu_reset_query_session_clears_navigation_runtime_state(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_session_reset")

    class _FakeSubstrate:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeGraph:
        def __init__(self):
            self.reset_calls = 0

        def reset_traversal_state(self):
            self.reset_calls += 1

    fake_substrate = _FakeSubstrate()
    fake_graph = _FakeGraph()
    kv._query_head_substrate = fake_substrate
    kv._semantic_csr_graph = fake_graph
    kv._led_pathfinder = object()
    kv._gpu_reasoning_programs["math"] = {"id": "math"}
    kv._query_sequence = 9

    kv.reset_query_session()

    assert fake_substrate.closed is True
    assert fake_graph.reset_calls == 1
    assert kv._query_head_substrate is None
    assert kv._led_pathfinder is None
    assert kv._gpu_reasoning_programs == {}
    assert kv._query_sequence == 0


def test_mmlu_subject_seed_bias_promotes_subject_matched_candidate(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_seed_bias")

    catalog = [
        {
            "id": "generic_math",
            "galaxy": "Math",
            "embedding16": [1.0] + [0.0] * 15,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Math"),
            "confidence": 0.5,
        },
        {
            "id": "subject_reality",
            "galaxy": "Reality",
            "embedding16": [0.0, 1.0] + [0.0] * 14,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Reality"),
            "confidence": 0.5,
        },
    ]

    class _FakeSubstrate:
        def morton_locate(self, **kwargs):
            return np.asarray([0, 1], dtype=np.uint32)

        def frustum_visible(self, **kwargs):
            return np.asarray([0, 1], dtype=np.uint32)

        def lod_metrics(self, **kwargs):
            return {
                0: (0.5, 5),
                1: (0.5, 5),
            }

    class _FakeGraph:
        def select_seed_nodes(self, **kwargs):
            return [(0, 0.6), (1, 0.4)]

        def extract_local_kernel(self, **kwargs):
            return (
                [0, 1],
                np.asarray([0, 0, 0], dtype=np.uint32),
                np.asarray([], dtype=np.uint32),
                np.asarray([], dtype=np.uint32),
            )

    monkeypatch.setattr(kv, "get_query_head_substrate", lambda: _FakeSubstrate())
    monkeypatch.setattr(kv, "get_gpu_galaxy_catalog", lambda: list(catalog))
    monkeypatch.setattr(kv, "get_semantic_csr_graph", lambda: _FakeGraph())
    monkeypatch.setattr(kv, "get_led_pathfinder", lambda: None)
    monkeypatch.setattr(kv, "_embedding_similarities", lambda reference, candidates: [0.6, 0.4])
    monkeypatch.setattr(
        kv,
        "_subject_anchor_match_score",
        lambda entry, subject_hint, match_mode="mmlu": 1.0 if entry.get("id") == "subject_reality" else 0.0,
    )

    selection_steps: list[str] = []
    candidates = kv._compose_head_navigation_candidates(
        binding={"galaxies": list(kv.GPU_MMLU_TARGET_GALAXIES)},
        target_galaxies=list(kv.GPU_MMLU_TARGET_GALAXIES),
        galaxy_weights=None,
        reasoning_program_id=kv.GPU_CHAT_REASONING_PROGRAM_ID,
        query_embedding=[1.0] + [0.0] * 15,
        task_type="MMLU_TASK",
        selection_steps=selection_steps,
        task={"type": "MMLU_TASK", "task_id": "mmlu_seed_bias"},
        query_text="college physics anchor question",
        domain_hint="college_physics",
    )

    assert candidates
    assert candidates[0]["match"]["id"] == "subject_reality"
    assert candidates[0]["subject_anchor_focus"] == pytest.approx(1.0)
    assert any("MMLU seed bias: 1/2 subject-matched candidates" in step for step in selection_steps)


def test_mmlu_priority_seed_injection_adds_subject_matched_reality_anchor(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_seed_injection")

    catalog = [
        {
            "id": "generic_math",
            "galaxy": "Math",
            "embedding16": [1.0] + [0.0] * 15,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Math"),
            "confidence": 0.5,
        },
        {
            "id": "subject_reality",
            "galaxy": "Reality",
            "embedding16": [0.0, 1.0] + [0.0] * 14,
            "gpu_galaxy_index": kv._gpu_galaxy_index("Reality"),
            "confidence": 0.7,
        },
    ]

    class _FakeSubstrate:
        def morton_locate(self, **kwargs):
            return np.asarray([0], dtype=np.uint32)

        def frustum_visible(self, **kwargs):
            return np.asarray([0], dtype=np.uint32)

        def lod_metrics(self, **kwargs):
            return {
                0: (0.5, 5),
                1: (0.5, 5),
            }

    class _FakeGraph:
        def select_seed_nodes(self, **kwargs):
            return [(0, 0.6)]

        def extract_local_kernel(self, **kwargs):
            return (
                [0, 1],
                np.asarray([0, 0, 0], dtype=np.uint32),
                np.asarray([], dtype=np.uint32),
                np.asarray([], dtype=np.uint32),
            )

    monkeypatch.setattr(kv, "get_query_head_substrate", lambda: _FakeSubstrate())
    monkeypatch.setattr(kv, "get_gpu_galaxy_catalog", lambda: list(catalog))
    monkeypatch.setattr(kv, "get_semantic_csr_graph", lambda: _FakeGraph())
    monkeypatch.setattr(kv, "get_led_pathfinder", lambda: None)
    monkeypatch.setattr(kv, "_embedding_similarities", lambda reference, candidates: [0.6, 0.4])
    monkeypatch.setattr(
        kv,
        "_subject_anchor_match_score",
        lambda entry, subject_hint, match_mode="mmlu": 1.0 if entry.get("id") == "subject_reality" else 0.0,
    )

    selection_steps: list[str] = []
    candidates = kv._compose_head_navigation_candidates(
        binding={"galaxies": list(kv.GPU_MMLU_TARGET_GALAXIES)},
        target_galaxies=list(kv.GPU_MMLU_TARGET_GALAXIES),
        galaxy_weights=None,
        reasoning_program_id=kv.GPU_CHAT_REASONING_PROGRAM_ID,
        query_embedding=[1.0] + [0.0] * 15,
        task_type="MMLU_TASK",
        selection_steps=selection_steps,
        task={"type": "MMLU_TASK", "task_id": "mmlu_seed_injection"},
        query_text="college physics anchor question",
        domain_hint="college_physics",
    )

    candidate_ids = {candidate["match"]["id"] for candidate in candidates}
    assert "subject_reality" in candidate_ids
    assert any("MMLU priority seed injection: 1 Reality anchors" in step for step in selection_steps)


def test_nsi_routing_bias_holds_on_zero_outcome():
    node = SpecialistBase(name="root", domain="generic")
    node.routing_bias["child"] = 0.42

    node.update_routing_bias("child", ternary_outcome=0)

    assert node.routing_bias["child"] == pytest.approx(0.42)


def test_nsi_mark_query_ternary_uncertain_count():
    node = SpecialistBase(name="root", domain="generic")

    node.mark_query(ternary_outcome=0)

    assert node.query_count == 1
    assert node.success_count == 0
    assert node.failure_count == 0
    assert node.uncertain_count == 1
    assert node.exploration_pressure == 1


def test_nsi_routing_topology_no_bias_on_zero(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nsi_topology_zero")
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        kv.navigator_specialist.router,
        "adjust_specialist_bias",
        lambda specialist, delta: calls.append((str(specialist), float(delta))),
    )

    kv.navigator_specialist.learn_routing_topology(
        "uncertain specialist route",
        specialist="GrammarSpecialist",
        ternary_outcome=0,
    )

    signature = kv.navigator_specialist._query_signature("uncertain specialist route")
    bucket = kv.navigator_specialist.routing_topology[signature]["GrammarSpecialist"]
    assert bucket["success"] == 0
    assert bucket["failure"] == 0
    assert bucket["uncertain"] == 1
    assert calls == []


def test_nsi_consolidation_uses_verdict_trit(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nsi_consolidation")
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        kv.trm_navigator,
        "learn_from_feedback",
        lambda **kwargs: captured.append(dict(kwargs)),
    )

    summary = kv.trm_navigator.consolidate_weights_from_events(
        [
            {
                "type": "defeasible_verdict",
                "data": {
                    "specialist": "math",
                    "query": "factorial guard",
                    "verdict_trit": -1,
                    "confidence": 0.73,
                    "was_defeated_by": "strict_factorial_axiom",
                },
                "confidence": 0.73,
            }
        ]
    )

    assert summary["updated_count"] == 1
    assert captured
    assert captured[0]["ternary_outcome"] == -1
    assert captured[0]["defeat_source"] == "strict_factorial_axiom"


def test_nsi_grammar_detector_exploratory_polarity(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nsi_exploratory")
    detector = kv.trm_navigator.execution_grammar_detector
    assert detector is not None

    event = {
        "tool_id": "tool_explore_probe",
        "query_context": "explore conflict zone alternatives",
        "domain_hint": "multimodal",
        "outcome": 0,
        "quality_signal": 0.55,
        "timestamp_us": 123,
        "chain_tool_ids": [
            "tool_alpha_explore",
            "tool_beta_explore",
        ],
        "chain_runtime_statuses": [
            "ptx_bridge_available",
            "ptx_bridge_available",
        ],
    }

    summary_a = detector.observe_event(event)
    summary_b = detector.observe_event({**event, "timestamp_us": 124})
    summary_c = detector.observe_event({**event, "timestamp_us": 125})

    assert summary_a["updated_patterns"]
    assert summary_b["updated_patterns"]
    assert summary_c["updated_patterns"]
    pattern_key = summary_a["updated_patterns"][0]
    record = detector._state["patterns"][pattern_key]
    assert record["polarity"] == "exploratory"
    promoted = detector._state["promoted_rules"][pattern_key]
    assert promoted["polarity"] == "exploratory"
    assert promoted["live_inserted"] is False
    target = promoted["entry"]
    assert target["semantics"]["source"] == "auto_detected_exploratory"
    assert target["semantics"]["contrastive_recommendation"] == "explore_alternatives"
    assert target["semantics"]["ternary_confidence"] == 0


def test_nsi_backward_compat_bool_still_works():
    via_bool = SpecialistBase(name="bool_node", domain="generic")
    via_ternary = SpecialistBase(name="ternary_node", domain="generic")
    via_bool.routing_bias["child"] = 0.5
    via_ternary.routing_bias["child"] = 0.5

    via_bool.mark_query(success=True)
    via_ternary.mark_query(ternary_outcome=1)
    via_bool.mark_query(success=False)
    via_ternary.mark_query(ternary_outcome=-1)
    via_bool.update_routing_bias("child", success=True)
    via_ternary.update_routing_bias("child", ternary_outcome=1)
    via_bool.update_routing_bias("child", success=False)
    via_ternary.update_routing_bias("child", ternary_outcome=-1)

    assert via_bool.query_count == via_ternary.query_count
    assert via_bool.success_count == via_ternary.success_count
    assert via_bool.failure_count == via_ternary.failure_count
    assert via_bool.uncertain_count == via_ternary.uncertain_count
    assert via_bool.routing_bias["child"] == pytest.approx(via_ternary.routing_bias["child"])


def test_phase_track1_defeasible_compatibility_mode_preserves_raw_scores(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_defeasible_compat")

    record_a = {
        "option_text": "A",
        "path_score": 0.7,
        "candidate": {
            "match": {"id": "a", "galaxy": "Math"},
            "program": {"id": "gsm_consume_from_total"},
        },
    }
    record_b = {
        "option_text": "B",
        "path_score": 0.2,
        "candidate": {
            "match": {"id": "b", "galaxy": "Grammar"},
            "program": {"id": "gsm_answer_final_stack"},
        },
    }
    selection_steps: list[str] = []

    kv._apply_defeasible_specialist_resolution(
        records=[record_a, record_b],
        task_type="MMLU_TASK",
        gsm8k_mode=False,
        selection_steps=selection_steps,
    )

    assert record_a["specialist_defeasible_verdict"] == pytest.approx(0.7)
    assert record_b["specialist_defeasible_verdict"] == pytest.approx(0.2)
    assert record_a["path_score"] > record_b["path_score"]
    assert any("compatibility mode" in step for step in selection_steps)


def test_direction_b_math_rule_pack_includes_failure_specific_patterns():
    galaxy = GrammarGalaxy()
    required_ids = {
        "gsm_fractional_total_materials",
        "gsm_markup_profit_after_repairs",
        "gsm_repeated_schedule_distance",
        "gsm_scaled_total_minus_meals",
        "gsm_alternating_discount_pairs",
        "gsm_successive_ratio_family_total",
        "gsm_restart_from_beginning_time",
        "gsm_turnaround_distance_balance",
        "gsm_overtime_total_earnings",
    }

    assert required_ids.issubset(set(galaxy.rules.keys()))
    robe_rule = galaxy.rules["gsm_fractional_total_materials"]
    assert robe_rule.rule_strength == 0
    assert "math_template_arithmetic_chain_gpu" in robe_rule.symbol_refs
    assert robe_rule.examples


def test_direction_b_bootstrap_includes_targeted_math_patterns_and_algebra_reality_facts():
    grammar_entry_ids: set[str] = set()
    for rows in foundational_reasoning_entries().values():
        grammar_entry_ids.update(str(entry.get("id", "")).strip() for entry in rows)
    reality_entry_ids = {str(entry.get("id", "")).strip() for entry in _foundational_reality_entries()}

    assert {
        "operation_pattern_fractional_material_total",
        "operation_pattern_markup_profit_after_costs",
        "operation_pattern_repeat_groups_total",
        "operation_pattern_scaled_total_minus_meals",
        "operation_pattern_alternating_discount_pairs",
        "operation_pattern_successive_ratio_family_sum",
        "operation_pattern_restart_progress_time",
        "operation_pattern_outbound_return_distance",
        "operation_pattern_overtime_total_pay",
    }.issubset(grammar_entry_ids)

    assert {
        "reality_anchor_abstract_algebra_core",
        "reality_abstract_algebra_homomorphic_image_factor_group",
        "reality_abstract_algebra_finite_field_size_prime_power",
        "reality_abstract_algebra_diagonal_quotient_order",
        "reality_abstract_algebra_s10_max_order",
        "reality_abstract_algebra_polynomial_factor_example_z7",
    }.issubset(reality_entry_ids)


def test_track_c_foundational_reality_expands_mmlu_subject_coverage():
    tagged_entries = []
    subjects: set[str] = set()
    for entry in _foundational_reality_entries():
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        mmlu_subjects = [
            str(value).strip()
            for value in list(metadata.get("mmlu_subjects", []) or [])
            if str(value).strip()
        ]
        if not mmlu_subjects:
            continue
        tagged_entries.append(str(entry.get("id", "")).strip())
        subjects.update(mmlu_subjects)

    assert len(tagged_entries) >= 70
    assert len(subjects) >= 25
    assert {
        "abstract_algebra",
        "formal_logic",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "computer_security",
        "machine_learning",
        "jurisprudence",
        "professional_law",
        "high_school_microeconomics",
        "high_school_government_and_politics",
        "high_school_world_history",
        "sociology",
    }.issubset(subjects)


def test_track_c_grammar_rules_and_anchor_context_cover_new_mmlu_subjects(tmp_path):
    galaxy = GrammarGalaxy()
    required_rules = {
        "mmlu_algebra_field_extension_degree",
        "mmlu_formal_logic_truth_evaluation",
        "mmlu_ml_bias_variance_diagnosis",
        "mmlu_law_precedent_vs_statute",
        "mmlu_government_branch_reasoning",
    }
    assert required_rules.issubset(set(galaxy.rules.keys()))

    entry_map = {
        str(entry.get("id", "")).strip(): entry
        for entry in galaxy.entries
        if str(entry.get("id", "")).strip() in required_rules
    }
    assert set(entry_map["mmlu_algebra_field_extension_degree"]["metadata"]["mmlu_subjects"]) == {"abstract_algebra"}
    assert "machine_learning" in entry_map["mmlu_ml_bias_variance_diagnosis"]["metadata"]["mmlu_subjects"]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_track_c_subject_anchor")
    government_entry = next(
        entry
        for entry in _foundational_reality_entries()
        if str(entry.get("id", "")).strip() == "reality_government_separation_of_powers"
    )
    assert (
        kv._subject_anchor_match_score(
            entry=government_entry,
            subject_hint="high_school_government_and_politics",
            match_mode="mmlu",
        )
        > 0.0
    )


def test_phase_d_boot_binding_reuses_all_default_catalog_for_subset_requests(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_bind_once")

    assert kv._gpu_galaxy_binding is not None
    assert list(kv._gpu_galaxy_binding.get("galaxies", [])) == list(kv._live_galaxy_order)
    assert kv._pinned_all_default_binding is True
    initial_rebuilds = int(kv.metrics.gpu_bind_rebuilds)
    initial_entries = len(kv.get_gpu_galaxy_catalog())

    subset_binding = kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar"])

    assert list(subset_binding.get("galaxies", [])) == list(kv._live_galaxy_order)
    assert len(kv.get_gpu_galaxy_catalog()) == initial_entries
    assert int(kv.metrics.gpu_bind_rebuilds) == initial_rebuilds


def test_phase_d_benchmark_task_types_are_detected_without_expected_answer(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_benchmark_detection")

    assert kv._is_benchmark_evaluation_task({"type": "ARC_TASK", "task_id": "arc_0"}) is True
    assert kv._is_benchmark_evaluation_task({"type": "MMLU_TASK", "task_id": "mmlu_0"}) is True


def test_phase_d_shadow_mode_does_not_change_query_answer(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_shadow_query")
    task = {"type": "CHAT_TASK", "prompt": "What is the atomic number of carbon?"}

    monkeypatch.delenv("K3D_TRM_SHADOW", raising=False)
    baseline = kv.query(
        task["prompt"],
        specialist="chat",
        task=task,
        route={"specialist": "chat", "domain_hint": "general"},
        query_type=task["type"],
        use_enriched=True,
    )

    monkeypatch.setenv("K3D_TRM_SHADOW", "1")
    shadow = kv.query(
        task["prompt"],
        specialist="chat",
        task=task,
        route={"specialist": "chat", "domain_hint": "general"},
        query_type=task["type"],
        use_enriched=True,
    )

    assert baseline["status"] == "ok"
    assert shadow["status"] == "ok"
    assert baseline["answer"] == shadow["answer"]
    assert "trm_shadow" in baseline
    assert "trm_shadow" in shadow
    assert baseline["trm_shadow"]["python_program"] == shadow["trm_shadow"]["python_program"]
    assert float(shadow["trm_shadow"]["trm_latency_us"]) > 0.0


def test_phase_d_galaxy_decoder_checkpoint_round_trip(tmp_path):
    storage_root = tmp_path / "kv_trm_decoder"
    checkpoint_path = storage_root / "checkpoints" / "trm_galaxy_nav_weights.npz"
    galaxy_root = storage_root / "galaxies"
    galaxy_root.mkdir(parents=True, exist_ok=True)
    (galaxy_root / "Physics.jsonl").write_text(
        "{\"id\":\"physics_anchor\",\"name\":\"Physics Anchor\",\"embedding\":[0.1,0.2,0.3]}\n",
        encoding="utf-8",
    )
    decoder = {
        "W_galaxy": np.arange(10 * 512, dtype=np.float32).reshape(10, 512) / np.float32(1000.0),
        "b_galaxy": np.linspace(-0.25, 0.25, num=10, dtype=np.float32),
        "galaxy_order": np.asarray(DEFAULT_GALAXY_ORDER, dtype="<U32"),
    }
    save_galaxy_decoder_checkpoint(checkpoint_path, decoder, metadata={"trace_count": 3})

    kv = Knowledgeverse(storage_root=storage_root)

    assert kv._trm_galaxy_decoder is not None
    assert kv._trm_galaxy_decoder["W_galaxy"].shape == (len(kv._live_galaxy_order), 512)
    assert kv._trm_galaxy_decoder["b_galaxy"].shape == (len(kv._live_galaxy_order),)
    for galaxy_name in DEFAULT_GALAXY_ORDER:
        live_idx = kv._live_galaxy_order.index(galaxy_name)
        saved_idx = DEFAULT_GALAXY_ORDER.index(galaxy_name)
        np.testing.assert_allclose(
            kv._trm_galaxy_decoder["W_galaxy"][live_idx],
            decoder["W_galaxy"][saved_idx],
        )
        np.testing.assert_allclose(
            kv._trm_galaxy_decoder["b_galaxy"][live_idx],
            decoder["b_galaxy"][saved_idx],
        )
    language_idx = kv._live_galaxy_order.index("Language")
    physics_idx = kv._live_galaxy_order.index("Physics")
    np.testing.assert_array_equal(kv._trm_galaxy_decoder["W_galaxy"][language_idx], np.zeros(512, dtype=np.float32))
    np.testing.assert_array_equal(kv._trm_galaxy_decoder["W_galaxy"][physics_idx], np.zeros(512, dtype=np.float32))
    assert float(kv._trm_galaxy_decoder["b_galaxy"][language_idx]) == 0.0
    assert float(kv._trm_galaxy_decoder["b_galaxy"][physics_idx]) == 0.0
    assert kv._trm_galaxy_decoder_path == str(checkpoint_path)


def test_phase_d_trm_weight_checkpoint_round_trip(tmp_path):
    storage_root = tmp_path / "kv_trm_weights"
    checkpoint_path = storage_root / "checkpoints" / "trm_weights.npz"
    weights = initialize_trm_weight_matrices()
    weights["matryoshka"] = np.eye(512, dtype=np.float32)
    save_trm_weight_checkpoint(checkpoint_path, weights, metadata={"trace_count": 5})

    kv = Knowledgeverse(storage_root=storage_root)

    for name in ("W1", "W2", "W3", "W4"):
        np.testing.assert_allclose(kv._trm_host_weights[name], weights[name])
    np.testing.assert_allclose(kv._trm_matryoshka_host_weights, weights["matryoshka"])


def test_phase_d_decoder_training_reduces_entropy_and_improves_top1():
    traces: list[dict[str, object]] = []
    basis = np.zeros((10, 512), dtype=np.float32)
    for idx in range(10):
        basis[idx, 64 + idx] = 3.0
        basis[idx, 128 + idx] = 1.5
    for idx, galaxy in enumerate(DEFAULT_GALAXY_ORDER[:6]):
        for _ in range(3):
            traces.append(
                {
                    "python_galaxies": [galaxy],
                    "python_program": "reasoning_factual_lookup_top1",
                    "y_new_vector_512": (basis[idx] + np.float32(0.05)).astype(np.float32).tolist(),
                    "query_embedding_512": basis[idx].astype(np.float32).tolist(),
                }
            )
    metrics_before = evaluate_decoder_on_traces(traces, None)
    decoder = fit_galaxy_decoder_from_traces(traces)
    metrics_after = evaluate_decoder_on_traces(traces, decoder)

    assert metrics_before["avg_entropy"] > 2.0
    assert metrics_after["avg_entropy"] < 1.5
    assert metrics_after["top1_match_rate"] > 0.5


def test_phase_d_trace_balance_weights_prioritize_rare_targets():
    traces = [
        {
            "python_galaxies": ["Drawing"],
            "query_embedding_512": np.ones(512, dtype=np.float32).tolist(),
        },
        {
            "python_galaxies": ["Math"],
            "query_embedding_512": np.full(512, 2.0, dtype=np.float32).tolist(),
        },
        {
            "python_galaxies": ["Math"],
            "query_embedding_512": np.full(512, 3.0, dtype=np.float32).tolist(),
        },
        {
            "python_galaxies": ["Math"],
            "query_embedding_512": np.full(512, 4.0, dtype=np.float32).tolist(),
        },
    ]
    weights = build_trace_balance_weights(traces)

    assert np.isclose(np.sum(weights), len(traces))
    assert float(weights[0]) > float(weights[1])
    assert float(weights[1]) == float(weights[2]) == float(weights[3])


def test_phase_d_prediction_summary_groups_by_benchmark():
    traces = [
        {
            "benchmark": "ARC",
            "task_id": "arc_0",
            "python_galaxies": ["Drawing"],
            "y_new_vector_512": np.pad(np.array([4.0], dtype=np.float32), (0, 511)).tolist(),
        },
        {
            "benchmark": "MMLU",
            "task_id": "mmlu_0",
            "python_galaxies": ["Math"],
            "y_new_vector_512": np.pad(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float32), (0, 506)).tolist(),
        },
    ]
    summary = summarize_trace_top1_predictions(traces, decoder=None)

    assert summary["per_benchmark"]["ARC"]["correct"] == 1
    assert summary["per_benchmark"]["MMLU"]["correct"] == 1
    assert summary["per_benchmark"]["ARC"]["rows"][0]["predicted_top1"] == "Drawing"
    assert summary["per_benchmark"]["MMLU"]["rows"][0]["predicted_top1"] == "Math"
    assert summary["per_benchmark"]["ARC"]["drawing_above_0_05"] == 1
    assert summary["per_benchmark"]["ARC"]["drawing_weight_min"] > 0.05


def test_phase_d_galaxy_idf_prioritizes_discriminative_arc_targets():
    traces = [
        {"task_type": "ARC_TASK", "python_galaxies": ["Drawing", "Grammar", "Tool"]},
        {"task_type": "MATH_TASK", "python_galaxies": ["Math", "Grammar", "Tool"]},
        {"task_type": "GSM8K_TASK", "python_galaxies": ["Math", "Grammar", "Number", "Word"]},
        {"task_type": "LHE_TASK", "python_galaxies": ["Reality", "Math", "Grammar", "Word", "Character"]},
        {"task_type": "MMLU_TASK", "python_galaxies": ["Reality", "Math", "Grammar", "Word", "Character"]},
    ]
    galaxy_idf = compute_galaxy_idf(traces)
    logits = trace_target_logits(["Drawing", "Grammar", "Tool"], galaxy_idf=galaxy_idf)

    drawing_idx = DEFAULT_GALAXY_ORDER.index("Drawing")
    grammar_idx = DEFAULT_GALAXY_ORDER.index("Grammar")
    tool_idx = DEFAULT_GALAXY_ORDER.index("Tool")

    assert galaxy_idf["Drawing"] > galaxy_idf["Grammar"]
    assert logits[drawing_idx] > logits[tool_idx] > logits[grammar_idx]


def test_phase_d_target_smoothing_preserves_auxiliary_arc_signal():
    traces = [
        {"task_type": "ARC_TASK", "python_galaxies": ["Drawing", "Grammar", "Tool"]},
        {"task_type": "MATH_TASK", "python_galaxies": ["Math", "Grammar", "Tool"]},
        {"task_type": "GSM8K_TASK", "python_galaxies": ["Math", "Grammar", "Number", "Word"]},
        {"task_type": "LHE_TASK", "python_galaxies": ["Reality", "Math", "Grammar", "Word", "Character"]},
        {"task_type": "MMLU_TASK", "python_galaxies": ["Reality", "Math", "Grammar", "Word", "Character"]},
    ]
    galaxy_idf = compute_galaxy_idf(traces)
    sharp = trace_target_logits(
        ["Drawing", "Grammar", "Tool"],
        galaxy_idf=galaxy_idf,
        target_blend_alpha=1.0,
    )
    smoothed = trace_target_logits(
        ["Drawing", "Grammar", "Tool"],
        galaxy_idf=galaxy_idf,
        target_blend_alpha=0.7,
    )

    drawing_idx = DEFAULT_GALAXY_ORDER.index("Drawing")
    grammar_idx = DEFAULT_GALAXY_ORDER.index("Grammar")
    tool_idx = DEFAULT_GALAXY_ORDER.index("Tool")

    assert smoothed[drawing_idx] < sharp[drawing_idx]
    assert smoothed[grammar_idx] > sharp[grammar_idx]
    assert (smoothed[drawing_idx] - smoothed[grammar_idx]) < (sharp[drawing_idx] - sharp[grammar_idx])
    assert smoothed[drawing_idx] > smoothed[tool_idx] > smoothed[grammar_idx]


def test_phase_d_oracle_galaxy_contribution_overrides_one_hot_targets():
    logits = trace_target_logits(
        ["Math"],
        galaxy_contribution={
            "Math": 0.72,
            "Grammar": 0.18,
            "Reality": 0.06,
            "Tool": 0.04,
        },
        galaxy_idf={"Math": 1.0, "Grammar": 10.0, "Reality": 10.0, "Tool": 10.0},
        target_blend_alpha=1.0,
    )

    math_idx = DEFAULT_GALAXY_ORDER.index("Math")
    grammar_idx = DEFAULT_GALAXY_ORDER.index("Grammar")
    reality_idx = DEFAULT_GALAXY_ORDER.index("Reality")
    tool_idx = DEFAULT_GALAXY_ORDER.index("Tool")
    drawing_idx = DEFAULT_GALAXY_ORDER.index("Drawing")

    assert logits[math_idx] > logits[grammar_idx] > logits[reality_idx] > logits[tool_idx] > 0.0
    assert logits[drawing_idx] < 0.0


def test_phase_d_trm_weight_training_improves_raw_and_decoder_metrics():
    traces: list[dict[str, object]] = []
    for idx, galaxy in enumerate(DEFAULT_GALAXY_ORDER[:4]):
        for sample in range(4):
            query = np.zeros(512, dtype=np.float32)
            query[idx] = np.float32(1.0 + 0.1 * sample)
            query[32 + idx] = np.float32(0.5)
            traces.append(
                {
                    "python_galaxies": [galaxy],
                    "python_program": "reasoning_factual_lookup_top1",
                    "query_embedding_512": query.tolist(),
                    "y_new_vector_512": np.zeros(512, dtype=np.float32).tolist(),
                }
            )
    initial_weights = initialize_trm_weight_matrices()
    seeded_traces = apply_trm_weights_to_traces(traces, initial_weights)
    decoder = fit_galaxy_decoder_from_traces(seeded_traces)
    trained = train_trm_weights_from_traces(
        seeded_traces,
        decoder,
        initial_weights=initial_weights,
        epochs=250,
        learning_rate=5e-3,
        clip_norm=1.0,
    )
    trained_traces = apply_trm_weights_to_traces(seeded_traces, trained["weights"])
    refit_decoder = fit_galaxy_decoder_from_traces(trained_traces)
    metrics_after = evaluate_trm_weights_on_traces(trained_traces, trained["weights"], refit_decoder)

    assert trained["metrics_after"]["raw_top1_match_rate"] > trained["metrics_before"]["raw_top1_match_rate"]
    assert metrics_after["decoder_top1_match_rate"] >= 0.75
    assert metrics_after["decoder_avg_entropy"] < 1.5


def test_trm_weights_persist_across_knowledgeverse_restarts(tmp_path):
    storage_root = tmp_path / "kv_weights"
    kv1 = Knowledgeverse(storage_root=storage_root)

    for _ in range(3):
        kv1.log_event(
            "math_problem_success",
            {
                "specialist": "math",
                "query": "Find derivative of x^2 at x=3",
                "confidence": 0.9,
            },
        )

    state_path = storage_root / "checkpoints" / "trm_routing_state.json"
    assert state_path.exists()

    kv2 = Knowledgeverse(storage_root=storage_root)
    bias = kv2.trm_navigator.specialist_router.get_specialist_bias()
    assert bias["math"] > 0.0
    assert kv2.navigator_specialist.routing_topology


def test_sleeptime_stage_b_consolidates_trm_weights(tmp_path):
    storage_root = tmp_path / "kv_sleeptime_weights"
    kv = Knowledgeverse(storage_root=storage_root)

    kv.log_event(
        "arc_task_success",
        {
            "specialist": "visual",
            "query": "Rotate this ARC grid and reflect color mapping",
            "confidence": 0.88,
        },
    )
    kv.log_event(
        "math_problem_failure",
        {
            "specialist": "math",
            "query": "Compute hard olympiad expression",
            "confidence": 0.2,
        },
    )

    result = kv.sleeptime.execute()
    stage_b = result["stage_b"]
    assert stage_b["success"] is True
    assert stage_b["updated_count"] >= 1
    weights_path = Path(stage_b["weights_path"])
    assert weights_path.exists()
    latest_checkpoint = storage_root / "checkpoints" / "galaxy_consolidated_latest.json"
    assert latest_checkpoint.exists()
    assert result["checkpoint"]["galaxy_consolidated"]["saved"] is True

    kv_reloaded = Knowledgeverse(storage_root=storage_root)
    bias = kv_reloaded.trm_navigator.specialist_router.get_specialist_bias()
    assert bias["visual"] > 0.0
    assert kv_reloaded.house_state_summary()["warm_boot"] is True
