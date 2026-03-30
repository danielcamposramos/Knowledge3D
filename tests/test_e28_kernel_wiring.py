from __future__ import annotations

import numpy as np

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_e28_specialist_domain_bucket_normalizes_spec_domains():
    assert Knowledgeverse._specialist_domain_bucket(task_type="MATH_TASK", path={"domain_hint": "physics_kinematics"}) == "physics"
    assert Knowledgeverse._specialist_domain_bucket(task_type="ARC_TASK", path={"domain_hint": "spatial_navigation"}) == "spatial"
    assert Knowledgeverse._specialist_domain_bucket(task_type="ARC_TASK", path={"domain_hint": "visual"}) == "visual"
    assert Knowledgeverse._specialist_domain_bucket(task_type="LHE_TASK", path={"domain_hint": "formal_logic"}) == "logic"
    assert Knowledgeverse._specialist_domain_bucket(task_type="GENERAL_TASK", path={"domain_hint": "temporal_sequence"}) == "temporal"


def test_e28_world_model_threads_physics_signal_into_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_e28_world_model",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
    )

    class _FakeWorldModel:
        def __init__(self):
            self.calls = []

        def enhance_galaxy_resonance(self, query_embedding, galaxy_embeddings):
            query = np.asarray(query_embedding, dtype=np.float32)
            rows = np.asarray(galaxy_embeddings, dtype=np.float32)
            self.calls.append({"query_shape": tuple(query.shape), "rows_shape": tuple(rows.shape)})
            return np.asarray([0.92, 0.18], dtype=np.float32)

    fake_world_model = _FakeWorldModel()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    monkeypatch.setattr(kv, "get_geometry_router", lambda: None)
    monkeypatch.setattr(kv, "get_temporal_reasoning", lambda: None)
    monkeypatch.setattr(kv, "get_fractal_emitter", lambda: None)
    monkeypatch.setattr(kv, "get_world_model", lambda: fake_world_model)

    local_candidates = [
        {
            "match": {
                "embedding16": [1.0] + [0.0] * 15,
                "galaxy": "Reality",
                "category": "reality_system",
                "confidence": 0.9,
            },
            "candidate_global_idx": 10,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "similarity": 0.55,
            "option_similarity": 0.55,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {
                "embedding16": [0.0, 1.0] + [0.0] * 14,
                "galaxy": "Grammar",
                "category": "rule",
                "confidence": 0.75,
            },
            "candidate_global_idx": 11,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "similarity": 0.35,
            "option_similarity": 0.35,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="ARC_TASK",
        path={"label": "primary", "domain_hint": "physics_kinematics"},
        selection_steps=selection_steps,
    )

    assert len(fake_world_model.calls) == 1
    assert fake_world_model.calls[0]["query_shape"] == (16,)
    assert fake_world_model.calls[0]["rows_shape"] == (2, 16)
    assert local_candidates[0]["specialist_world_model"] > local_candidates[1]["specialist_world_model"]
    assert local_candidates[0]["specialist_coherence"] > local_candidates[1]["specialist_coherence"]
    assert "gre_world_model" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[0],
        primary_program_id="reasoning_arc_grid_transform_top1",
        target_galaxies=["Reality", "Grammar"],
        task_type="ARC_TASK",
        domain_hint="physics_kinematics",
    )
    assert kv._gpu_scalar_literal(local_candidates[0]["specialist_world_model"]) in expr


def test_e30_cognitive_executive_threads_trust_into_swarm_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_e30_cognitive",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
    )

    class _FakeExecutive:
        def __init__(self):
            self.calls = []

        def compute_trust_weights(self, resonance_matrix, chain_norms):
            matrix = np.asarray(resonance_matrix, dtype=np.float32)
            norms = np.asarray(chain_norms, dtype=np.float32)
            self.calls.append({"matrix_shape": tuple(matrix.shape), "norms_shape": tuple(norms.shape)})
            return np.asarray([0.91, 0.17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.75

    fake_executive = _FakeExecutive()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_world_model", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    monkeypatch.setattr(kv, "get_geometry_router", lambda: None)
    monkeypatch.setattr(kv, "get_temporal_reasoning", lambda: None)
    monkeypatch.setattr(kv, "get_fractal_emitter", lambda: None)
    monkeypatch.setattr(kv, "get_atomic_fission_fusion", lambda: None)
    monkeypatch.setattr(kv, "get_cognitive_executive", lambda: fake_executive)

    local_candidates = [
        {
            "match": {
                "embedding16": [1.0] + [0.0] * 15,
                "galaxy": "Reality",
                "category": "concept",
                "confidence": 0.9,
            },
            "candidate_global_idx": 20,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "similarity": 0.61,
            "option_similarity": 0.61,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {
                "embedding16": [0.0, 1.0] + [0.0] * 14,
                "galaxy": "Grammar",
                "category": "rule",
                "confidence": 0.8,
            },
            "candidate_global_idx": 21,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "similarity": 0.42,
            "option_similarity": 0.42,
            "program": {"id": "reasoning_arc_grid_transform_top1"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="ARC_TASK",
        path={"label": "primary", "domain_hint": "visual"},
        selection_steps=selection_steps,
    )

    assert len(fake_executive.calls) == 1
    assert fake_executive.calls[0]["matrix_shape"] == (8, 8)
    assert fake_executive.calls[0]["norms_shape"] == (8,)
    assert local_candidates[0]["specialist_trust"] > local_candidates[1]["specialist_trust"]
    assert local_candidates[0]["specialist_coherence"] > local_candidates[1]["specialist_coherence"]
    assert "gre_cognitive_executive" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[0],
        primary_program_id="reasoning_arc_grid_transform_top1",
        target_galaxies=["Reality", "Grammar"],
        task_type="ARC_TASK",
        domain_hint="visual",
    )
    assert kv._gpu_scalar_literal(local_candidates[0]["specialist_trust"]) in expr


def test_e30_atomic_fission_threads_composition_into_swarm_candidates(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_e30_atomic",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
    )

    class _FakeAtomicFissionFusion:
        def __init__(self):
            self.calls = []

        def decompose(self, compound, atoms):
            compound_arr = np.asarray(compound, dtype=np.float32).reshape(-1)
            atoms_arr = np.asarray(atoms, dtype=np.float32)
            self.calls.append({"compound_shape": tuple(compound_arr.shape), "atoms_shape": tuple(atoms_arr.shape)})
            consistency = 0.88 if float(compound_arr[0]) > float(compound_arr[1]) else 0.21
            return compound_arr, consistency

    fake_atomic = _FakeAtomicFissionFusion()
    monkeypatch.setattr(kv, "get_vector_resonator", lambda: None)
    monkeypatch.setattr(kv, "get_galaxy_resonance_engine", lambda: None)
    monkeypatch.setattr(kv, "get_graph_crystallizer", lambda: None)
    monkeypatch.setattr(kv, "get_world_model", lambda: None)
    monkeypatch.setattr(kv, "get_resonance_field", lambda: None)
    monkeypatch.setattr(kv, "get_geometry_router", lambda: None)
    monkeypatch.setattr(kv, "get_temporal_reasoning", lambda: None)
    monkeypatch.setattr(kv, "get_fractal_emitter", lambda: None)
    monkeypatch.setattr(kv, "get_cognitive_executive", lambda: None)
    monkeypatch.setattr(kv, "get_atomic_fission_fusion", lambda: fake_atomic)

    local_candidates = [
        {
            "match": {
                "embedding16": [1.0, 0.0] + [0.0] * 14,
                "galaxy": "Math",
                "category": "template",
                "confidence": 0.9,
            },
            "candidate_global_idx": 30,
            "graph_neighbors": [],
            "led_focus": 1.0,
            "similarity": 0.57,
            "option_similarity": 0.57,
            "program": {"id": "reasoning_word_problem_fission"},
            "galaxy_weight": 1.0,
        },
        {
            "match": {
                "embedding16": [0.0, 1.0] + [0.0] * 14,
                "galaxy": "Grammar",
                "category": "rule",
                "confidence": 0.75,
            },
            "candidate_global_idx": 31,
            "graph_neighbors": [],
            "led_focus": 0.0,
            "similarity": 0.33,
            "option_similarity": 0.33,
            "program": {"id": "reasoning_word_problem_fission"},
            "galaxy_weight": 1.0,
        },
    ]
    selection_steps: list[str] = []

    kv._apply_specialist_swarm_features(
        local_candidates=local_candidates,
        reference_embedding=[1.0] + [0.0] * 15,
        task_type="MATH_TASK",
        path={"label": "primary", "domain_hint": "math"},
        selection_steps=selection_steps,
    )

    assert len(fake_atomic.calls) == 2
    assert fake_atomic.calls[0]["compound_shape"] == (16,)
    assert fake_atomic.calls[0]["atoms_shape"] == (1, 16)
    assert local_candidates[0]["specialist_composition"] > local_candidates[1]["specialist_composition"]
    assert "gre_atomic_fission_fusion" in local_candidates[0]["specialist_worker"]

    expr = kv._build_gpu_candidate_score_expression(
        candidate=local_candidates[0],
        primary_program_id="reasoning_word_problem_fission",
        target_galaxies=["Math", "Grammar"],
        task_type="MATH_TASK",
        domain_hint="math",
    )
    assert kv._gpu_scalar_literal(local_candidates[0]["specialist_composition"]) in expr
