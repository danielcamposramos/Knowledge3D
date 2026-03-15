from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
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

    uniform = 1.0 / float(len(kv.DEFAULT_GALAXIES))
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

    for galaxy_name in kv.DEFAULT_GALAXIES:
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


def test_phase_d_boot_binding_reuses_all_default_catalog_for_subset_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_TRM_NAVIGATE", "1")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_trm_bind_once")

    assert kv._gpu_galaxy_binding is not None
    assert list(kv._gpu_galaxy_binding.get("galaxies", [])) == list(kv.DEFAULT_GALAXIES)
    assert kv._pinned_all_default_binding is True
    initial_rebuilds = int(kv.metrics.gpu_bind_rebuilds)
    initial_entries = len(kv.get_gpu_galaxy_catalog())

    subset_binding = kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar"])

    assert list(subset_binding.get("galaxies", [])) == list(kv.DEFAULT_GALAXIES)
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
    assert "trm_shadow" not in baseline
    assert "trm_shadow" in shadow
    assert float(shadow["trm_shadow"]["trm_latency_us"]) > 0.0


def test_phase_d_galaxy_decoder_checkpoint_round_trip(tmp_path):
    storage_root = tmp_path / "kv_trm_decoder"
    checkpoint_path = storage_root / "checkpoints" / "trm_galaxy_nav_weights.npz"
    decoder = {
        "W_galaxy": np.arange(10 * 512, dtype=np.float32).reshape(10, 512) / np.float32(1000.0),
        "b_galaxy": np.linspace(-0.25, 0.25, num=10, dtype=np.float32),
        "galaxy_order": np.asarray(DEFAULT_GALAXY_ORDER, dtype="<U32"),
    }
    save_galaxy_decoder_checkpoint(checkpoint_path, decoder, metadata={"trace_count": 3})

    kv = Knowledgeverse(storage_root=storage_root)

    assert kv._trm_galaxy_decoder is not None
    np.testing.assert_allclose(kv._trm_galaxy_decoder["W_galaxy"], decoder["W_galaxy"])
    np.testing.assert_allclose(kv._trm_galaxy_decoder["b_galaxy"], decoder["b_galaxy"])
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

    kv_reloaded = Knowledgeverse(storage_root=storage_root)
    bias = kv_reloaded.trm_navigator.specialist_router.get_specialist_bias()
    assert bias["visual"] > 0.0
