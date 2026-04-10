"""Focused surface tests for ARC Phase R0."""

from __future__ import annotations

import pathlib

import pytest

from benchmarks.arc2_local_runner import decode_arc_predictions, decode_grid_from_star, run_evaluation
from benchmarks.arc3_sdk_agent import K3DAgent, sdk_status
from benchmarks.arc_task_galaxy_seeder import (
    grid_to_program_words,
    grid_to_rpn,
    pair_to_grammar_rule,
    rpn_to_grid,
    seed_task,
)
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine
from knowledge3d.ingestion import ingest_arc_task_rules
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def _make_engine() -> ModularRPNEngine | None:
    try:
        return ModularRPNEngine()
    except Exception:
        return None


def test_grid_to_rpn_round_trip() -> None:
    grid = [[0, 1, 2], [3, 4, 5]]
    rpn = grid_to_rpn(grid)
    assert "GRID_BEGIN 2 3" in rpn
    assert "ROW_BEGIN 0 1 2 ROW_END" in rpn
    assert rpn_to_grid(rpn) == grid


def test_pair_to_grammar_rule() -> None:
    rule = pair_to_grammar_rule("test_task", 0, [[0, 1]], [[1, 0]])
    assert isinstance(rule, MeaningCentricStar)
    assert rule.meaning_class == "spatial_grid_transform"
    assert rule.domain == "grammar"
    assert rule.star_id == "spatial_grid_transform:test_task:pair0"
    assert rule.taxonomy_refs == ["spatial_grid_transform", "two_dimensional", "input_output_pair"]


def test_seed_task_from_json() -> None:
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
        ],
        "test": [{"input": [[0, 1], [0, 1]]}],
    }
    stars = seed_task(task, task_id="demo_task")
    assert len(stars) == 2
    assert all(star.meaning_class == "spatial_grid_transform" for star in stars)


def test_arc_seeder_no_hot_path_imports() -> None:
    src = pathlib.Path("benchmarks/arc_task_galaxy_seeder.py").read_text(encoding="utf-8")
    for forbidden in ["numpy", "torch", "cupy", "symengine"]:
        assert forbidden not in src


def test_ingest_arc_task_rules_stores_meaning_stars() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.rows = []

        def bulk_disk_sync(self):
            from contextlib import nullcontext

            return nullcontext()

        def store_meaning_star(self, galaxy_name: str, star, *, category: str = "meaning_star", metadata=None):
            self.rows.append((galaxy_name, star, category, dict(metadata or {})))
            return "inserted"

    manager = _Manager()
    stars = ingest_arc_task_rules(
        {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[1]]}],
        },
        task_id="arc_demo",
        galaxy_manager=manager,
    )
    assert len(stars) == 1
    assert len(manager.rows) == 1
    assert manager.rows[0][0] == "Grammar"
    assert manager.rows[0][1].meaning_class == "spatial_grid_transform"
    assert manager.rows[0][3]["benchmark_source"] == "arc_agi_2"
    assert manager.rows[0][3]["task_id"] == "arc_demo"
    assert manager.rows[0][3]["pair_idx"] == 0


def test_arc3_agent_imports_and_reports_sdk_status() -> None:
    status = sdk_status()
    assert isinstance(status, dict)
    assert "package_available" in status
    assert K3DAgent is not None


def test_arc3_remote_compat_demotes_missing_feed_to_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyEnv:
        def close(self) -> None:
            return None

    def _raise_missing_feed():
        raise RuntimeError("sovereign_build_feed_missing:run scripts/rebuild_sovereign_artifact.py --refresh-build-feed --force-rebuild")

    monkeypatch.setattr("benchmarks.arc3_sdk_agent.Knowledgeverse", _raise_missing_feed)
    agent = K3DAgent("ls20", allow_remote_compat=True, env=_DummyEnv())
    try:
        delegate = agent._ensure_delegate()
        assert delegate is None
        assert agent.policy_error is None
        assert agent.policy_warning == "sovereign_build_feed_missing: proceeding with spatial primitives only"
    finally:
        agent.close()


def test_arc3_agent_refuses_python_action_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyEnv:
        def close(self) -> None:
            return None

    agent = K3DAgent("ls20", allow_remote_compat=True, env=_DummyEnv())
    monkeypatch.setattr(agent, "_ensure_delegate", lambda: None)
    agent.policy_warning = "sovereign_build_feed_missing: proceeding with spatial primitives only"
    try:
        with pytest.raises(RuntimeError, match="sovereign_build_feed_missing"):
            agent.decide_action({"grid": [[0]], "available_actions": [0, 1]})
    finally:
        agent.close()


def test_arc3_run_level_seeds_episode_memory_and_consolidates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyEnv:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self):
            return {"grid": [[0, 0], [0, 0]], "available_actions": [0, 1, 2, 3]}

        def step(self, action, data=None):
            self.steps += 1
            return {"grid": [[0, 1], [0, 0]], "available_actions": [0, 1, 2, 3]}, 1.0, True, {"levels_completed": 1}

        def close(self) -> None:
            return None

    class _FakeEpisode:
        def __init__(self) -> None:
            self.seed_frames: list[dict[str, object]] = []
            self.consolidations: list[tuple[object, float]] = []

        def seed_frame(self, **kwargs) -> None:
            self.seed_frames.append(dict(kwargs))

        def get_episode_context(self, last_n: int = 10):
            return {
                "objects": {5: "goal"},
                "recent_outcomes": [{"action": "ACTION1"}],
            }

        def consolidate_to_house(self, knowledgeverse, final_score: float):
            self.consolidations.append((knowledgeverse, float(final_score)))
            return {"rules_persisted": 1}

    class _FakeDelegate:
        def __init__(self) -> None:
            self.kv = object()
            self._episode_galaxy = _FakeEpisode()
            self.choose_packets: list[dict[str, object]] = []
            self.learn_packets: list[dict[str, object]] = []

        def choose_action(self, frame, **kwargs):
            self.choose_packets.append(dict(kwargs))
            return {"action": "ACTION1"}

        def learn_from_outcome(self, **kwargs):
            self.learn_packets.append(dict(kwargs))
            return 1

    agent = K3DAgent("ls20", allow_remote_compat=True, env=_DummyEnv())
    delegate = _FakeDelegate()
    monkeypatch.setattr(agent, "_ensure_delegate", lambda: delegate)
    try:
        result = agent.run_level(max_steps=1)
        assert delegate._episode_galaxy.seed_frames
        assert delegate.choose_packets[0]["episode_context"]["objects"] == {5: "goal"}
        assert delegate.learn_packets[0]["action"] == "ACTION1"
        assert delegate.learn_packets[0]["levels_delta"] == 1
        assert delegate._episode_galaxy.consolidations == [(delegate.kv, 1.0)]
        assert result["episode_context_seen"] is True
        assert result["episode_consolidation"] == {"rules_persisted": 1}
    finally:
        agent.close()


def test_decode_arc_predictions_reads_primary_and_secondary_grids() -> None:
    payload = {
        "emitted": {
            "output_grid": [[1, 2], [3, 4]],
            "task_result": {"second_prediction": [[4, 3], [2, 1]]},
        }
    }
    assert decode_arc_predictions(payload) == [[[1, 2], [3, 4]], [[4, 3], [2, 1]]]


def test_arc2_runner_seeds_task_before_tablet_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    task_path = tmp_path / "demo.json"
    task_path.write_text(
        '{"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[4]]}]}',
        encoding="utf-8",
    )

    created_kv: list[object] = []

    class _FakeKV:
        def __init__(self) -> None:
            self.galaxy_manager = object()
            created_kv.append(self)

    class _FakeTablet:
        def __init__(self, *, knowledgeverse) -> None:
            self.knowledgeverse = knowledgeverse

        def submit(self, envelope, *, use_enriched: bool = True):
            return {
                "emitted": {
                    "output_grid": [[4]],
                    "correct": True,
                    "task_result": {"program_type": "grammar"},
                    "actual_result_kind": "grid",
                },
                "tablet_contract": {"surface_kind": "GAME_2D"},
            }

    seeded: list[tuple[object, str]] = []

    def _fake_seed_task(task_json, galaxy_manager=None, *, task_id=None):
        seeded.append((galaxy_manager, str(task_id)))
        return []

    monkeypatch.setattr("benchmarks.arc2_local_runner.Knowledgeverse", _FakeKV)
    monkeypatch.setattr("benchmarks.arc2_local_runner.HeadlessTabletMPC", _FakeTablet)
    monkeypatch.setattr("benchmarks.arc2_local_runner.seed_task", _fake_seed_task)

    summary = run_evaluation(tmp_path, max_tasks=1)
    assert summary["correct"] == 1
    assert len(created_kv) == 1
    assert seeded == [(created_kv[0].galaxy_manager, "demo")]


def test_arc_verification_bridge_smoke() -> None:
    engine = _make_engine()
    if engine is None:
        pytest.skip("ModularRPNEngine unavailable in this environment")
    try:
        candidate = [[1, 0], [0, 1]]
        training = [[[1, 0], [0, 1]]]
        score = engine.launch_arc_verify_candidate(candidate, training)
        assert score == 1
        score2 = engine.launch_arc_verify_candidate([[0, 0], [0, 0]], training)
        assert score2 == 0
        batch = engine.launch_arc_score_candidates([candidate, [[0, 0], [0, 0]]], training)
        assert batch == [1, 0]
    finally:
        engine.cleanup()


def test_decode_grid_from_star_round_trip() -> None:
    engine = _make_engine()
    if engine is None:
        pytest.skip("ModularRPNEngine unavailable in this environment")
    try:
        engine.bind_cas_pool()
        root = engine.launch_k3d_expr_build(grid_to_program_words([[2, 1], [0, 3]]))
        decoded = decode_grid_from_star(engine, root)
        assert decoded == [[2, 1], [0, 3]]
    finally:
        engine.cleanup()
