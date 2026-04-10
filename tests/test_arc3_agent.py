from __future__ import annotations

from pathlib import Path
import pytest

from benchmarks.arc_agi_3 import ACTION_LABELS, ACTION_NAMES, K3DARC3Agent


class _FakeKnowledgeverse:
    def __init__(self, result: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.result = dict(result or {})

    def execute_task(self, *, task, route=None, specialist=None, domain_hint=None, **kwargs):
        self.calls.append(
            {
                "task": task,
                "route": route,
                "specialist": specialist,
                "domain_hint": domain_hint,
                "kwargs": kwargs,
            }
        )
        return dict(self.result)

    def get_led_pathfinder(self):
        return None


def test_arc3_agent_routes_through_execute_task():
    kv = _FakeKnowledgeverse(
        {
            "action_index": 2,
            "action_name": "ACTION3",
            "confidence": 0.8,
            "convergence_signal": 1,
            "iterations_used": 6,
            "gpu_execution": True,
        }
    )
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [[0, 1], [1, 0]]
    goal = [[1, 0], [0, 1]]
    train = [{"input": [[0, 0]], "output": goal}]
    action = agent.choose_action(
        frame,
        goal_frame=goal,
        task_data={"train": train},
        available_actions=[1, 2, 3],
        game_id="ls20",
        levels_completed=1,
        episode_context={"objects": {5: "goal"}, "recent_outcomes": [{"action": "ACTION1"}]},
    )

    assert len(kv.calls) == 1
    call = kv.calls[0]
    assert call["task"]["surface_kind"] == "GAME_2D"
    assert call["task"]["input_grid"] == frame
    assert call["task"]["expected_output"] == goal
    assert call["task"]["training_examples"] == train
    assert call["task"]["task_id"] == "arc3_live_0001"
    assert call["task"]["query"].startswith("arc3 game frame")
    assert "adjacent cells" in call["task"]["query"]
    assert "action1" in call["task"]["query"]
    assert "levels" in call["task"]["query"]
    assert "primary action" not in call["task"]["query"]
    assert "action move" not in call["task"]["query"]
    assert call["task"]["options"] == ACTION_NAMES
    assert call["task"]["step_count"] == 0
    assert call["task"]["game_id"] == "ls20"
    assert call["task"]["levels_completed"] == 1
    assert call["task"]["known_objects"] == {5: "goal"}
    assert call["route"]["specialist"] == "visual"
    assert call["route"]["domain_hint"] == "game_2d"
    assert call["specialist"] == "visual"
    assert call["domain_hint"] == "game_2d"
    assert action["action"] == "ACTION3"
    assert action["label"] == ACTION_LABELS[2]
    assert action["gpu_execution"] is True
    assert action["actual_result_kind"] == "control"
    assert action["game_id"] == "ls20"
    assert action["levels_completed"] == 1


def test_arc3_agent_uses_emitted_action_payload_and_metadata():
    kv = _FakeKnowledgeverse(
        {
            "action_name": "ACTION6",
            "action_input": {"x": 4, "y": 7},
            "confidence": 0.42,
            "gpu_execution": True,
            "route_family": "game2d_reasoning",
            "failure_code": "policy_warning:spatial_only",
        }
    )
    agent = K3DARC3Agent(knowledgeverse=kv)

    action = agent.choose_action([[0, 1], [0, 0]], available_actions=[6])

    assert action["action"] == "ACTION6"
    assert action["x"] == 4
    assert action["y"] == 7
    assert action["click_reason"] == "tablet_boundary_click"
    assert action["confidence"] == 0.42
    assert action["route_family"] == "game2d_reasoning"
    assert action["failure_code"] == "policy_warning:spatial_only"


def test_arc3_agent_requires_click_payload_for_action6():
    kv = _FakeKnowledgeverse({"action_name": "ACTION6", "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    with pytest.raises(RuntimeError, match="arc3_sovereign_action_not_materialized|arc3_click_payload_missing"):
        agent.choose_action([[5, 5, 5], [5, 6, 5], [5, 5, 5]], available_actions=[6])


def test_arc3_agent_fails_when_result_is_not_actionable():
    kv = _FakeKnowledgeverse({"status": "ok"})
    agent = K3DARC3Agent(knowledgeverse=kv)

    with pytest.raises(RuntimeError, match="arc3_sovereign_action_not_materialized"):
        agent.choose_action([[0, 1], [0, 0]], available_actions=[0, 3, 4])


def test_arc3_agent_does_not_use_spatial_plan_fallback(monkeypatch):
    kv = _FakeKnowledgeverse({"status": "ok"})
    agent = K3DARC3Agent(knowledgeverse=kv)
    monkeypatch.setattr(agent.tablet_boundary, "submit", lambda envelope, use_enriched=True: {})

    with pytest.raises(RuntimeError, match="arc3_sovereign_action_not_materialized"):
        agent.choose_action([[0, 1], [0, 0]], available_actions=[0, 1, 2, 3])


def test_arc3_agent_surfaces_tablet_submit_exception(monkeypatch):
    kv = _FakeKnowledgeverse({"status": "ok"})
    agent = K3DARC3Agent(knowledgeverse=kv)

    def _boom(envelope, use_enriched=True):
        raise RuntimeError("tablet boundary exploded")

    monkeypatch.setattr(agent.tablet_boundary, "submit", _boom)
    with pytest.raises(RuntimeError, match="arc3_tablet_boundary_failed"):
        agent.choose_action([[0, 1], [0, 0]], available_actions=[0, 1, 2, 3])


def test_arc3_agent_does_not_query_episode_rule_dict(monkeypatch):
    kv = _FakeKnowledgeverse({"action_index": 1, "action_name": "ACTION2", "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)
    monkeypatch.setattr(
        agent._episode_galaxy,
        "query_rule_for_state",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("episode dict fallback should not run")),
    )
    action = agent.choose_action(
        [[0, 5], [0, 0]],
        available_actions=[0, 1, 2, 3],
        episode_context={"objects": {5: "goal"}},
    )

    assert action["action"] == "ACTION2"
    assert action["direct_action_materialized"] is True


def test_arc3_agent_tracks_budget_and_lives_from_frame():
    kv = _FakeKnowledgeverse({"action_index": 0, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [[4] * 64 for _ in range(64)]
    for row in (61, 62):
        for col in range(12, 52):
            frame[row][col] = 3
        for col in range(52, 56):
            frame[row][col] = 11
    for row in (61, 62):
        for base in (58, 62):
            frame[row][base] = 8
            frame[row][base + 1] = 8

    action = agent.choose_action(frame, available_actions=[1, 2, 3, 4])

    assert action["movement_budget"]["bucket"] == "critical"
    assert action["lives_remaining"] == 2


def test_learn_from_outcome_tracks_completion_and_stall(tmp_path):
    kv = _FakeKnowledgeverse({"action_index": 0})
    agent = K3DARC3Agent(knowledgeverse=kv, log_path=tmp_path / "arc3_agent.jsonl")
    frame = [[0, 1], [0, 0]]

    agent.choose_action(frame)
    improved = agent.learn_from_outcome(
        levels_completed=1,
        frame=[[1, 0], [0, 0]],
        action="ACTION1",
        prev_frame=frame,
        reward=1.0,
        lives_delta=0,
        levels_delta=1,
    )
    agent.choose_action(frame)
    stalled = agent.learn_from_outcome(
        levels_completed=1,
        frame=frame,
        action="ACTION1",
        prev_frame=frame,
        reward=0.0,
        lives_delta=0,
        levels_delta=0,
    )
    agent.close()

    assert improved == 1
    assert stalled == -1
    assert agent.action_history[0]["levels_completed"] == 1
    assert agent.action_history[0]["reward"] == 1.0
    assert agent.action_history[0]["micro_sleeptime_scheduled"] is True
    assert agent._step_count == 2
    assert Path(tmp_path / "arc3_agent.jsonl").exists()


def test_arc3_agent_source_has_no_transitional_local_runner_shortcuts():
    source = Path("benchmarks/arc_agi_3.py").read_text(encoding="utf-8")

    assert "benchmarks.arc3_local" not in source
    assert "arc_transform_inferrer" not in source
    assert "ARCRPNExecutor" not in source
    assert source.count("def choose_action(") == 1
