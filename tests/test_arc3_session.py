from __future__ import annotations

from scripts import run_arc3_session
from scripts.run_arc3_agent import normalize_goal_frame


def test_inter_game_consolidation_runs_three_passes():
    class _Knowledgeverse:
        def __init__(self) -> None:
            self.calls = 0

        def jarvis_sleep_consolidation(self):
            self.calls += 1
            return {"updated": True, "pass": self.calls}

    kv = _Knowledgeverse()
    result = run_arc3_session._inter_game_consolidation(
        kv,
        {"state": "NOT_FINISHED", "levels_completed": 0},
    )

    assert kv.calls == 3
    assert len(result) == 3
    assert all(row["session_outcome"] == "stalled" for row in result)


def test_run_arc3_session_reuses_one_knowledgeverse(tmp_path, monkeypatch):
    import benchmarks.arc_agi_3 as arc3_module
    import knowledge3d.knowledgeverse.knowledgeverse as kv_module

    created_kv = []
    seen_agents = []

    class _Knowledgeverse:
        def __init__(self, *args, **kwargs) -> None:
            created_kv.append(self)

        def jarvis_sleep_consolidation(self):
            return {"updated": True}

    class _Agent:
        def __init__(self, *, max_actions=500, log_path=None, knowledgeverse=None) -> None:
            self.max_actions = max_actions
            self.log_path = log_path
            self.kv = knowledgeverse
            seen_agents.append(self)

        def close(self) -> None:
            return None

    def _fake_run_single_game(agent, *, game_id, api_url, log_dir):
        return {
            "game_id": game_id,
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "actions": 0,
            "log_path": str(log_dir / f"{game_id}.jsonl"),
            "kv_id": id(agent.kv),
        }

    monkeypatch.setattr(kv_module, "Knowledgeverse", _Knowledgeverse)
    monkeypatch.setattr(arc3_module, "K3DARC3Agent", _Agent)
    monkeypatch.setattr(run_arc3_session, "run_single_game", _fake_run_single_game)

    summary = run_arc3_session.run_arc3_session(
        game_ids=["game-a", "game-b"],
        max_actions_per_game=12,
        log_dir=tmp_path,
    )

    assert len(created_kv) == 1
    assert len(seen_agents) == 2
    assert all(agent.kv is created_kv[0] for agent in seen_agents)
    assert [row["game_id"] for row in summary["games"]] == ["game-a", "game-b"]
    assert all(len(row["inter_game_consolidation"]) == 3 for row in summary["games"])


def test_normalize_goal_frame_extracts_from_task_data():
    goal = normalize_goal_frame(
        {
            "frame": [[0, 1], [0, 0]],
            "task_data": {
                "train": [{"input": [[0, 0]], "output": [[1, 2], [3, 4]]}],
            },
        }
    )
    assert goal == [[1, 2], [3, 4]]


def test_normalize_goal_frame_prefers_explicit_goal():
    goal = normalize_goal_frame(
        {
            "frame": [[0]],
            "goal": [[1, 2]],
            "task_data": {
                "train": [{"input": [[0]], "output": [[9, 9]]}],
            },
        }
    )
    assert goal == [[1, 2]]
