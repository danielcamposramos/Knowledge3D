from __future__ import annotations

from benchmarks.arc3_sdk_agent import K3DAgent, _RemoteArcCompatEnv


class _DummyEnv:
    def __init__(self) -> None:
        self.attempt = 0

    def reset(self):
        self.attempt += 1
        return {"grid": [[0, 0], [0, 0]], "available_actions": [0, 1, 2, 3]}

    def step(self, action, data=None):
        levels_completed = 1 if self.attempt >= 3 else 0
        done = True
        return {"grid": [[0, 1], [0, 0]], "available_actions": [0, 1, 2, 3]}, float(levels_completed), done, {"levels_completed": levels_completed}

    def close(self) -> None:
        return None


class _FakeEpisodeGalaxy:
    def __init__(self) -> None:
        self.seed_frames: list[dict[str, object]] = []
        self.seed_outcomes: list[dict[str, object]] = []
        self.micro_runs = 0
        self.deep_runs = 0
        self.final_consolidations = 0

    def seed_frame(self, **kwargs) -> None:
        self.seed_frames.append(dict(kwargs))

    def seed_outcome(self, **kwargs) -> None:
        self.seed_outcomes.append(dict(kwargs))

    def get_episode_context(self, last_n: int = 10):
        return {"objects": {}, "recent_outcomes": []}

    def run_micro_sleeptime(self):
        self.micro_runs += 1

    def run_deep_consolidation(self):
        self.deep_runs += 1
        return {"rules_persisted": 1, "rule_ids": [f"arc3_rule:test:{self.deep_runs}"]}

    def consolidate_to_house(self, knowledgeverse, final_score: float):
        self.final_consolidations += 1
        return {"rules_persisted": 2, "session_entries": 1}


class _FakeDelegate:
    def __init__(self) -> None:
        self.kv = object()
        self._episode_galaxy = _FakeEpisodeGalaxy()
        self.reset_calls = 0
        self.learn_calls = 0

    def reset_attempt_state(self) -> None:
        self.reset_calls += 1

    def choose_action(self, frame, **kwargs):
        return {"action": "ACTION1"}

    def learn_from_outcome(self, **kwargs):
        self.learn_calls += 1
        return 1


class _ProbeEnv:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self):
        return {"grid": [[0, 0], [0, 0]], "available_actions": [0, 1, 2, 3]}

    def step(self, action, data=None):
        self.actions.append(str(getattr(action, "name", getattr(action, "value", action))))
        return {"grid": [[0, 1], [0, 0]], "available_actions": [0, 1, 2, 3]}, 0.0, True, {"levels_completed": 0}

    def close(self) -> None:
        return None


class _ProbeThenCrashEnv:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self):
        return {"grid": [[0, 0], [0, 0]], "available_actions": [0, 1, 2, 3]}

    def step(self, action, data=None):
        self.calls += 1
        if self.calls == 1:
            return {"grid": [[0, 1], [0, 0]], "available_actions": [0, 1, 2, 3]}, 0.0, False, {"levels_completed": 0}
        raise RuntimeError("remote step exploded")

    def close(self) -> None:
        return None


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = int(status_code)
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self) -> None:
        self.get_calls: list[tuple[str, dict[str, object]]] = []
        self.post_calls: list[tuple[str, dict[str, object]]] = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.get_calls.append((str(url), dict(kwargs)))
        return _FakeResponse(
            [
                {"game_id": "ls20-9607627b", "title": "LS20", "tags": ["keyboard"]},
                {"game_id": "wa30-ee6fef47", "title": "WA30", "tags": ["keyboard"]},
            ]
        )

    def post(self, url, **kwargs):
        payload = dict(kwargs.get("json") or {})
        self.post_calls.append((str(url), dict(kwargs)))
        url = str(url)
        if url.endswith("/api/scorecard/open"):
            return _FakeResponse({"card_id": "card-123"})
        if url.endswith("/api/cmd/RESET"):
            return _FakeResponse(
                {
                    "guid": "guid-456",
                    "state": "IN_PROGRESS",
                    "available_actions": [1, 2, 3, 4],
                    "frame": [[0, 0], [0, 0]],
                    "levels_completed": 0,
                    "task_data": {},
                }
            )
        if url.endswith("/api/cmd/ACTION3"):
            return _FakeResponse(
                {
                    "guid": payload.get("guid", "guid-456"),
                    "state": "IN_PROGRESS",
                    "available_actions": [1, 2, 3, 4],
                    "frame": [[0, 1], [0, 0]],
                    "levels_completed": 0,
                    "action_input": {"id": 3},
                }
            )
        if url.endswith("/api/scorecard/close"):
            return _FakeResponse({})
        raise AssertionError(f"unexpected URL: {url}")

    def close(self) -> None:
        return None




def test_autonomous_retry_persists_episode_memory_across_attempts(monkeypatch) -> None:
    env = _DummyEnv()
    agent = K3DAgent("ls20", allow_remote_compat=True, env=env)
    delegate = _FakeDelegate()
    agent._episode_galaxy = delegate._episode_galaxy
    monkeypatch.setattr(agent, "_ensure_delegate", lambda: delegate)
    try:
        result = agent.run_until_level_complete(target_levels=1, max_attempts=3, steps_per_attempt=1)
    finally:
        agent.close()

    assert result["autonomous"] is True
    assert result["attempts_used"] == 3
    assert result["first_completion_attempt"] == 3
    assert result["levels_completed_per_attempt"] == [0, 0, 1]
    assert delegate._episode_galaxy.deep_runs == 2
    assert delegate._episode_galaxy.final_consolidations == 1
    assert len(delegate._episode_galaxy.seed_frames) == 3
    assert delegate.reset_calls == 3


def test_run_level_starts_with_delegate_action(monkeypatch) -> None:
    env = _ProbeEnv()
    agent = K3DAgent("ls20", allow_remote_compat=True, env=env)
    delegate = _FakeDelegate()
    agent._episode_galaxy = delegate._episode_galaxy
    monkeypatch.setattr(agent, "_ensure_delegate", lambda: delegate)
    try:
        result = agent.run_level(max_steps=1)
    finally:
        agent.close()

    assert env.actions == ["ACTION1"]
    assert result["steps"] == 1
    assert result["session_steps"] == 1
    assert delegate.learn_calls == 1


def test_run_level_survives_remote_step_exception_after_first_action(monkeypatch) -> None:
    env = _ProbeThenCrashEnv()
    agent = K3DAgent("ls20", allow_remote_compat=True, env=env)
    delegate = _FakeDelegate()
    agent._episode_galaxy = delegate._episode_galaxy
    monkeypatch.setattr(agent, "_ensure_delegate", lambda: delegate)
    try:
        result = agent.run_level(max_steps=2)
    finally:
        agent.close()

    assert result["steps"] == 2
    assert result["session_steps"] == 2
    assert len(agent.frame_history) == 2
    assert agent.frame_history[-1]["step_error"] == "remote step exploded"


def test_remote_arc_env_resolves_full_game_id_and_uses_sdk_style_action_payload(monkeypatch) -> None:
    fake_session = _FakeSession()
    monkeypatch.setattr(_RemoteArcCompatEnv, "_ensure_session", lambda self: fake_session)

    env = _RemoteArcCompatEnv("ls20")
    obs = env.reset()
    next_obs, reward, done, info = env.step("ACTION3")
    env.close()

    assert env.requested_game_id == "ls20"
    assert env.game_id == "ls20-9607627b"
    assert env.game_tags == ["keyboard"]
    assert obs["state"] == "IN_PROGRESS"
    assert next_obs["state"] == "IN_PROGRESS"
    assert reward == 0.0
    assert done is False
    assert info["card_id"] == "card-123"

    open_call = next(call for call in fake_session.post_calls if call[0].endswith("/api/scorecard/open"))
    reset_call = next(call for call in fake_session.post_calls if call[0].endswith("/api/cmd/RESET"))
    step_call = next(call for call in fake_session.post_calls if call[0].endswith("/api/cmd/ACTION3"))

    assert open_call[1]["json"] == {"tags": ["k3d-sovereign-r0"]}
    assert reset_call[1]["json"] == {"card_id": "card-123", "game_id": "ls20-9607627b"}
    assert step_call[1]["json"] == {"game_id": "ls20-9607627b", "guid": "guid-456"}
