from __future__ import annotations

from contextlib import nullcontext

from knowledge3d.knowledgeverse.arc3_episode_galaxy import ARC3EpisodeGalaxy


class _FakeGalaxy:
    def __init__(self) -> None:
        self.entries: list[dict] = []


class _FakeGalaxyManager:
    def __init__(self) -> None:
        self.rows: list[tuple[str, object, str, dict]] = []
        self.entries: list[tuple[str, dict]] = []
        self._galaxies: dict[str, _FakeGalaxy] = {}

    def bulk_disk_sync(self):
        return nullcontext()

    def get_galaxy(self, name: str):
        galaxy_name = str(name)
        if galaxy_name not in self._galaxies:
            self._galaxies[galaxy_name] = _FakeGalaxy()
        return self._galaxies[galaxy_name]

    def store_meaning_star(self, galaxy_name: str, star, *, category: str = "meaning_star", metadata=None, name=None):
        self.rows.append((galaxy_name, star, category, dict(metadata or {})))
        return "inserted"

    def upsert_entry(self, galaxy_name: str, entry):
        galaxy_name = str(galaxy_name)
        payload = dict(entry)
        self.entries.append((galaxy_name, payload))
        galaxy = self.get_galaxy(galaxy_name)
        entry_id = str(payload.get("id", "")).strip()
        if entry_id:
            for index, current in enumerate(galaxy.entries):
                if str(current.get("id", "")).strip() == entry_id:
                    galaxy.entries[index] = dict(payload)
                    return "updated"
        galaxy.entries.append(dict(payload))
        return "inserted"


class _FakeShadowCopy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def record_event(self, event_type: str, event_data: dict, parent_event_id=None):
        self.events.append((str(event_type), dict(event_data)))
        return "event"


class _FakeKnowledgeverse:
    def __init__(self) -> None:
        self.galaxy_manager = _FakeGalaxyManager()
        self.shadow_copy = _FakeShadowCopy()


def test_episode_galaxy_seed_frame_and_context_shape() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)
    episode.seed_frame(step_count=0, grid=[[0, 1], [1, 0]], lives=3, budget_pct=0.5, levels_completed=0)

    context = episode.get_episode_context()

    assert context["objects"] == {}
    assert context["recent_outcomes"] == []
    assert episode.frames[-1]["grid_hash"]
    assert any(entry["id"].startswith("arc3_rule:ls20:agent_adjacent_to_untested_object") for _, entry in kv.galaxy_manager.entries)
    episode.close()


def test_micro_sleeptime_drains_and_updates_context() -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())
    prev = [[0, 5], [0, 0]]
    next_grid = [[0, 5], [0, 0]]
    for step in range(3):
        episode.seed_outcome(
            step_count=step,
            action="ACTION4",
            prev_grid=prev,
            next_grid=next_grid,
            reward=0.0,
            lives_delta=0,
            levels_delta=0,
        )
    episode.run_micro_sleeptime()
    assert episode._pending_futures
    for future in list(episode._pending_futures):
        future.result(timeout=2.0)
    episode.seed_frame(step_count=3, grid=next_grid)

    context = episode.get_episode_context()
    assert episode._pending_futures == []
    assert context["recent_outcomes"]
    assert context["objects"][5] == "static_obstacle"
    episode.close()


def test_single_gpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr("knowledge3d.knowledgeverse.arc3_episode_galaxy._detect_gpu_count", lambda: 1)
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())
    assert episode.device_roles == {
        "inference": 0,
        "crystallize": 0,
        "reinforce": 0,
        "object": 0,
    }
    episode.close()


def test_consolidate_to_house_writes_only_strong_rules_and_clears_buffers() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)
    episode.seed_frame(step_count=0, grid=[[0, 1], [1, 0]], lives=3, budget_pct=0.5, levels_completed=0)
    for step in range(3):
        episode.seed_outcome(
            step_count=step,
            action="ACTION4",
            prev_grid=[[0, 5], [0, 0]],
            next_grid=[[0, 5], [0, 0]],
            reward=1.0 if step == 2 else 0.0,
            lives_delta=0,
            levels_delta=1 if step == 2 else 0,
        )
    episode._crystallize_rules()

    summary = episode.consolidate_to_house(kv, final_score=1.0)

    assert summary["rules_persisted"] >= 1
    assert any(row[0] == "Grammar" for row in kv.galaxy_manager.entries) or any(row[0] == "Grammar" for row in kv.galaxy_manager.rows)
    assert any(row[0] == "Reality" for row in kv.galaxy_manager.rows)
    assert len(episode.frames) == 0
    assert len(episode.outcomes) == 0
    episode.close()


def test_seed_rule_upserts_live_route_capable_entry() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)

    episode.seed_rule(
        state="agent_adjacent_to_color_5",
        action=4,
        outcome="frame_changed",
        confidence=0.9,
        evidence_count=3,
    )

    grammar_entries = [entry for galaxy, entry in kv.galaxy_manager.entries if galaxy == "Grammar"]
    assert grammar_entries
    live_entry = grammar_entries[-1]
    assert live_entry["route_family"] == "GAME_2D"
    assert live_entry["selection_role"] == "validator"
    assert live_entry["answer_eligible"] is True
    assert live_entry["metadata"]["action_index"] == 4
    assert live_entry["metadata"]["action_name"] == "ACTION5"
    assert live_entry["metadata"]["query_anchor"] == "interact grey state shifted transformed"
    assert live_entry["content"] == "interact state shifted transformed at grey surface"
    episode.close()


def test_seed_object_upserts_live_object_star() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)

    episode.seed_object(
        {
            "color": 7,
            "semantic_hint": "door",
            "position_label": "upper_left",
            "interaction_tested": True,
        }
    )

    grammar_entries = [entry for galaxy, entry in kv.galaxy_manager.entries if galaxy == "Grammar"]
    object_entries = [entry for entry in grammar_entries if str(entry.get("id", "")).startswith("arc3_object:ls20:color_7")]
    assert object_entries
    assert object_entries[-1]["metadata"]["behavior"] == "door"
    assert object_entries[-1]["metadata"]["query_anchor"] == "orange door object terrain"
    assert object_entries[-1]["content"] == "orange door object surface"
    episode.close()


def test_rule_anchor_uses_direction_color_and_valence() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)

    episode.seed_rule(
        state="agent_adjacent_to_color_5",
        action=1,
        outcome="blocked",
        confidence=0.8,
        evidence_count=2,
    )

    grammar_entries = [entry for galaxy, entry in kv.galaxy_manager.entries if galaxy == "Grammar"]
    rule_entries = [entry for entry in grammar_entries if str(entry.get("id", "")).startswith("arc3_rule:ls20:agent_adjacent_to_color_5:ACTION2")]
    assert rule_entries
    latest = rule_entries[-1]
    assert latest["metadata"]["query_anchor"] == "south grey collision barrier impassable"
    assert latest["content"] == "south collision barrier impassable at grey surface"
    episode.close()


def test_significant_outcome_upserts_observation_star() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)

    episode.seed_outcome(
        step_count=4,
        action="ACTION2",
        prev_grid=[[0, 5], [0, 0]],
        next_grid=[[0, 5], [0, 0]],
        reward=0.0,
        lives_delta=0,
        levels_delta=0,
    )

    grammar_entries = [entry for galaxy, entry in kv.galaxy_manager.entries if galaxy == "Grammar"]
    observation_entries = [entry for entry in grammar_entries if str(entry.get("id", "")).startswith("arc3_observation:ls20:4:ACTION2:blocked")]
    assert observation_entries
    episode.close()


def test_crystallize_from_single_observation_creates_rule() -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())

    episode.seed_outcome(
        step_count=0,
        action="ACTION4",
        prev_grid=[[0, 5], [0, 0]],
        next_grid=[[0, 5], [0, 0]],
        reward=0.0,
        lives_delta=0,
        levels_delta=0,
    )
    episode._crystallize_rules(full_history=True)

    rule = episode._rules_by_key.get(("ACTION4", "agent_adjacent_to_color_5"))
    assert rule is not None
    assert rule["predicted_outcome"] == "blocked"
    assert rule["evidence_count"] == 1
    assert 0.19 <= float(rule["confidence"]) <= 0.21
    episode.close()


def test_crystallize_blocked_generates_avoidance_rules() -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())
    episode.seed_outcome(
        step_count=0,
        action="ACTION2",
        prev_grid=[[0, 5], [0, 0]],
        next_grid=[[0, 5], [0, 0]],
        reward=0.0,
        lives_delta=0,
        levels_delta=0,
    )
    episode._crystallize_rules(full_history=True)
    blocked_key = next(
        (
            key
            for key, rule in episode._rules_by_key.items()
            if key[0] == "ACTION2" and str(rule.get("predicted_outcome", "")) == "blocked"
        ),
        None,
    )
    assert blocked_key is not None
    blocked_condition = blocked_key[1]

    for alternative in ("ACTION1", "ACTION3", "ACTION4"):
        key = (alternative, blocked_condition)
        assert key in episode._rules_by_key
        assert str(episode._rules_by_key[key]["predicted_outcome"]).startswith("avoid_")
    episode.close()


def test_negative_rules_persist_with_evidence_two() -> None:
    kv = _FakeKnowledgeverse()
    episode = ARC3EpisodeGalaxy("ls20", kv)
    episode.seed_rule(
        state="agent_adjacent_to_color_8",
        action="ACTION2",
        outcome="blocked",
        confidence=0.9,
        evidence_count=2,
        source="test",
    )
    episode.seed_rule(
        state="agent_adjacent_to_color_9",
        action="ACTION1",
        outcome="moved",
        confidence=0.9,
        evidence_count=2,
        source="test",
    )

    persisted = episode._persist_strong_rules(min_confidence=0.5, min_evidence=3)
    assert any(rule_id.startswith("arc3_rule:ls20:agent_adjacent_to_color_8:ACTION2") for rule_id in persisted)
    assert not any(rule_id.startswith("arc3_rule:ls20:agent_adjacent_to_color_9:ACTION1") for rule_id in persisted)
    episode.close()


def test_episode_boot_loads_persisted_rules_from_house() -> None:
    kv = _FakeKnowledgeverse()
    first = ARC3EpisodeGalaxy("ls20", kv)
    first.seed_rule(
        state="agent_adjacent_to_color_5",
        action="ACTION1",
        outcome="moved",
        confidence=0.8,
        evidence_count=4,
        source="session_a",
    )
    first._persist_strong_rules(min_confidence=0.5, min_evidence=3)
    first.close()

    second = ARC3EpisodeGalaxy("ls20", kv)
    loaded = second._rules_by_key.get(("ACTION1", "agent_adjacent_to_color_5"))
    assert loaded is not None
    assert loaded["source"] in {"session_a", "arc3_episode_galaxy", "house_persisted"}
    second.close()


def test_micro_sleeptime_emits_arc3_learn_log(capsys) -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())
    episode.seed_outcome(
        step_count=1,
        action="ACTION2",
        prev_grid=[[0, 5], [0, 0]],
        next_grid=[[0, 5], [0, 0]],
        reward=0.0,
        lives_delta=0,
        levels_delta=0,
    )
    episode.run_micro_sleeptime()
    for future in list(episode._pending_futures):
        future.result(timeout=2.0)
    episode._drain_pending_futures()
    output = capsys.readouterr().out
    assert "[ARC3-LEARN]" in output
    assert "galaxy_stars=" in output
    episode.close()


def test_episode_rule_query_prefers_non_blocking_rule() -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())
    for step in range(3):
        episode.seed_outcome(
            step_count=step,
            action="ACTION4",
            prev_grid=[[0, 5], [0, 0]],
            next_grid=[[0, 1], [0, 0]],
            reward=0.0,
            lives_delta=0,
            levels_delta=0,
        )
    for step in range(3, 6):
        episode.seed_outcome(
            step_count=step,
            action="ACTION3",
            prev_grid=[[0, 5], [0, 0]],
            next_grid=[[0, 5], [0, 0]],
            reward=0.0,
            lives_delta=0,
            levels_delta=0,
        )
    episode._crystallize_rules(full_history=True)

    action_index = episode.query_rule_for_state(
        avatar_centroid=(0.0, 0.0),
        adjacent_colors=[5],
        frame_state="gameplay",
        blocked_actions={2},
    )

    assert action_index in {0, 3}
    episode.close()


def test_episode_rule_query_uses_adjacent_untested_prior() -> None:
    episode = ARC3EpisodeGalaxy("ls20", _FakeKnowledgeverse())

    action_index = episode.query_rule_for_state(
        avatar_centroid=(1.0, 1.0),
        adjacent_colors=[9],
        frame_state="gameplay",
        blocked_actions=set(),
        has_adjacent_untested=True,
    )

    assert action_index == 4
    episode.close()
