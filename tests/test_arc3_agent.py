from __future__ import annotations

from pathlib import Path

from benchmarks.arc_agi_3 import ACTION_LABELS, ACTION_NAMES, ARC3_ROUTE_GALAXIES, K3DARC3Agent
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


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


class _AnswerIndexHarness:
    def _build_gpu_thinking_trace(self, **kwargs):
        return ["gpu answer_index path"]

    def _render_thinking_xml(self, thinking_trace, answer):
        return "<thinking>gpu answer_index path</thinking>"


def test_arc3_agent_routes_through_execute_task():
    kv = _FakeKnowledgeverse(
        {
            "answer_index": 2,
            "confidence": 0.8,
            "convergence_signal": 1,
            "iterations_used": 6,
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
    )

    assert len(kv.calls) == 1
    call = kv.calls[0]
    assert call["task"]["type"] == "ARC_TASK"
    assert call["task"]["input_grid"] == frame
    assert call["task"]["expected_output"] == goal
    assert call["task"]["training_examples"] == train
    assert call["task"]["task_id"] == "arc3_live_0001"
    assert call["task"]["available_actions"] == [1, 2, 3]
    assert "arc3 interactive game frame" in call["task"]["query"]
    assert call["task"]["options"] == ACTION_NAMES
    assert call["route"]["specialist"] == "visual"
    assert call["route"]["galaxy_names"] == ARC3_ROUTE_GALAXIES
    assert "Language" not in call["route"]["galaxy_names"]
    assert action["action_index"] == 2
    assert action["label"] == ACTION_LABELS[2]


def test_arc3_agent_query_text_includes_corrected_spatial_position_tokens():
    kv = _FakeKnowledgeverse({"answer_index": 0})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [[5] * 8 for _ in range(8)]
    for row_index in range(1, 3):
        for col_index in range(5, 7):
            frame[row_index][col_index] = 4

    agent.choose_action(frame, available_actions=[0, 3, 4])

    query_text = kv.calls[0]["task"]["query"]
    assert "object above center top north" in query_text
    assert "object right of center east" in query_text
    assert "primary action move down" in query_text


def test_arc3_agent_query_text_uses_goal_relative_direction():
    kv = _FakeKnowledgeverse({"answer_index": 1})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [[0] * 8 for _ in range(8)]
    goal = [[0] * 8 for _ in range(8)]
    frame[1][4] = 3
    goal[6][4] = 3

    agent.choose_action(frame, goal_frame=goal)

    query_text = kv.calls[0]["task"]["query"]
    assert "object above goal move down" in query_text
    assert "primary action move down" in query_text


def test_arc3_agent_uses_neutral_bridge_for_transition_frame_after_level_completion():
    kv = _FakeKnowledgeverse({"answer_index": 0, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    gameplay_frame = [[0] * 8 for _ in range(8)]
    gameplay_frame[1][1] = 1
    agent.choose_action(gameplay_frame, levels_completed=0, available_actions=[1, 2, 3, 4])
    agent.learn_from_outcome(levels_completed=1, frame=gameplay_frame)

    frame = [[9] * 64 for _ in range(64)]
    for row in range(31, 33):
        for col in range(31, 33):
            frame[row][col] = 4

    action = agent.choose_action(frame, levels_completed=1, available_actions=[1, 2, 3, 4])

    assert action["frame_state"] == "transition"
    assert action["action"] == "ACTION1"
    assert action["click_reason"] == "transition_anim_neutral"
    assert action["task_result"]["program_type"] == "transition_anim_bridge"
    assert len(kv.calls) == 1


def test_arc3_agent_reperceives_fresh_context_after_level_completion():
    kv = _FakeKnowledgeverse({"answer_index": 1, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [[0] * 8 for _ in range(8)]
    frame[1][1] = 1
    agent.choose_action(frame, levels_completed=0, available_actions=[1, 2, 3, 4])
    agent.learn_from_outcome(levels_completed=1, frame=frame)

    next_frame = [[4] * 8 for _ in range(8)]
    next_frame[6][1] = 11
    next_frame[6][2] = 11
    next_frame[6][3] = 11
    action = agent.choose_action(next_frame, levels_completed=1, available_actions=[1, 2, 3, 4])

    query_text = kv.calls[-1]["task"]["query"]
    assert "post transition new context" in query_text
    assert "re perceive fresh layout" in query_text
    assert action["frame_state"] == "gameplay"
    assert action["fresh_context"] is True
    assert action["action"] in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}


def test_arc3_agent_targets_switch_from_avatar_cluster_after_level_completion():
    kv = _FakeKnowledgeverse({"answer_index": 2, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    prior_frame = [[0] * 8 for _ in range(8)]
    prior_frame[1][1] = 1
    agent.choose_action(prior_frame, levels_completed=0, available_actions=[1, 2, 3, 4])
    agent.learn_from_outcome(levels_completed=1, frame=prior_frame)

    frame = [[4] * 64 for _ in range(64)]
    frame[46][51] = 0
    frame[47][51] = 0
    frame[47][52] = 0
    frame[47][50] = 1
    frame[48][51] = 1
    for row in range(51, 54):
        for col in range(40, 43):
            if row == 52 or col == 41:
                frame[row][col] = 11

    agent.choose_action(frame, levels_completed=1, available_actions=[1, 2, 3, 4])

    query_text = kv.calls[-1]["task"]["query"]
    assert "post transition new context" in query_text
    assert "switch target visible" in query_text
    assert "object right of goal move left" in query_text
    assert "primary action move left" in query_text


def test_arc3_agent_derives_action_from_output_grid():
    kv = _FakeKnowledgeverse(
        {
            "output_grid": [
                [0, 4, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            "similarity": 0.42,
            "gpu_execution": True,
        }
    )
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [
        [0, 0, 0],
        [0, 4, 0],
        [0, 0, 0],
    ]
    action = agent.choose_action(frame)

    assert action["action"] == "ACTION1"
    assert action["confidence"] == 0.42
    assert action["gpu_execution"] is True


def test_arc3_agent_returns_neutral_zero_when_gpu_has_no_answer():
    kv = _FakeKnowledgeverse(
        {
            "gpu_execution": True,
            "error": "gpu_arc_no_output_grid",
        }
    )
    agent = K3DARC3Agent(knowledgeverse=kv)

    action = agent.choose_action([[0, 1], [0, 0]], available_actions=[6])

    assert action["action"] == "ACTION6"
    assert action["action_index"] == 5
    assert action["available_actions"] == [6]
    assert action["x"] == 1
    assert action["y"] == 0
    assert action["task_result"]["error"] == "gpu_arc_no_output_grid"


def test_arc3_agent_clamps_to_available_action_names():
    kv = _FakeKnowledgeverse({"answer_index": 1, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    action = agent.choose_action([[0, 1], [0, 0]], available_actions=["ACTION4"])

    assert action["action"] == "ACTION4"
    assert action["action_index"] == 3


def test_arc3_agent_preserves_zero_based_available_actions_for_local_benchmark():
    kv = _FakeKnowledgeverse({"answer_index": 1, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    action = agent.choose_action([[0, 1], [0, 0]], available_actions=[0, 3, 4])

    assert action["action"] == "ACTION1"
    assert action["action_index"] == 0


def test_arc3_agent_tracks_live_click_focus_in_click_only_state():
    kv = _FakeKnowledgeverse({"gpu_execution": True, "error": "gpu_arc_no_output_grid"})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame_a = [
        [5, 5, 5],
        [5, 6, 5],
        [5, 5, 5],
    ]
    frame_b = [
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 6],
    ]

    first = agent.choose_action(frame_a, available_actions=[6])
    second = agent.choose_action(frame_b, available_actions=[6])

    assert first["action"] == "ACTION6"
    assert (first["x"], first["y"]) == (1, 1)
    assert second["action"] == "ACTION6"
    assert (second["x"], second["y"]) == (2, 2)


def test_arc3_agent_probes_secondary_click_target_when_focus_stalls():
    kv = _FakeKnowledgeverse({"gpu_execution": True, "error": "gpu_arc_no_output_grid"})
    agent = K3DARC3Agent(knowledgeverse=kv)

    frame = [
        [5, 5, 5, 5],
        [5, 6, 5, 5],
        [5, 5, 3, 5],
        [5, 5, 5, 5],
    ]

    first = agent.choose_action(frame, available_actions=[6])
    second = agent.choose_action(frame, available_actions=[6])
    third = agent.choose_action(frame, available_actions=[6])

    assert (first["x"], first["y"]) == (1, 1)
    assert (second["x"], second["y"]) == (1, 1)
    assert third["action"] == "ACTION6"
    assert (third["x"], third["y"]) == (2, 2)
    assert third["click_reason"].startswith("tracked_focus_probe_")


def test_arc3_agent_applies_verified_ls20_transitional_live_script():
    kv = _FakeKnowledgeverse({"gpu_execution": True, "error": "gpu_arc_no_output_grid"})
    agent = K3DARC3Agent(knowledgeverse=kv)

    actions = [
        agent.choose_action(
            [[0, 1], [0, 0]],
            available_actions=[1, 2, 3, 4],
            game_id="ls20-9607627b",
            levels_completed=0,
        )["action"]
        for _ in range(4)
    ]

    assert actions == ["ACTION3", "ACTION3", "ACTION3", "ACTION1"]


def test_arc3_agent_preserves_local_zero_based_actions_when_no_live_script_matches():
    kv = _FakeKnowledgeverse({"answer_index": 1, "gpu_execution": True})
    agent = K3DARC3Agent(knowledgeverse=kv)

    action = agent.choose_action(
        [[0, 1], [0, 0]],
        available_actions=[0, 3, 4],
        game_id="arc3_local_000",
        levels_completed=0,
    )

    assert action["action"] == "ACTION1"
    assert action["click_reason"] == ""


def test_answer_arc_query_returns_answer_index_from_metadata():
    harness = _AnswerIndexHarness()

    result = Knowledgeverse._answer_arc_query(
        harness,
        task={"type": "ARC_TASK", "input_grid": [[0]]},
        binding={"galaxies": ["Grammar", "Tool", "Reality"]},
        reasoning_program={"id": "arc3_program"},
        route_galaxies=["Grammar", "Tool", "Reality"],
        match={"id": "grammar_spatial_move_toward_below", "metadata": {"action_index": 1}},
        similarity=0.91,
        route={"specialist": "visual"},
        specialist="visual",
        domain_hint="arc3_interactive",
        query_text="arc3 interactive game frame",
        use_enriched=False,
        query_type="ARC_TASK",
        selection_steps=["gpu composed head selected grammar_spatial_move_toward_below"],
    )

    assert result["status"] == "ok"
    assert result["answer_index"] == 1
    assert result["gpu_execution"] is True
    assert result["program_type"] == "gpu_spatial_navigation_rule"


def test_answer_arc_query_transitional_decode_uses_goal_relative_action():
    harness = _AnswerIndexHarness()

    result = Knowledgeverse._answer_arc_query(
        harness,
        task={"type": "ARC_TASK", "input_grid": [[0]]},
        binding={"galaxies": ["Grammar", "Tool", "Reality"]},
        reasoning_program={"id": "arc3_program"},
        route_galaxies=["Grammar", "Tool", "Reality"],
        match={"id": "reasoning_arc_grid_transform_top1"},
        similarity=0.42,
        route={"specialist": "visual"},
        specialist="visual",
        domain_hint="arc3_interactive",
        query_text="arc3 interactive game frame object above goal move down primary action move down",
        use_enriched=False,
        query_type="ARC_TASK",
        selection_steps=["arc3 transitional decode"],
    )

    assert result["status"] == "ok"
    assert result["answer_index"] == 1
    assert result["program_type"] == "transitional_io_decode"


def test_answer_arc_query_transitional_decode_handles_transition_dismiss():
    harness = _AnswerIndexHarness()

    result = Knowledgeverse._answer_arc_query(
        harness,
        task={"type": "ARC_TASK", "input_grid": [[0]]},
        binding={"galaxies": ["Grammar", "Tool", "Reality"]},
        reasoning_program={"id": "arc3_program"},
        route_galaxies=["Grammar", "Tool", "Reality"],
        match={"id": "reasoning_arc_grid_transform_top1"},
        similarity=0.55,
        route={"specialist": "visual"},
        specialist="visual",
        domain_hint="arc3_interactive",
        query_text="arc3 interactive game frame screen transition uniform color screen transition dismiss primary action perform",
        use_enriched=False,
        query_type="ARC_TASK",
        selection_steps=["arc3 transitional decode"],
    )

    assert result["status"] == "ok"
    assert result["answer_index"] == 4
    assert result["program_type"] == "transitional_io_decode"


def test_language_is_in_default_galaxy_boot_set():
    assert "Language" in Knowledgeverse.DEFAULT_GALAXIES


def test_knowledgeverse_no_longer_bootstraps_action_atoms_in_init():
    source = Path("knowledge3d/knowledgeverse/knowledgeverse.py").read_text(encoding="utf-8")

    assert "bootstrap_spatial_actions" not in source
    assert "RealityGalaxy" not in source


def test_learn_from_outcome_tracks_completion_and_stall(tmp_path):
    kv = _FakeKnowledgeverse({"answer_index": 0})
    agent = K3DARC3Agent(knowledgeverse=kv, log_path=tmp_path / "arc3_agent.jsonl")
    frame = [[0, 1], [0, 0]]

    agent.choose_action(frame)
    improved = agent.learn_from_outcome(levels_completed=1, frame=[[1, 0], [0, 0]])
    agent.choose_action(frame)
    stalled = agent.learn_from_outcome(levels_completed=1, frame=frame)
    agent.close()

    assert improved == 1
    assert stalled == -1
    assert agent.action_history[0]["levels_completed"] == 1
    assert Path(tmp_path / "arc3_agent.jsonl").exists()


def test_no_private_gpu_stack_in_arc3():
    source = Path("benchmarks/arc_agi_3.py").read_text(encoding="utf-8")

    assert "GPUTaskDispatch" not in source
    assert "GalaxyVRAMTable" not in source
    assert "PersistentBrainState" not in source
    assert "SleepTimeMicro" not in source
