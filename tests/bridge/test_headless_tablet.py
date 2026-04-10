from __future__ import annotations

from pathlib import Path

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletEmit, TabletIngest
from knowledge3d.cranium.actions import ActionType
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class _FakeDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        task = dict(payload.get("task") or {})
        task_type = str(task.get("surface_kind") or task.get("type", ""))
        if task_type == "GAME_2D":
            return {
                "status": "ok",
                "route": {"specialist": "visual", "galaxy_names": ["Drawing", "Tool"], "domain": "visual"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "grid",
                    "answer_materialized": True,
                    "output_grid": task.get("expected_output"),
                },
            }
        if task_type == "MATH":
            return {
                "status": "ok",
                "route": {"specialist": "math", "galaxy_names": ["Math", "Grammar"], "domain": "math"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "numeric",
                    "answer_text": str(task.get("expected_answer") or ""),
                    "numeric_answer": 4.0,
                    "answer_materialized": True,
                },
            }
        if task_type == "QUESTION":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar", "Reality"], "domain": task.get("domain_hint", "multi")},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "choice",
                    "answer_choice": str(task.get("expected_answer") or "unknown"),
                    "answer_text": str(task.get("expected_answer") or "unknown"),
                    "answer_materialized": True,
                },
            }
        return {"status": "error", "error": "unsupported_task"}


def test_tablet_ingest_arc_builds_standard_route_payload_and_action_buffer():
    envelope = TabletIngest.arc_task(
        task_id="arc_demo",
        training_examples=[{"input": [[1]], "output": [[2]]}],
        input_grid=[[3]],
        expected_output=[[4]],
    )

    payload = envelope.to_route_payload(use_enriched=True)
    assert payload["command"] == "ROUTE"
    assert payload["specialist"] == "visual"
    assert payload["task"]["type"] == "ARC_TASK"
    assert payload["task"]["surface_kind"] == "GAME_2D"

    buf = envelope.to_action_buffer()
    mutation_type, mutation_payload = buf.extract_tablet_mutation()
    assert buf.get_action_type() == ActionType.UPDATE_TABLET
    assert mutation_type == 1
    assert mutation_payload.shape == (6,)


def test_headless_tablet_mpc_routes_math_through_standard_contract(tmp_path: Path):
    boundary = HeadlessTabletMPC(command_handler=_FakeDaemon(), storage_root=tmp_path)
    envelope = TabletIngest.math_problem(
        task_id="math_demo",
        question="What is 2 + 2?",
        competition="AMC",
        expected_answer="4",
    )

    result = boundary.submit(envelope, use_enriched=True)

    assert result["response"]["status"] == "ok"
    assert result["route_payload"]["task"]["type"] == "MATH_TASK"
    assert result["emitted"]["status"] == "success"
    assert result["emitted"]["numeric_answer"] == 4.0
    assert result["emitted"]["answer_text"] == "4"
    assert result["emitted"]["correct"] is True
    assert result["tablet_contract"]["action_type"] == "UPDATE_TABLET"


def test_tablet_emit_lhe_uses_typed_option_answer_from_task_result():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_demo",
        question="Choose the correct answer.",
        options=["A", "B", "C"],
        domain="logic",
        expected_answer="B",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "choice",
            "answer_choice": "B",
            "answer_text": "B",
            "answer_materialized": True,
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["status"] == "success"
    assert emitted["predicted_answer"] == "B"
    assert emitted["correct"] is True
    assert envelope.task["type"] == "QUESTION_TASK"


def test_tablet_emit_flattens_nested_route_response_and_blocks_internal_answer_labels():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_nested_demo",
        question="Choose the correct answer.",
        options=["A", "B", "C"],
        domain="logic",
        expected_answer="B",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Reality"]},
            "task_result": {
                "status": "ok",
                "answer_kind": "choice",
                "answer_index": 1,
                "answer": "grammar_answer_validator",
                "response": "grammar_answer_validator",
                "winner_star_id": "grammar_answer_validator",
                "route_depth": 2,
            },
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)

    assert emitted["status"] == "success"
    assert emitted["predicted_answer"] == "B"
    assert emitted["answer_materialized"] is True
    assert emitted["route"]["galaxy_names"] == ["Reality"]


def test_tablet_emit_blocks_humanized_internal_labels_from_answer_text():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_internal_label_demo",
        question="Explain the best answer.",
        options=[],
        domain="logic",
        expected_answer="grounded explanation",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "chat", "domain": "logic", "galaxy_names": ["Grammar"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "text",
            "answer_text": "Anti Pattern Missing Validator Traversal",
            "answer_materialized": True,
            "winner_star_id": "anti_pattern_missing_validator_traversal",
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)

    assert emitted["answer_text"] == ""
    assert emitted["predicted_answer"] == ""
    assert emitted["answer_materialized"] is False


def test_tablet_emit_prefers_task_specific_route_when_present():
    envelope = TabletIngest.lhe_question(
        task_id="lhe_route_demo",
        question="Explain the best cybersecurity practice.",
        options=[],
        domain="cybersecurity",
        expected_answer="least privilege",
    )
    response = {
        "status": "ok",
        "route": {"specialist": "grammar", "domain": "cybersecurity", "galaxy_names": ["Grammar", "Tool"]},
        "task_result": {
            "status": "ok",
            "answer_kind": "text",
            "answer_text": "least privilege",
            "answer_materialized": True,
            "route": {
                "specialist": "grammar",
                "domain": "cybersecurity",
                "galaxy_names": ["Grammar", "Reality", "Tool"],
            },
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["route"]["galaxy_names"] == ["Grammar", "Reality", "Tool"]


def test_dispatch_sovereign_task_preserves_typed_question_mode():
    class _FakeRuntime:
        def __init__(self) -> None:
            self.last_task: dict[str, object] | None = None

        def dispatch_task(self, task: dict[str, object]) -> dict[str, object]:
            self.last_task = dict(task)
            return {"status": "ok", "route": {"route_family": "QUESTION"}}

    kv = Knowledgeverse.__new__(Knowledgeverse)
    runtime = _FakeRuntime()
    kv._enter_query_activity = lambda: None
    kv._mark_runtime_activity = lambda: None
    kv._leave_query_activity = lambda: None
    kv._get_sovereign_hot_path = lambda: runtime
    kv._query_sequence = 0

    result = Knowledgeverse._dispatch_sovereign_task(
        kv,
        task={
            "type": "QUESTION_TASK",
            "surface_kind": "QUESTION",
            "task_id": "q_demo",
            "query": "Which element has atomic number 6?",
            "options": ["H", "C", "O", "N"],
        },
        route={"specialist": "auto"},
        specialist="auto",
        domain_hint="chemistry",
        use_enriched=False,
    )

    assert runtime.last_task is not None
    assert runtime.last_task["type"] == "QUESTION_TASK"
    assert runtime.last_task["surface_kind"] == "QUESTION"
    assert result["query_id"] == "kvq_00000001"
