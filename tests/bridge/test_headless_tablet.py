from __future__ import annotations

from pathlib import Path

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletEmit, TabletIngest
from knowledge3d.cranium.actions import ActionType


class _FakeDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        task = dict(payload.get("task") or {})
        task_type = str(task.get("type", ""))
        if task_type == "ARC_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "visual", "galaxy_names": ["Drawing", "Tool"], "domain": "visual"},
                "task_result": {
                    "status": "ok",
                    "output_grid": task.get("expected_output"),
                },
            }
        if task_type == "MATH_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "math", "galaxy_names": ["Math", "Grammar"], "domain": "math"},
                "task_result": {
                    "status": "success",
                    "result": task.get("expected_answer"),
                },
            }
        if task_type == "LHE_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar", "Reality"], "domain": task.get("domain_hint", "multi")},
                "task_result": {
                    "status": "success",
                    "response": task.get("expected_answer") or "unknown",
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
    assert result["emitted"]["status"] == "success"
    assert result["emitted"]["predicted_answer"] == "4"
    assert result["emitted"]["correct"] is True
    assert result["tablet_contract"]["action_type"] == "UPDATE_TABLET"


def test_tablet_emit_lhe_extracts_option_answer_from_chat_response():
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
        "task_result": {"status": "success", "response": "The best option is B because it matches the constraint."},
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["status"] == "success"
    assert emitted["predicted_answer"] == "B"
    assert emitted["correct"] is True


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
            "status": "success",
            "response": "least privilege",
            "route": {
                "specialist": "grammar",
                "domain": "cybersecurity",
                "galaxy_names": ["Grammar", "Reality", "Tool"],
            },
        },
    }

    emitted = TabletEmit.lhe_result(envelope, response)
    assert emitted["route"]["galaxy_names"] == ["Grammar", "Reality", "Tool"]
