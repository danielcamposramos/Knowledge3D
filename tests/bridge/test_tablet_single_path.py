from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge3d.bridge.headless_tablet import (
    HeadlessTabletMPC,
    TabletIngest,
    TabletSessionFrame,
    TabletSessionTape,
)


class _SinglePathKnowledgeverse:
    DEFAULT_GALAXIES = ("Drawing", "Math", "Grammar", "Reality")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def ensure_default_galaxies_loaded(self) -> dict[str, int]:
        return {"Drawing": 4, "Math": 4, "Grammar": 4, "Reality": 4}

    def execute_task(
        self,
        *,
        task: dict[str, Any],
        route: dict[str, Any] | None = None,
        specialist: str = "auto",
        domain_hint: str | None = None,
        use_enriched: bool = True,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "task": dict(task),
                "route": dict(route or {}),
                "specialist": str(specialist),
                "domain_hint": domain_hint,
                "use_enriched": bool(use_enriched),
            }
        )
        return {
            "status": "ok",
            "route": {
                "specialist": str(route.get("specialist") or specialist if isinstance(route, dict) else specialist),
                "route_family": str(task.get("surface_kind") or "QUESTION"),
            },
            "task_result": {
                "status": "ok",
                "program_id": "gpu_task_dispatch_sovereign",
                "trace_star_ids": ["meaning_router", "executor_star"],
                "answer_materialized": True,
                "gpu_execution": True,
                "route_family": str(task.get("surface_kind") or "QUESTION"),
                "answer_text": "ok",
            },
        }

    def log_event(self, name: str, payload: dict[str, Any]) -> None:
        return None


def test_tablet_submitters_share_one_sovereign_program_path(tmp_path: Path) -> None:
    kv = _SinglePathKnowledgeverse()
    boundary = HeadlessTabletMPC(command_handler=lambda payload: payload, knowledgeverse=kv, storage_root=tmp_path)
    tape = TabletSessionTape(
        session_id="single_path_demo",
        suite_name="natural_queries",
        surface_kind="QUESTION",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="mmlu_like",
                envelope=TabletIngest.question_task(
                    task_id="mmlu_like",
                    question="Which option best explains the evidence chain?",
                    options=["A", "B", "C", "D"],
                    domain="history",
                    expected_answer="B",
                ),
                expected="B",
                source_meta={"suite": "mmlu"},
            ),
            TabletSessionFrame(
                frame_id="gsm8k_like",
                envelope=TabletIngest.math_problem(
                    task_id="gsm8k_like",
                    question="Janet has 16 ducks, gives away 4, then doubles the remainder. What is the result?",
                    expected_answer="24",
                ),
                expected="24",
                source_meta={"suite": "gsm8k"},
            ),
            TabletSessionFrame(
                frame_id="arc_like",
                envelope=TabletIngest.game2d_task(
                    task_id="arc_like",
                    query="Transform this grid using the examples.",
                    training_examples=[{"input": [[1]], "output": [[1]]}],
                    input_grid=[[1]],
                    expected_output=[[1]],
                    result_kind="grid",
                ),
                expected=[[1]],
                source_meta={"suite": "arc"},
            ),
        ),
    )

    result = boundary.run_tape_session(tape, enforce_preflight=False)

    rows = list(result.get("results") or [])
    assert len(rows) == 3
    program_ids = {str(row["response"]["task_result"].get("program_id") or "") for row in rows}
    assert program_ids == {"gpu_task_dispatch_sovereign"}
    for row in rows:
        assert list(row["response"]["task_result"].get("trace_star_ids") or [])
        assert str(row["tablet_contract"].get("sovereign_path") or "") == "knowledgeverse_dispatch_session"
