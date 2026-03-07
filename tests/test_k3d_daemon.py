from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.daemon.main import DaemonConfig, K3DDaemon


class _FakeTRM:
    def __init__(self) -> None:
        self._trace: list[str] = []

    def route(self, *, query, specialist="auto", domain_hint=None, galaxy_names=None):
        return {
            "specialist": specialist,
            "domain": domain_hint or "any",
            "galaxy_names": galaxy_names or ["Grammar"],
            "reason": "fake",
        }

    def query(self, *, query, galaxy_names=None, top_k=10, specialist="auto", domain_hint=None):
        query_text = str(query)
        lowered = query_text.lower()
        if "candidate answer:" in lowered:
            candidate = query_text.split("Candidate answer:", 1)[1].strip()
            return [
                {
                    "entry": {
                        "id": candidate.lower(),
                        "name": candidate,
                        "content": f"Candidate evidence for {candidate}",
                    },
                    "score": 1.0,
                    "galaxy": (galaxy_names or ["Grammar"])[0],
                }
            ]
        return [
            {
                "entry": {
                    "id": "x",
                    "name": "evidence",
                    "content": query_text,
                },
                "score": 1.0,
                "galaxy": (galaxy_names or ["Grammar"])[0],
            }
        ]

    def navigate_and_compose(self, *, query, specialist, domain_hint, use_enriched):
        return {"program_type": "math_expression", "expression": query, "route": {"specialist": specialist}}

    def execute(self, composed):
        return 4.0 if "2x + 3 = 11" in str(composed.get("expression", "")) else None

    def get_reasoning_trace(self):
        return list(self._trace)

    def clear_trace(self):
        self._trace.clear()

    def process_chat(self, messages, use_enriched=True):
        return "ok"


class _FakeNavigatorSpecialist:
    def plan_routes(self, query: str, *, specialist: str = "auto", galaxy_names=None, use_forward_backward: bool = False, domain_hint=None):
        base = {
            "specialist": specialist,
            "domain": domain_hint or "any",
            "galaxy_names": galaxy_names or ["Math", "Grammar", "Tool"],
            "query_variant": query,
        }
        if not use_forward_backward:
            return [base]
        return [
            {
                **base,
                "forward_parse": {
                    "context": [{"type": "context", "raw": "Janet lays 16 eggs"}],
                    "goal": {"type": "goal", "raw": "How much does she make"},
                },
            },
            {
                **base,
                "backward_parse": {
                    "dependencies": [{"type": "context", "raw": "She sells the remainder for 2 dollars each"}],
                    "goal": {"type": "goal", "raw": "How much does she make"},
                },
            },
            {
                **base,
                "fusion_parse": {
                    "merged_variables": {},
                    "unified_goal": {"type": "goal", "raw": "How much does she make"},
                },
            },
        ]


class _FakeKV:
    manifest_version = "test"

    def __init__(self) -> None:
        self.trm_navigator = _FakeTRM()
        self.navigator_specialist = _FakeNavigatorSpecialist()

    def ensure_default_galaxies_loaded(self):
        return {"Grammar": 1}


class _FakeMathSpecialist:
    def __init__(self) -> None:
        self.last_task = None

    def process(self, task, *, use_enriched=True):
        self.last_task = dict(task)
        question = str(task.get("question", "") or task.get("query", ""))
        if "2x + 3 = 11" in question:
            try:
                from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
                ModularRPNEngine._global_gpu_call_count += 1
            except Exception:
                pass
            return {
                "status": "success",
                "result": 4.0,
                "rpn_program": "11 3 - 2 /",
                "coefficients": {"a": 2.0, "b": 3.0, "c": 11.0},
                "pattern_id": "fake_linear",
                "template_id": "fake_template",
            }
        return {"status": "error", "reason": "unsupported_question"}


def _daemon() -> K3DDaemon:
    cfg = DaemonConfig(storage_root=Path("/tmp/k3d_daemon_test"), require_ptx_query=False)
    math_specialist = _FakeMathSpecialist()
    return K3DDaemon(
        config=cfg,
        knowledgeverse=_FakeKV(),  # type: ignore[arg-type]
        math_specialist=math_specialist,  # type: ignore[arg-type]
    )


def test_ping_and_status() -> None:
    daemon = _daemon()
    out = daemon.handle_command({"command": "PING"})
    assert out["status"] == "ok"
    assert out["manifest_version"] == "test"
    assert "drawing_warmup" in out
    assert "geometry_warmup" in out
    assert "material_warmup" in out


def test_route_query_and_solve_math() -> None:
    daemon = _daemon()

    route = daemon.handle_command({"command": "ROUTE", "query": "solve x"})
    assert route["status"] == "ok"
    assert route["route"]["reason"] == "fake"

    query = daemon.handle_command({"command": "QUERY", "query": "algebra"})
    assert query["status"] == "ok"
    assert query["count"] == 1

    solve = daemon.handle_command({"command": "SOLVE_MATH", "question": "If 2x + 3 = 11, what is x?"})
    assert solve["status"] == "ok"
    assert solve["result"] == 4.0


def test_shutdown_sets_flag() -> None:
    daemon = _daemon()
    out = daemon.handle_command({"command": "SHUTDOWN"})
    assert out["status"] == "ok"
    assert daemon.should_shutdown is True


def test_handle_line_reports_gpu_call_delta() -> None:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

    ModularRPNEngine.reset_global_gpu_call_count()
    daemon = _daemon()
    raw = json.dumps({"command": "SOLVE_MATH", "question": "If 2x + 3 = 11, what is x?"})
    out = json.loads(daemon._handle_line(raw))
    telemetry = out["telemetry"]
    assert out["status"] == "ok"
    assert telemetry["gpu_calls_this_command"] >= 1
    assert telemetry["gpu_calls_total"] >= 1
    assert telemetry["fallback_triggered"] is False


def test_solve_math_includes_directional_parse_bundle() -> None:
    daemon = _daemon()
    out = daemon.handle_command({"command": "SOLVE_MATH", "question": "If 2x + 3 = 11, what is x?"})
    assert out["status"] == "ok"
    last_task = daemon.math_specialist.last_task
    assert isinstance(last_task, dict)
    assert "route_plan" in last_task
    assert "forward_parse" in last_task
    assert "backward_parse" in last_task
    assert "fusion_parse" in last_task


def test_route_lhe_non_math_uses_structured_four_pass_choice() -> None:
    daemon = _daemon()
    out = daemon.handle_command(
        {
            "command": "ROUTE",
            "query": "Pick B.",
            "domain_hint": "logic",
            "task": {
                "type": "LHE_TASK",
                "task_id": "lhe_logic_1",
                "prompt": "Pick B.",
                "options": ["A", "B", "C"],
                "domain_hint": "logic",
                "expected_answer": "B",
            },
        }
    )
    assert out["status"] == "ok"
    task_result = out["task_result"]
    assert task_result["status"] == "ok"
    assert task_result["response"] == "B"
    assert "four_pass" in task_result
    assert int(task_result["four_pass"]["composition_depth"]) >= 4
    assert task_result["reasoning_trace"]


def test_lhe_open_answer_prefers_semantic_text_over_code_like_tokens() -> None:
    daemon = _daemon()
    answer = daemon._synthesize_lhe_open_answer(
        fused_entities=[{"kind": "phrase", "value": "energy level"}],
        goal={
            "raw": "How many energy levels are filled?",
            "tokens": ["how", "many", "energy", "levels", "filled"],
        },
        evidence_rows=[
            {
                "rank_weight": 1.0,
                "fields": {
                    "content": "The filled shell count is 3.",
                    "description": "This system has three filled levels.",
                    "rpn_program": "DCT8 BLOCKS_TO_GRID",
                    "pattern_form": "English SVO",
                },
            }
        ],
    )
    assert answer == "3"


def test_lhe_open_answer_canonicalizes_number_words_for_short_numeric_goal() -> None:
    daemon = _daemon()
    answer = daemon._synthesize_lhe_open_answer(
        fused_entities=[{"kind": "phrase", "value": "filled levels"}],
        goal={
            "raw": "How many filled levels are there?",
            "tokens": ["how", "many", "filled", "levels", "there"],
        },
        evidence_rows=[
            {
                "rank_weight": 1.0,
                "fields": {
                    "description": "There are three filled levels in this system.",
                    "semantics": "filled shell count",
                },
            }
        ],
    )
    assert answer == "3"
