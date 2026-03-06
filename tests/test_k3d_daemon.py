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
        return [{"entry": {"id": "x"}, "score": 1.0, "galaxy": (galaxy_names or ["Grammar"])[0]}]

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


class _FakeKV:
    manifest_version = "test"

    def __init__(self) -> None:
        self.trm_navigator = _FakeTRM()

    def ensure_default_galaxies_loaded(self):
        return {"Grammar": 1}


class _FakeMathSpecialist:
    def process(self, task, *, use_enriched=True):
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
    return K3DDaemon(
        config=cfg,
        knowledgeverse=_FakeKV(),  # type: ignore[arg-type]
        math_specialist=_FakeMathSpecialist(),  # type: ignore[arg-type]
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
