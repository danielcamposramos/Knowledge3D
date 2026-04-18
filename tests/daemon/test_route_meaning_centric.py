from __future__ import annotations

from knowledge3d.daemon.main import K3DDaemon


JANET_QUERY = "Janet had 16 ducks and bought 2 more. How many ducks does Janet have now?"


class _FakeKnowledgeverse:
    manifest_version = "test"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def ensure_default_galaxies_loaded(self) -> dict[str, int]:
        return {"Math": 8, "Grammar": 4, "Number": 4, "Word": 4}

    def _discover_live_galaxy_names(self) -> list[str]:
        return ["Math", "Grammar", "Number", "Word"]

    def execute_task(self, *, task, route=None, specialist="auto", domain_hint=None, use_enriched=True):
        task_snapshot = dict(task or {})
        route_snapshot = dict(route or {})
        self.calls.append(
            {
                "task": task_snapshot,
                "route": route_snapshot,
                "specialist": str(specialist),
                "domain_hint": domain_hint,
                "use_enriched": bool(use_enriched),
            }
        )
        answer = "18" if JANET_QUERY in str(task_snapshot.get("query", "")) else ""
        return {
            "status": "ok",
            "answer": answer,
            "result": answer,
            "runtime": "knowledgeverse_gpu_query",
            "route": route_snapshot,
            "task_result": {
                "status": "success",
                "answer": answer,
                "result": answer,
                "runtime": "knowledgeverse_gpu_query",
            },
        }


def _fake_daemon() -> tuple[K3DDaemon, _FakeKnowledgeverse]:
    kv = _FakeKnowledgeverse()
    daemon = K3DDaemon.__new__(K3DDaemon)
    daemon._command_count = 0
    daemon._default_counts = {"Math": 1, "Grammar": 1, "Number": 1, "Word": 1}
    daemon.kv = kv
    return daemon, kv


def test_route_meaning_centric_same_query_ignores_labels() -> None:
    daemon, kv = _fake_daemon()

    base = daemon.handle_command(
        {
            "command": "ROUTE",
            "task": {
                "task_id": "janet_base",
                "question": JANET_QUERY,
            },
        }
    )
    labeled_math = daemon.handle_command(
        {
            "command": "ROUTE",
            "task": {
                "task_id": "janet_math",
                "question": JANET_QUERY,
                "surface_kind": "MATH",
                "type": "MATH_TASK",
            },
        }
    )
    labeled_game = daemon.handle_command(
        {
            "command": "ROUTE",
            "task": {
                "task_id": "janet_game",
                "question": JANET_QUERY,
                "surface_kind": "GAME_2D",
                "type": "ARC_TASK",
            },
        }
    )

    assert base["status"] == "ok"
    assert labeled_math["status"] == "ok"
    assert labeled_game["status"] == "ok"
    assert base["task_result"]["result"] == "18"
    assert labeled_math["task_result"]["result"] == "18"
    assert labeled_game["task_result"]["result"] == "18"
    assert base["route"] == labeled_math["route"] == labeled_game["route"]

    for call in kv.calls:
        task_payload = dict(call["task"] or {})
        assert task_payload["query"] == JANET_QUERY
        assert "surface_kind" not in task_payload
        assert "type" not in task_payload
        assert "task_type" not in task_payload

