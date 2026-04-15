from __future__ import annotations

from knowledge3d.bridge.headless_tablet import TabletSessionFrame, TabletSessionTape, TabletIngest
from knowledge3d.daemon.main import K3DDaemon


class _FakeTabletBoundary:
    def __init__(self) -> None:
        self.opened: list[dict[str, object]] = []
        self.closed = 0
        self.last_tape: TabletSessionTape | None = None
        self.active = False

    def open_live_session(self, **kwargs: object) -> dict[str, object]:
        self.active = True
        self.opened.append(dict(kwargs))
        return {"session_id": str(kwargs.get("session_id") or "session"), "preflight": {"status": "ok"}}

    def live_session_status(self) -> dict[str, object]:
        return {"active": self.active}

    def close_live_session(self) -> dict[str, object]:
        self.active = False
        self.closed += 1
        return {"closed": True}

    def run_tape_session(self, tape: TabletSessionTape, **kwargs: object) -> dict[str, object]:
        self.last_tape = tape
        return {
            "session_id": tape.session_id,
            "suite_name": tape.suite_name,
            "surface_kind": tape.surface_kind,
            "preflight": {"status": "ok"},
            "results": [
                {
                    "frame_id": frame.frame_id,
                    "envelope": frame.envelope,
                    "emitted": {
                        "status": "success",
                        "correct": True,
                        "task_result": {"gpu_execution": True, "runtime": "tablet_bridge_session_query"},
                    },
                    "tablet_contract": {"sovereign_path": "tablet_bridge_session"},
                }
                for frame in tape.frames
            ],
        }


def _fake_daemon() -> K3DDaemon:
    daemon = K3DDaemon.__new__(K3DDaemon)
    daemon._command_count = 0
    daemon._tablet_boundary = _FakeTabletBoundary()
    return daemon


def test_daemon_exposes_tablet_session_open_status_close_commands() -> None:
    daemon = _fake_daemon()

    opened = daemon.handle_command({"command": "TABLET_SESSION_OPEN", "session_id": "demo"})
    status = daemon.handle_command({"command": "TABLET_SESSION_STATUS"})
    closed = daemon.handle_command({"command": "TABLET_SESSION_CLOSE"})

    assert opened["status"] == "ok"
    assert status["status"] == "ok"
    assert status["session"]["active"] is True
    assert closed["status"] == "ok"


def test_daemon_runs_serializable_tablet_tape() -> None:
    daemon = _fake_daemon()
    tape = TabletSessionTape(
        session_id="demo_tape",
        suite_name="math",
        surface_kind="MATH",
        use_enriched=True,
        frames=(
            TabletSessionFrame(
                frame_id="f1",
                envelope=TabletIngest.math_problem(
                    task_id="m1",
                    question="What is 2 + 2?",
                    competition="AMC",
                    expected_answer="4",
                ),
                expected="4",
                source_meta={"suite": "math"},
            ),
        ),
    )

    response = daemon.handle_command({"command": "TABLET_SESSION_RUN_TAPE", "tape": tape.to_payload()})

    assert response["status"] == "ok"
    assert response["session"]["session_id"] == "demo_tape"
    assert response["results"][0]["envelope"]["task_id"] == "m1"
    assert response["results"][0]["tablet_contract"]["sovereign_path"] == "tablet_bridge_session"
