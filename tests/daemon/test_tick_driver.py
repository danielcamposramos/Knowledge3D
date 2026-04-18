from __future__ import annotations

import time

from knowledge3d.daemon.main import K3DDaemon
from knowledge3d.daemon.tick_driver import TickDriver


class _FakeTickKnowledgeverse:
    def __init__(self) -> None:
        self.pending: list[str] = []
        self.output: list[str] = []
        self._next = 0

    def enqueue_task(self) -> str:
        request_id = f"req_{self._next}"
        self._next += 1
        self.pending.append(request_id)
        return request_id

    def run_ticks(self, n: int = 1) -> int:
        processed = 0
        for _ in range(max(0, int(n))):
            if self.pending:
                self.output.append(self.pending.pop(0))
                processed += 1
        return processed


class _FakeDriver:
    def stats(self) -> dict[str, object]:
        return {"ticks_total": 7, "running": True}


def test_tick_driver_start_stop_stats_and_cadence_bounds() -> None:
    kv = _FakeTickKnowledgeverse()
    driver = TickDriver(kv, max_hz=50.0, idle_backoff_ms=20)

    driver.start()
    time.sleep(0.2)
    driver.stop()
    driver.stop()

    stats = driver.stats()
    assert stats["running"] is False
    assert int(stats["ticks_total"]) > 5
    assert int(stats["ticks_total"]) < 200
    assert int(stats["idle_ticks"]) >= 1


def test_tick_driver_processes_enqueued_task_without_waiting_client() -> None:
    kv = _FakeTickKnowledgeverse()
    driver = TickDriver(kv, max_hz=50.0, idle_backoff_ms=20)

    driver.start()
    driver.start()
    request_id = kv.enqueue_task()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and request_id not in kv.output:
        time.sleep(0.01)
    driver.stop()

    stats = driver.stats()
    assert request_id in kv.output
    assert int(stats["active_ticks"]) >= 1


def test_daemon_tick_status_returns_driver_stats() -> None:
    daemon = K3DDaemon.__new__(K3DDaemon)
    daemon._command_count = 0
    daemon._tick_driver = _FakeDriver()

    result = daemon.handle_command({"command": "TICK_STATUS"})

    assert result["status"] == "ok"
    assert result["tick_driver"]["ticks_total"] == 7
