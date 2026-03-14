from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def _send(host: str, port: int, payload: dict) -> dict:
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    with socket.create_connection((host, port), timeout=60.0) as sock:
        sock.sendall(wire)
        data = b""
        while b"\n" not in data:
            block = sock.recv(65536)
            if not block:
                break
            data += block
    line = data.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    return json.loads(line) if line else {"status": "error", "error": "empty_response"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _guard_math_questions() -> list[tuple[str, str]]:
    return [
        ("Solve linear equation 2x + 3 = 11.", "4"),
        ("Solve linear equation 3x + 5 = 20.", "5"),
        ("Solve linear equation 4x + 7 = 31.", "6"),
        ("Solve linear equation 6x + 2 = 26.", "4"),
        ("Solve linear equation 8x + 1 = 41.", "5"),
        ("What is 4 factorial?", "24"),
        ("What is 5 factorial?", "120"),
        ("What is 6 factorial?", "720"),
        ("What is 7 factorial?", "5040"),
        ("What is 10 factorial?", "3628800"),
        ("What is 8 choose 2?", "28"),
        ("What is 10 choose 3?", "120"),
        ("What is 12 choose 4?", "495"),
        ("What is 9 choose 4?", "126"),
        ("What is 11 choose 5?", "462"),
        ("What is the sum of the first 5 positive integers?", "15"),
        ("What is the sum of the first 10 positive integers?", "55"),
        ("What is the sum of the first 12 positive integers?", "78"),
        ("What is the sum of the first 20 positive integers?", "210"),
        ("What is the sum of the first 25 positive integers?", "325"),
    ]


def test_daemon_stability_100_math_tasks(tmp_path: Path) -> None:
    host = "127.0.0.1"
    port = _free_port()
    storage_root = tmp_path / "k3d_daemon_world"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "knowledge3d.daemon.main",
            "--mode",
            "tcp",
            "--host",
            host,
            "--port",
            str(port),
            "--storage-root",
            str(storage_root),
            "--allow-nonsovereign-query",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 30.0
        started = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"daemon exited early rc={proc.returncode}")
            try:
                pong = _send(host, port, {"command": "PING"})
                if pong.get("status") == "ok":
                    started = True
                    break
            except OSError:
                time.sleep(0.2)
        assert started, "daemon failed to start in tcp mode"

        total_gpu_calls = 0
        successful_tasks = 0
        guard_questions = _guard_math_questions()
        assert len(guard_questions) == 20
        for i in range(100):
            question, _expected = guard_questions[i % len(guard_questions)]
            response = _send(
                host,
                port,
                {
                    "command": "ROUTE",
                    "specialist": "math",
                    "task": {"type": "MATH_TASK", "task_id": f"math_{i}", "question": question},
                },
            )
            assert response.get("status") == "ok"
            task_result = response.get("task_result")
            assert isinstance(task_result, dict)
            assert proc.poll() is None, "daemon exited during command loop"
            if task_result.get("status") == "success":
                successful_tasks += 1

            telemetry = response.get("telemetry")
            assert isinstance(telemetry, dict)
            gpu_calls = int(telemetry.get("gpu_calls_this_command", 0) or 0)
            assert telemetry.get("fallback_triggered") is False
            total_gpu_calls += gpu_calls

        assert total_gpu_calls >= 50, f"expected substantial GPU activity, got {total_gpu_calls}"
        assert successful_tasks >= 1, "expected at least one successful GPU math answer during stability run"

        status = _send(host, port, {"command": "STATUS"})
        assert status.get("status") == "ok"
        assert int(status.get("gpu_calls_total", 0) or 0) >= 100

        shutdown = _send(host, port, {"command": "SHUTDOWN"})
        assert shutdown.get("status") == "ok"

        rc = proc.wait(timeout=10.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
