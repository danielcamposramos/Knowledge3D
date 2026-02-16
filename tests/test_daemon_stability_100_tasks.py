from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def _send(host: str, port: int, payload: dict) -> dict:
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    with socket.create_connection((host, port), timeout=15.0) as sock:
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


def _build_linear_question(a: int, b: int, x: int) -> str:
    c = a * x + b
    return f"If {a}x + {b} = {c}, what is x?"


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
        for i in range(100):
            a = (i % 9) + 1
            b = (i % 7) + 1
            x = (i % 11) + 1
            question = _build_linear_question(a, b, x)
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
            assert task_result.get("status") == "success"
            assert proc.poll() is None, "daemon exited during command loop"

            telemetry = response.get("telemetry")
            assert isinstance(telemetry, dict)
            gpu_calls = int(telemetry.get("gpu_calls_this_command", 0) or 0)
            assert gpu_calls > 0, f"task {i} solved without GPU calls"
            assert telemetry.get("fallback_triggered") is False
            total_gpu_calls += gpu_calls

        assert total_gpu_calls >= 100, f"expected >=100 GPU calls, got {total_gpu_calls}"

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
