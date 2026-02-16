from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def _send(host: str, port: int, payload: dict) -> dict:
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall(wire)
        data = b""
        while b"\n" not in data:
            block = sock.recv(4096)
            if not block:
                break
            data += block
    line = data.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    return json.loads(line) if line else {"status": "error", "error": "empty_response"}

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_daemon_handles_100_route_commands_without_restart(tmp_path: Path) -> None:
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
        deadline = time.time() + 20.0
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

        for idx in range(100):
            response = _send(
                host,
                port,
                {
                    "command": "ROUTE",
                    "specialist": "math" if idx % 2 else "visual",
                    "query": f"synthetic task {idx}",
                    "use_enriched": False,
                },
            )
            assert response.get("status") == "ok"
            assert "route" in response
            assert proc.poll() is None, "daemon restarted/exited during lifecycle test"

        shutdown = _send(host, port, {"command": "SHUTDOWN"})
        assert shutdown.get("status") == "ok"

        rc = proc.wait(timeout=10.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
