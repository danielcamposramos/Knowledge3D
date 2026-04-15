"""Simple JSON line client for K3D daemon commands."""

from __future__ import annotations

import json
import socket
from typing import Any

from knowledge3d.bridge.headless_tablet import TabletSessionTape


class SovereigntyViolation(RuntimeError):
    """Raised when a solved command reports zero GPU calls."""


class DaemonClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 7777, timeout_sec: float = 30.0):
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
        with socket.create_connection((self.host, self.port), timeout=self.timeout_sec) as sock:
            sock.sendall(wire)
            response = self._readline(sock)
        if not response:
            return {"status": "error", "error": "empty_response"}
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError as exc:
            return {"status": "error", "error": "invalid_json_response", "detail": str(exc), "raw": response}
        return decoded if isinstance(decoded, dict) else {"status": "error", "error": "response_not_object"}

    def send_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compatibility alias for sender/test call sites."""
        return self.send(payload)

    def tablet_session_open(
        self,
        *,
        session_id: str,
        tick_hz: float = 50.0,
        delta_time: float = 0.02,
    ) -> dict[str, Any]:
        return self.send(
            {
                "command": "TABLET_SESSION_OPEN",
                "session_id": str(session_id),
                "tick_hz": float(tick_hz),
                "delta_time": float(delta_time),
            }
        )

    def tablet_session_run_tape(
        self,
        tape: TabletSessionTape,
        *,
        frame_timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return self.send(
            {
                "command": "TABLET_SESSION_RUN_TAPE",
                "tape": tape.to_payload(),
                "frame_timeout_s": float(frame_timeout_s),
            }
        )

    def tablet_session_status(self) -> dict[str, Any]:
        return self.send({"command": "TABLET_SESSION_STATUS"})

    def tablet_session_close(self) -> dict[str, Any]:
        return self.send({"command": "TABLET_SESSION_CLOSE"})

    def assert_gpu_for_solved_command(
        self,
        response: dict[str, Any],
        *,
        solved_key: str = "task_result",
        context: str = "daemon_command",
    ) -> None:
        """
        Enforce sovereignty at sender level.

        If a command is solved successfully but reports zero GPU calls,
        raise a hard failure to prevent silent CPU fallback regressions.
        """
        if not isinstance(response, dict):
            return
        if str(response.get("status", "")).lower() != "ok":
            return

        solved = response.get(solved_key)
        solved_ok = False
        if isinstance(solved, dict):
            solved_ok = str(solved.get("status", "")).lower() == "success"
        elif solved_key == "" or solved is None:
            solved_ok = True

        if not solved_ok:
            return

        telemetry = response.get("telemetry")
        gpu_calls = 0
        if isinstance(telemetry, dict):
            try:
                gpu_calls = int(telemetry.get("gpu_calls_this_command", 0) or 0)
            except Exception:
                gpu_calls = 0
        if gpu_calls <= 0:
            raise SovereigntyViolation(
                f"{context}: solved command reported gpu_calls_this_command={gpu_calls}; "
                "expected > 0 for PTX sovereignty"
            )

    def _readline(self, sock: socket.socket) -> str:
        chunks: list[bytes] = []
        while True:
            block = sock.recv(4096)
            if not block:
                break
            chunks.append(block)
            if b"\n" in block:
                break
        if not chunks:
            return ""
        joined = b"".join(chunks)
        return joined.split(b"\n", 1)[0].decode("utf-8", errors="replace")
