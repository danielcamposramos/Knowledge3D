#!/usr/bin/env python3
"""Minimal TCP client for the K3D daemon."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Prompt text for implicit CHAT requests.")
    parser.add_argument("--host", default="127.0.0.1", help="Daemon host.")
    parser.add_argument("--port", type=int, default=7777, help="Daemon port.")
    parser.add_argument(
        "--command",
        default=None,
        help="Explicit daemon command such as PING, VRAM_REPORT, or SHUTDOWN.",
    )
    return parser


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    command = str(args.command or "").strip()
    text = " ".join(str(part) for part in args.text).strip()
    if command:
        return {"command": command.upper()}
    if text:
        return {"command": "CHAT", "prompt": text}
    raise ValueError("provide prompt text or --command")


def _send(host: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    wire = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    with socket.create_connection((host, int(port)), timeout=30.0) as sock:
        sock.sendall(wire)
        chunks: list[bytes] = []
        while True:
            block = sock.recv(65536)
            if not block:
                break
            chunks.append(block)
            if b"\n" in block:
                break
    if not chunks:
        return {"status": "error", "error": "empty_response"}
    line = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace")
    try:
        decoded = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": "invalid_json_response", "detail": str(exc), "raw": line}
    return decoded if isinstance(decoded, dict) else {"status": "error", "error": "response_not_object"}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _payload_from_args(args)
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1

    try:
        response = _send(str(args.host), int(args.port), payload)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "connection_failed",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(response, indent=2, ensure_ascii=True))
    return 0 if str(response.get("status", "")).lower() != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
