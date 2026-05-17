"""CLI: python -m knowledge3d.tablet.chat --text '...'

Sends a single-turn CHAT envelope to the in-process daemon handler.
Prints the response JSON to stdout. Exit 0 on ok, 1 on error.

This is a thin I/O wrapper — it does NOT implement chat logic.

Usage:
    python -m knowledge3d.tablet.chat --text "what is 2+3?"
    python -m knowledge3d.tablet.chat --text "explain gravity" --pretty

See: TEMP/CLAUDE_CHAT_WINE_SPEC_04.20.2026.md §8
"""
from __future__ import annotations

import argparse
import json
import sys


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="knowledge3d.tablet.chat",
        description="Send a single-turn CHAT query to the K3D daemon in-process.",
    )
    ap.add_argument("--text", required=True, help="User message text")
    ap.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output (default: compact single line)",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse()
    from knowledge3d.daemon.main import handle_command_inprocess
    payload = {
        "command": "CHAT",
        "messages": [{"role": "user", "content": args.text}],
    }
    result = handle_command_inprocess(payload)
    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
