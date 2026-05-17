"""CLI: python -m knowledge3d.tablet.sleeptime --run-once

Runs one sleeptime consolidation tick via the in-process daemon handler.
Prints the tick telemetry JSON to stdout.  Exit 0 on ok, 1 on error.

Thin I/O wrapper — no sleeptime logic.  All computation happens on GPU
inside sleeptime_lane_a.cu (and other existing tick kernels).

Usage:
    python -m knowledge3d.tablet.sleeptime --run-once [--pretty]
"""
from __future__ import annotations

import argparse
import json
import sys


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="knowledge3d.tablet.sleeptime")
    ap.add_argument(
        "--run-once",
        action="store_true",
        required=True,
        help="Run exactly one consolidation tick and exit.",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return ap.parse_args()


def main() -> int:
    args = _parse()
    payload = {"command": "SLEEP_TICK"}
    try:
        from knowledge3d.daemon.main import handle_command_inprocess
        result = handle_command_inprocess(payload)
    except Exception as exc:
        result = {
            "status": "error",
            "error": "sleep_tick_failed",
            "detail": str(exc),
        }
    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
