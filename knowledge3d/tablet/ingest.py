"""CLI: python -m knowledge3d.tablet.ingest --source <uri> --mime <type>

Sends a single INGEST envelope to the in-process daemon handler.
Prints the ingest_receipt JSON to stdout. Exit 0 on ok, 1 on error.

Thin I/O wrapper — no ingestion logic.

Usage:
    python -m knowledge3d.tablet.ingest \
        --source file:///tmp/sample.pdf \
        --mime application/pdf \
        [--lang-hint pt-BR] \
        [--pretty]

See: TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md §9
"""
from __future__ import annotations

import argparse
import json
import sys


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="knowledge3d.tablet.ingest",
        description="Queue a document for ingestion into the K3D Knowledgeverse.",
    )
    ap.add_argument(
        "--source", required=True,
        help="Source URI (file://, https://, s3://)",
    )
    ap.add_argument(
        "--mime", required=True,
        help="IANA MIME type (e.g. application/pdf, text/markdown)",
    )
    ap.add_argument(
        "--lang-hint", default=None,
        help="Optional BCP-47 language hint (e.g. pt-BR, en)",
    )
    ap.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output (default: compact single line)",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse()
    from knowledge3d.daemon.main import handle_command_inprocess
    payload: dict = {
        "command": "INGEST",
        "source_uri": args.source,
        "mime": args.mime,
    }
    if args.lang_hint is not None:
        payload["lang_hint"] = args.lang_hint
    result = handle_command_inprocess(payload)
    indent = 2 if args.pretty else None
    sys.stdout.write(json.dumps(result, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
