from __future__ import annotations

"""
TransformerLab Bridge (CLI Stub)

Provides simple commands to:
- summarize dataset logs
- launch K3D builders for multimodal 50k GLBs
- report background job status

Usage
  python -m knowledge3d.tools.tlab_bridge status
  python -m knowledge3d.tools.tlab_bridge build --text data/ai_books_basic.txt --out viewer/public/text_50k.glb
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict


def status() -> Dict[str, str]:
    logs = Path("/home/daniel/K3D_llama_cpp/logs")
    out: Dict[str, str] = {}
    for name in ("coco_download.log", "clotho_download.log", "audiocaps_ytdlp.log", "vatex_ytdlp.log", "msrvtt_fetch.log"):
        p = logs / name
        try:
            if p.exists():
                out[name] = f"{p.stat().st_size} bytes"
            else:
                out[name] = "missing"
        except Exception:
            out[name] = "error"
    return out


def build(text: str | None, out: str) -> None:
    """Build a text GLB via the orchestrator."""
    cmd = [
        "python",
        "-m",
        "knowledge3d.tools.build_multimodal_50k",
        "--text",
        text or "",
        "--text-out",
        out,
    ]
    subprocess.run([c for c in cmd if c], check=False)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="TransformerLab Bridge CLI")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    b = sub.add_parser("build")
    b.add_argument("--text")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif args.cmd == "build":
        build(args.text, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

