#!/usr/bin/env python3
"""Generate demo galaxy data from drawing grammar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    k3d_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(__file__).resolve().parent / ".k3d-data"
    galaxy_pending = data_dir / "galaxy_pending"
    galaxy_pending.mkdir(parents=True, exist_ok=True)

    jsonl_path = galaxy_pending / "drawing_grammar.jsonl"
    txt_path = data_dir / "drawing_grammar.txt"
    gltf_path = k3d_dir / "viewer" / "public" / "galaxy.glb"

    # Step 1: Build drawing grammar
    print("Building drawing grammar...")
    subprocess.run(
        [
            sys.executable,
            str(k3d_dir / "scripts" / "build_drawing_grammar.py"),
            "--output",
            str(jsonl_path),
        ],
        check=True,
    )

    # Step 2: Convert JSONL to text summary
    print("Generating text summary...")
    lines: list[str] = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)
            entry_id = data.get("id", "")
            visual = data.get("visual_rpn", "") or data.get("type", "")
            lines.append(f"{entry_id}: {visual}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Step 3: Generate galaxy.glb
    print("Generating galaxy.glb...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "k3dgen",
            "--text",
            str(txt_path),
            "--gltf",
            str(gltf_path),
            "--k",
            "3",
            "--reducer",
            "pca",
        ],
        check=True,
    )

    print("Demo data generated successfully")
    print(f"Galaxy: {gltf_path}")
    print(f"Entries: {len(lines)}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Error during ingestion: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise SystemExit(1)
