#!/usr/bin/env python3
"""Download IMO Bench CSV resources into the local dataset workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request


DEFAULT_TARGET_DIR = Path("/K3D/Knowledge3D.local/datasets/imo_bench")
FILES = {
    "answerbench_v2.csv": "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv",
    "proofbench.csv": "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/proofbench.csv",
    "gradingbench.csv": "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/gradingbench.csv",
}


def download_imo_bench(target_dir: str | Path = DEFAULT_TARGET_DIR, *, timeout: int = 60) -> dict[str, dict[str, object]]:
    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for filename, url in FILES.items():
        target = root / filename
        with urllib.request.urlopen(url, timeout=timeout) as response:
            target.write_bytes(response.read())
        manifest[filename] = {
            "url": url,
            "path": str(target),
            "bytes": target.stat().st_size,
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IMO Bench resources")
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    manifest = download_imo_bench(target_dir=args.target_dir, timeout=max(1, int(args.timeout)))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
