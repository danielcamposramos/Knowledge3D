from __future__ import annotations

"""
Merge multiple metadata JSON arrays into a single array file.

Usage
  python -m knowledge3d.tools.merge_meta_json \
    --inputs a.meta.json b.meta.json c.meta.json \
    --out merged.meta.json
"""

import argparse
import json
from pathlib import Path
from typing import List


def read_array(path: Path) -> List[dict]:
    txt = path.read_text(encoding="utf-8")
    try:
        arr = json.loads(txt)
    except Exception:
        raise SystemExit(f"Invalid JSON in {path}")
    if not isinstance(arr, list):
        raise SystemExit(f"Expected JSON array in {path}")
    return [it if isinstance(it, dict) else {} for it in arr]


def merge_arrays(inputs: List[Path], out: Path) -> None:
    merged: List[dict] = []
    for p in inputs:
        merged.extend(read_array(p))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {len(inputs)} files -> {out} (rows={len(merged)})")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Merge JSON arrays (metadata) into one array file")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge_arrays([Path(p) for p in args.inputs], Path(args.out))


if __name__ == "__main__":
    main()

