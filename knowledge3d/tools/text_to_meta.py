from __future__ import annotations

"""
Convert a text lines file into a JSON metadata array aligned by line index.

Each output item: {"label": <snippet>, "text": <full>, "type": "text"}

Usage
  python -m knowledge3d.tools.text_to_meta \
    --text ../Knowledge3D.local/datasets/matched/pool.txt \
    --out  ../Knowledge3D.local/datasets/pool.meta.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Text lines -> metadata JSON array")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = Path(args.text)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    def snip(s: str, n: int = 48) -> str:
        return s if len(s) <= n else (s[: n - 3] + "...")
    arr = [{"label": snip(s), "text": s, "type": "text"} for s in lines]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(arr)} metas -> {out}")


if __name__ == "__main__":
    main()

