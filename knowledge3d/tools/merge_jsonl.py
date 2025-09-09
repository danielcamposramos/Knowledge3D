from __future__ import annotations

"""
Merge multiple JSONL files into one, with optional de-duplication by key.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.merge_jsonl \
    --out docs/reports/training/rlwhf_dataset_unified.jsonl \
    --dedup query \
    docs/reports/training/rlwhf_dataset_open_4000_anthropic.jsonl \
    docs/reports/training/rlwhf_dataset_open_1000.jsonl \
    docs/reports/training/rlwhf_dataset_glb.jsonl \
    docs/reports/training/rlwhf_dataset.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except OSError:
        return


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Merge JSONL files with optional de-dup by key")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup", help="Key to de-duplicate by (e.g., query)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    n_in = 0
    n_out = 0
    with out.open("w", encoding="utf-8") as f:
        for p in args.paths:
            P = Path(p)
            for row in iter_jsonl(P):
                n_in += 1
                if args.dedup:
                    k = str(row.get(args.dedup) or "")
                    if not k:
                        # keep rows without the key
                        pass
                    else:
                        if k in seen:
                            continue
                        seen.add(k)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
    print(json.dumps({"input_files": len(args.paths), "rows_in": n_in, "rows_out": n_out, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()

