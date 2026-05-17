"""
Merge multiple vector CSVs (id + v0..v{d-1}) into a single CSV.

Assumptions
- All inputs share the same vector dimensionality and header format.
- Output ids are reindexed to avoid collisions (0..N-1), preserving row order.

Usage
  python -m knowledge3d.tools.merge_vectors \
    --inputs /path/a.csv /path/b.csv /path/c.csv \
    --out ../Knowledge3D.local/datasets/ai_compendium_500k_vectors.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def read_header(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8', newline='') as f:
        r = csv.reader(f)
        header = next(r)
    if not header or header[0] != 'id':
        raise ValueError(f"Invalid header in {path}")
    return header


def merge_csvs(inputs: List[Path], out: Path) -> None:
    if not inputs:
        raise ValueError("No inputs provided")
    # Validate headers and dimension consistency
    headers = [read_header(p) for p in inputs]
    base = headers[0]
    for h, p in zip(headers[1:], inputs[1:]):
        if h != base:
            raise ValueError(f"Header mismatch in {p}")
    out.parent.mkdir(parents=True, exist_ok=True)
    next_id = 0
    with out.open('w', encoding='utf-8', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(base)
        for path in inputs:
            with path.open('r', encoding='utf-8', newline='') as fi:
                r = csv.reader(fi)
                _ = next(r, None)  # skip header
                for row in r:
                    if not row:
                        continue
                    row[0] = str(next_id)
                    w.writerow(row)
                    next_id += 1


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Merge multiple vector CSVs into one")
    ap.add_argument('--inputs', nargs='+', required=True, help='List of input CSVs (id,v0,...)')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    merge_csvs([Path(p) for p in args.inputs], Path(args.out))
    print(f"Merged {len(args.inputs)} files -> {args.out}")


if __name__ == '__main__':
    main()

