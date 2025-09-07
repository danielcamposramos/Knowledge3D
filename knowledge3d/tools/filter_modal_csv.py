from __future__ import annotations

"""
Filter a modality vectors CSV + metadata by keywords and (optionally) cap size.

Inputs
- --csv: vectors CSV with header [id,v0,v1,...]
- --meta: metadata JSON (array) or JSONL; length should match CSV rows

Behavior
- Keeps rows where the metadata text contains any of the provided keywords
  (case-insensitive). Text is constructed from common fields: caption, text,
  label, title, description, and any string fields present in the object.
- If not enough matches to reach --max, keeps what’s available.

Outputs
- --out-csv: filtered CSV with same header and only selected rows
- --out-meta: filtered metadata as a JSON array (aligned with CSV rows)

Usage
  python -m knowledge3d.tools.filter_modal_csv \
    --csv  ../Knowledge3D.local/datasets/coco.train.clip.csv \
    --meta ../Knowledge3D.local/datasets/coco.train.meta.json \
    --out-csv viewer/public/_world/coco.sample.csv \
    --out-meta viewer/public/_world/coco.sample.meta.json \
    --keywords rain,street,car,child,city --max 800
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _read_meta(meta_path: Path) -> List[dict]:
    txt = meta_path.read_text(encoding="utf-8")
    if txt.lstrip().startswith("["):
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [it if isinstance(it, dict) else {} for it in arr]
        return []
    out: List[dict] = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            j = json.loads(s)
        except Exception:
            j = {}
        out.append(j if isinstance(j, dict) else {})
    return out


def _meta_text(m: dict) -> str:
    keys = ["caption", "text", "label", "title", "description"]
    pieces: List[str] = []
    for k in keys:
        v = m.get(k)
        if isinstance(v, str) and v:
            pieces.append(v)
    # Add any other string fields (best-effort)
    for k, v in m.items():
        if k in keys:
            continue
        if isinstance(v, str) and v and len(pieces) < 8:
            pieces.append(v)
    return " \n ".join(pieces).lower()


def _match_any(text: str, kws: List[str]) -> bool:
    if not text:
        return False
    for k in kws:
        if k in text:
            return True
    return False


def filter_csv(csv_in: Path, meta_in: Path, csv_out: Path, meta_out: Path, keywords: List[str], max_items: int) -> Tuple[int, int]:
    meta = _read_meta(meta_in)
    keep_idx: List[int] = []
    kws = [k.strip().lower() for k in keywords if k.strip()]

    # First pass over metadata to choose indices
    for i, m in enumerate(meta):
        if len(keep_idx) >= max_items:
            break
        t = _meta_text(m)
        if _match_any(t, kws):
            keep_idx.append(i)

    # Write out filtered CSV rows
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_in.open("r", encoding="utf-8", newline="") as fi, csv_out.open("w", encoding="utf-8", newline="") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo)
        header = next(r, None)
        if header:
            w.writerow(header)
        # Stream rows by index
        i = 0
        sel_set = set(keep_idx)
        for row in r:
            if i in sel_set:
                w.writerow(row)
            i += 1

    # Write filtered metadata as JSON array
    meta_out.write_text(json.dumps([meta[i] for i in keep_idx], ensure_ascii=False, indent=2), encoding="utf-8")
    return len(meta), len(keep_idx)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Filter modality CSV+metadata by keywords")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--keywords", required=True, help="Comma-separated keywords")
    ap.add_argument("--max", type=int, default=800)
    args = ap.parse_args()
    n_all, n_keep = filter_csv(Path(args.csv), Path(args.meta), Path(args.out_csv), Path(args.out_meta), args.keywords.split(","), int(args.max))
    print(f"Filtered {n_keep}/{n_all} rows by keywords: {args.keywords}")


if __name__ == "__main__":
    main()

