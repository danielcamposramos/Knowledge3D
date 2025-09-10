from __future__ import annotations

"""
Align a metadata JSONL/JSON array file to the row order of a vectors CSV
whose first column is an id computed from the media path (e.g., md5(path)).

Usage
  python -m knowledge3d.tools.align_meta_to_csv \
    --csv viewer/public/balanced/video/msrvtt.clip.csv \
    --meta-in ../Knowledge3D.local/datasets/msrvtt/meta.jsonl \
    --out viewer/public/balanced/video/msrvtt.aligned.meta.json
"""

import argparse
import csv
import json
from hashlib import md5
from pathlib import Path
from typing import Dict, List


def read_meta(meta_path: Path) -> Dict[str, dict]:
    txt = meta_path.read_text(encoding='utf-8')
    out: Dict[str, dict] = {}
    if txt.lstrip().startswith('['):
        arr = json.loads(txt)
        if not isinstance(arr, list):
            return {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            p = str(it.get('path') or it.get('video') or it.get('file') or '')
            if not p:
                continue
            hid = md5(p.encode('utf-8')).hexdigest()[:16]
            out[hid] = it
    else:
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                it = json.loads(s)
            except Exception:
                continue
            if not isinstance(it, dict):
                continue
            p = str(it.get('path') or it.get('video') or it.get('file') or '')
            if not p:
                continue
            hid = md5(p.encode('utf-8')).hexdigest()[:16]
            out[hid] = it
    return out


def align(csv_path: Path, meta_in: Path, out_path: Path) -> None:
    by_id = read_meta(meta_in)
    arr: List[dict] = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            rid = row[0]
            m = dict(by_id.get(rid, {}))
            # ensure label/text presence minimally
            lab = str(m.get('label') or m.get('caption') or m.get('title') or rid)
            txt = str(m.get('text') or m.get('caption') or '')
            m['label'] = lab
            if txt:
                m['text'] = txt
            arr.append(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Aligned meta -> {out_path} (rows={len(arr)})")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description='Align meta JSON(L) to vectors CSV row order')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--meta-in', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    align(Path(args.csv), Path(args.meta_in), Path(args.out))


if __name__ == '__main__':
    main()

