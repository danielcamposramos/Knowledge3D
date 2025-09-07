from __future__ import annotations

"""
Extract COCO captions into a text lines file and aligned metadata array.

Usage
  python -m knowledge3d.tools.coco_captions_to_text \
    --captions /path/to/annotations/captions_train2017.json \
    --images-dir /path/to/train2017/train2017 \
    --out-text ../Knowledge3D.local/datasets/coco_captions.txt \
    --out-meta  ../Knowledge3D.local/datasets/coco_captions.meta.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="COCO captions -> text + meta")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()
    cap_path = Path(args.captions)
    img_dir = Path(args.images_dir)
    out_txt = Path(args.out_text)
    out_meta = Path(args.out_meta)
    j = json.loads(cap_path.read_text(encoding="utf-8"))
    id_to_file: Dict[int, str] = {}
    for im in j.get("images", []):
        try:
            id_to_file[int(im["id"])]=str(im["file_name"])  # type: ignore
        except Exception:
            continue
    lines: List[str] = []
    meta: List[dict] = []
    seen = set()
    for ann in j.get("annotations", []):
        try:
            fid = id_to_file.get(int(ann["image_id"]))
            cap = str(ann.get("caption") or "").strip()
            if not fid or not cap:
                continue
            key = (fid, cap)
            if key in seen:
                continue
            seen.add(key)
            lines.append(cap)
            meta.append({
                "label": cap[:48] + ("..." if len(cap) > 48 else ""),
                "text": cap,
                "type": "text",
                "image": str((img_dir / fid).as_posix()),
            })
        except Exception:
            continue
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(lines)} captions -> {out_txt} and meta -> {out_meta}")


if __name__ == "__main__":
    main()

