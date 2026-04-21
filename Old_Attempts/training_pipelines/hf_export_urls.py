from __future__ import annotations

"""
Export URL+caption JSONL from an HF dataset (streaming) for yt-dlp batching.

Usage
  python -m knowledge3d.tools.hf_export_urls \
    --dataset lmms-lab/vatex_from_url --split train \
    --out /home/daniel/K3D_llama_cpp/datasets/vatex_raw/urls.jsonl \
    --limit 5000
"""

import argparse
import json
from pathlib import Path
from typing import List


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Export URL+caption JSONL from HF dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", help="dataset config name when required (e.g., vatex_test)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url-keys", nargs="*", default=["url", "video_url", "VideoURL"])  # probe order
    ap.add_argument("--cap-keys", nargs="*", default=["caption", "text", "sentence", "sentences"])  # probe order
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit("pip install datasets") from e
    if args.name:
        ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for x in ds:
            u = None
            for k in args.url_keys:
                v = x.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    u = v; break
                if isinstance(v, list) and v and isinstance(v[0], str) and v[0].startswith("http"):
                    u = v[0]; break
            if not u:
                continue
            cap = None
            for k in args.cap_keys:
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    cap = v.strip(); break
                if isinstance(v, list) and v:
                    cap = str(v[0]); break
            rec = {"url": u}
            if cap:
                rec["caption"] = cap
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"Wrote {n} to {out}")


if __name__ == "__main__":
    main()
