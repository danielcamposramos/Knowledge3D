from __future__ import annotations

"""
Download MSR-VTT videos with yt-dlp into a local folder using URLs from the
Hugging Face dataset 'friedrichor/MSR-VTT'.

Usage
  python -m knowledge3d.tools.download_msrvtt_yt \
    --out ../Knowledge3D.local/datasets/msrvtt_dl \
    --split train --config train_9k --limit 400

Requires: yt-dlp installed in the active environment.
"""

import argparse
import json
import subprocess
from hashlib import md5
from pathlib import Path
from typing import Optional


def pick_url(ex: dict) -> Optional[str]:
    for k in ("video_url", "url", "link"):
        v = ex.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    # Sometimes nested
    v = ex.get("video")
    if isinstance(v, dict):
        u = v.get("url") or v.get("path")
        if isinstance(u, str) and u.startswith("http"):
            return u
    return None


def pick_caption(ex: dict) -> str:
    for k in ("caption", "captions", "sentence", "sentences", "text", "description"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            for s in v:
                if isinstance(s, str) and s.strip():
                    return s.strip()
    return ""


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Download MSR-VTT videos with yt-dlp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default="train_9k")
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()
    out = Path(args.out); media = out / "media"; out.mkdir(parents=True, exist_ok=True); media.mkdir(parents=True, exist_ok=True)
    meta = out / "meta.jsonl"
    from datasets import load_dataset
    ds = load_dataset("friedrichor/MSR-VTT", args.config, split=str(args.split), streaming=True)
    written = 0
    with meta.open("w", encoding="utf-8") as w:
        for ex in ds:
            url = pick_url(ex)
            if not url:
                continue
            cap = pick_caption(ex)
            hid = md5(url.encode("utf-8")).hexdigest()[:16]
            dest = media / f"{hid}.mp4"
            if not dest.exists():
                # yt-dlp download
                cmd = [
                    "yt-dlp", url,
                    "-f", "mp4/best", "--no-playlist",
                    "-o", str(dest),
                    "--quiet", "--no-warnings",
                ]
                try:
                    subprocess.run(cmd, check=True)
                except Exception:
                    continue
            w.write(json.dumps({"id": hid, "path": dest.as_posix(), "caption": cap, "url": url}, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"downloaded={written}")
            if written >= int(args.limit):
                break
    print(f"Wrote {written} items to {meta}")


if __name__ == "__main__":
    main()

