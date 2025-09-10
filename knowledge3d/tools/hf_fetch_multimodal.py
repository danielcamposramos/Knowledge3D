from __future__ import annotations

"""
Fetch small slices of audio/video datasets from Hugging Face with captions.

Supports datasets that host media on HF (preferred). Saves media files to a
local folder and writes a JSONL metadata file with id, caption(s), and path.

Examples
  # AudioCaps (audio + captions)
  python -m knowledge3d.tools.hf_fetch_multimodal \
    --dataset confit/audiocaps --split train \
    --kind audio --limit 30000 \
    --out-dir ../Knowledge3D.local/datasets/audiocaps

  # MSR-VTT (videos + captions)
  python -m knowledge3d.tools.hf_fetch_multimodal \
    --dataset friedrichor/MSR-VTT --split train \
    --kind video --limit 20000 \
    --out-dir ../Knowledge3D.local/datasets/msrvtt

Notes
- Requires: datasets, soundfile (for audio write) and av (for video copy when needed).
- This script is best‑effort; it only processes rows where a local file path
  is available via the datasets cache (feature type Audio/Video) or a direct URL
  that can be downloaded.
"""

import argparse
import json
import os
import shutil
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_name(text: str) -> str:
    base = "".join(c for c in text if c.isalnum() or c in ("-", "_"))
    return base[:64] or "item"


def _copy_or_download_media(item: Dict[str, Any], kind: str, out_dir: Path) -> Optional[Path]:
    # Prefer datasets cached file path
    media = item.get(kind)
    if isinstance(media, dict):
        # Audio/Video feature returns a dict with 'path'
        p = media.get("path") or media.get("filepath")
        if p and os.path.isfile(p):
            src = Path(p)
            ext = src.suffix or (".wav" if kind == "audio" else ".mp4")
            hid = md5(src.as_posix().encode("utf-8")).hexdigest()[:16]
            dest = out_dir / f"{hid}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)
            return dest
    # Try URL field fallbacks
    url_keys = [f"{kind}_url", "url", "link"]
    for k in url_keys:
        u = item.get(k)
        if isinstance(u, str) and u.startswith("http"):
            try:
                import urllib.request as ur
                ext = ".wav" if kind == "audio" else ".mp4"
                hid = md5(u.encode("utf-8")).hexdigest()[:16]
                dest = out_dir / f"{hid}{ext}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    req = ur.Request(u, headers={"User-Agent": "Mozilla/5.0 K3D-fetch/1.0"})
                    with ur.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                        chunk = r.read(1024*1024)
                        while chunk:
                            f.write(chunk)
                            chunk = r.read(1024*1024)
                return dest
            except Exception:
                return None
    return None


def _extract_caption(item: Dict[str, Any]) -> Optional[str]:
    # Common keys in AudioCaps/Clotho/MSR-VTT/VATEX/WebVid subsets
    for key in ("caption", "captions", "sentence", "sentences", "text", "description"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            # pick the first non-empty
            for s in v:
                if isinstance(s, str) and s.strip():
                    return s.strip()
    return None


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Fetch small audio/video slices with captions from HF datasets")
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g., confit/audiocaps")
    ap.add_argument("--name", help="dataset config name when required (e.g., train_9k)")
    ap.add_argument("--split", default="train", help="split name (default train)")
    ap.add_argument("--kind", choices=["audio", "video"], required=True)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    media_dir = out_dir / "media"
    meta_path = out_dir / "meta.jsonl"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import datasets
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit("Please install the 'datasets' package: pip install datasets soundfile av") from e

    if args.name:
        ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    # Avoid decoding heavy Audio/Video features; we only need cached file paths
    try:
        if args.kind == "audio":
            from datasets import Audio  # type: ignore
            ds = ds.cast_column("audio", Audio(decode=False))
        elif args.kind == "video":
            from datasets import Video  # type: ignore
            ds = ds.cast_column("video", Video(decode=False))
    except Exception:
        pass
    written = 0
    with meta_path.open("w", encoding="utf-8") as meta_out:
        for item in ds:
            try:
                cap = _extract_caption(item) or ""
                dest = _copy_or_download_media(item, args.kind, media_dir)
                if dest is None:
                    continue
                rid = md5((dest.name + cap).encode("utf-8")).hexdigest()[:16]
                rec = {"id": rid, "path": dest.as_posix(), "caption": cap}
                meta_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written % 500 == 0:
                    print(f"written={written}")
                if written >= args.limit:
                    break
            except Exception:
                continue
    print(f"Wrote {written} items to {meta_path}")


if __name__ == "__main__":
    main()
