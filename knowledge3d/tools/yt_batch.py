from __future__ import annotations

"""
Batch downloader using yt-dlp for audio/video JSONL manifests.

Input JSONL fields (any subset):
- url: full media URL (preferred)
- ytid / youtube_id: YouTube video id (builds url automatically)
- start / start_time: float seconds (optional)
- end / end_time: float seconds (optional)
- caption: optional text (ignored for download)

Usage
  python -m knowledge3d.tools.yt_batch \
    --jsonl /path/to/urls.jsonl --out-dir /path/to/media \
    --kind video --limit 2000 --concurrency 4 --log /tmp/vatex.log

  python -m knowledge3d.tools.yt_batch \
    --jsonl /home/daniel/K3D_llama_cpp/datasets/audiocaps_raw/manifest.jsonl \
    --out-dir /home/daniel/K3D_llama_cpp/datasets/audiocaps_raw/media \
    --kind audio --limit 200 --concurrency 4 --log /home/daniel/K3D_llama_cpp/logs/audiocaps_ytdlp.log
"""

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional


def build_url(item: Dict[str, object]) -> Optional[str]:
    u = item.get("url")
    if isinstance(u, str) and u.startswith("http"):
        return u
    for k in ("ytid", "youtube_id"):
        vid = item.get(k)
        if isinstance(vid, str) and vid:
            return f"https://www.youtube.com/watch?v={vid}"
    return None


def get_times(item: Dict[str, object]) -> tuple[Optional[float], Optional[float]]:
    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None
    ss = item.get("start_time") or item.get("start")
    ee = item.get("end_time") or item.get("end")
    return _to_float(ss), _to_float(ee)


def worker(tasks: "queue.Queue[Dict[str, object]]", out_dir: Path, kind: str, log_path: Path) -> None:
    while True:
        try:
            item = tasks.get_nowait()
        except queue.Empty:
            return
        url = build_url(item)
        if not url:
            tasks.task_done(); continue
        ss, ee = get_times(item)
        cmd = [
            "yt-dlp",
            "-q",
            "--no-playlist",
            "--no-part",
            "-o",
            str(out_dir / "%(id)s.%(ext)s"),
        ]
        if kind == "audio":
            cmd += ["-f", "bestaudio"]
        else:
            cmd += ["-f", "bestvideo+bestaudio/best"]
        if ss is not None and ee is not None and ee > ss:
            cmd += ["--download-sections", f"*{ss}-{ee}"]
        cmd += [url]
        # Append logs
        with log_path.open("a", encoding="utf-8") as logf:
            try:
                subprocess.run(cmd, stdout=logf, stderr=logf, check=False)
            except Exception as e:
                logf.write(f"ERROR: {e}\n")
        tasks.task_done()


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Batch download via yt-dlp from JSONL manifest")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kind", choices=["audio", "video"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    src = Path(args.jsonl)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log); log_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: "queue.Queue[Dict[str, object]]" = queue.Queue()
    loaded = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                tasks.put(j)
                loaded += 1
                if args.limit and loaded >= args.limit:
                    break
            except Exception:
                continue
    if loaded == 0:
        print("No tasks loaded; exiting")
        return
    threads = []
    for _ in range(max(1, int(args.concurrency))):
        t = threading.Thread(target=worker, args=(tasks, out_dir, args.kind, log_path), daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join()
    print(f"Completed: {loaded} attempted. Logs at {log_path}")


if __name__ == "__main__":
    main()

