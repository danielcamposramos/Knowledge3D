#!/usr/bin/env python3
"""Collect and download diagram images from foundational drawing source HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


_IMG_RE = re.compile(
    r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>",
    flags=re.IGNORECASE,
)
_EXT_ALLOW = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
_KEYWORD_ALLOW = {
    "bezier",
    "curve",
    "vector",
    "matrix",
    "projection",
    "transform",
    "rotation",
    "clip",
    "raster",
    "geometry",
    "grid",
    "spline",
    "shader",
    "normal",
    "camera",
}


def _extract_image_urls(html: str, base_url: str) -> list[str]:
    found = []
    for match in _IMG_RE.findall(html):
        raw = unescape(match.strip())
        if not raw:
            continue
        full = urljoin(base_url, raw)
        found.append(full)
    return found


def _is_candidate(url: str) -> bool:
    parsed = urlparse(url)
    path_lower = (parsed.path or "").lower()
    ext_ok = any(path_lower.endswith(ext) for ext in _EXT_ALLOW)
    kw_ok = any(kw in url.lower() for kw in _KEYWORD_ALLOW)
    return ext_ok or kw_ok


def _download(url: str, out_path: Path, timeout: int) -> tuple[bool, str]:
    req = Request(
        url,
        headers={
            "User-Agent": "Knowledge3D-ImageCollector/1.0 (+https://github.com/)",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return False, "empty_response"
        out_path.write_bytes(data)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download diagram images from source HTML.")
    parser.add_argument(
        "--manifest",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/manifest.jsonl",
        help="Source manifest path",
    )
    parser.add_argument(
        "--raw-dir",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/raw_html",
        help="Raw HTML dir",
    )
    parser.add_argument(
        "--image-dir",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/raw_images",
        help="Image output directory",
    )
    parser.add_argument(
        "--output-catalog",
        default="/K3D/Knowledge3D.local/datasets/foundational_drawing_sources/image_catalog.jsonl",
        help="Image catalog output JSONL",
    )
    parser.add_argument("--max-images", type=int, default=500, help="Maximum images to download")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout (seconds)")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    image_dir = Path(args.image_dir)
    output_catalog = Path(args.output_catalog)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_catalog.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            rows.append(row)

    candidates: list[dict[str, str]] = []
    seen = set()
    for row in rows:
        source_id = str(row["source_id"])
        base_url = str(row.get("final_url") or row.get("url"))
        html_path = raw_dir / f"{source_id}.html"
        if not html_path.exists():
            continue
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        urls = _extract_image_urls(html, base_url)
        for url in urls:
            if not _is_candidate(url):
                continue
            if url in seen:
                continue
            seen.add(url)
            candidates.append({"source_id": source_id, "url": url})

    downloaded = []
    failures = 0
    for idx, cand in enumerate(candidates[: args.max_images]):
        url = cand["url"]
        source_id = cand["source_id"]
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        if ext not in _EXT_ALLOW:
            ext = ".img"
        name_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        out_path = image_dir / f"{source_id}_{name_hash}{ext}"
        ok, err = _download(url, out_path, timeout=args.timeout)
        if not ok:
            failures += 1
            continue
        size = out_path.stat().st_size
        downloaded.append(
            {
                "source_id": source_id,
                "url": url,
                "local_path": str(out_path),
                "bytes": size,
                "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            }
        )
        if idx % 25 == 0:
            print(f"[images] downloaded {idx + 1}/{min(len(candidates), args.max_images)}")

    with output_catalog.open("w", encoding="utf-8") as fh:
        for row in downloaded:
            fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(
        f"Image collection done: {len(downloaded)} downloaded, {failures} failed, "
        f"catalog={output_catalog}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

