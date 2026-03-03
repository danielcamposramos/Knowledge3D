#!/usr/bin/env python3
"""Scrape H. Blazer Substack posts with randomized polite delays.

Behavior constraints implemented:
- Fetch one page at a time.
- Random delay between pages.
- Minimum delay is configurable and defaults to 60 seconds.
- Resume-safe state checkpointing after every page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SITEMAP = "https://hblazer.substack.com/sitemap.xml"
DEFAULT_KEYWORDS = [
    "sovereign",
    "sovereignty",
    "boundary",
    "boundaries",
    "charter",
    "autonomous",
    "ai",
    "llm",
    "ethic",
    "moral",
    "privacy",
    "transparency",
    "control",
    "governance",
    "system",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_text(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Knowledge3D-PMKR-ResearchBot/1.0 "
                "(respectful; contact: internal-pm-kr@w3.org)"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        encoding = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="replace")


def _parse_sitemap_urls(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for loc in root.findall("sm:url/sm:loc", ns):
        text = (loc.text or "").strip()
        if text:
            urls.append(text)
    return urls


def _url_slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1].lower()


def _is_relevant(url: str, keywords: list[str]) -> bool:
    if "/p/" not in url:
        return False
    slug = _url_slug(url)
    return any(kw in slug for kw in keywords)


def _safe_name(url: str) -> str:
    slug = _url_slug(url)
    slug = re.sub(r"[^a-z0-9\-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        slug = "post"
    digest = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_initial_queue(
    sitemap_url: str,
    keywords: list[str],
    include_urls: list[str],
    all_posts: bool,
) -> list[str]:
    xml_text = _fetch_text(sitemap_url, timeout=60.0)
    all_urls = _parse_sitemap_urls(xml_text)
    if all_posts:
        queue = [u for u in all_urls if "/p/" in u]
    else:
        queue = [u for u in all_urls if _is_relevant(u, keywords)]
    for forced in include_urls:
        if forced not in queue:
            queue.insert(0, forced)
    # De-duplicate while preserving order.
    seen = set()
    dedup = []
    for url in queue:
        if url in seen:
            continue
        seen.add(url)
        dedup.append(url)
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitemap-url", default=DEFAULT_SITEMAP)
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keyword list for slug filtering.",
    )
    parser.add_argument(
        "--all-posts",
        action="store_true",
        help="Ignore keyword filtering and scrape all /p/ posts from sitemap.",
    )
    parser.add_argument(
        "--include-url",
        action="append",
        default=["https://hblazer.substack.com/p/the-sovereign-systems-charter"],
        help="URL to force-include (can be repeated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Knowledge3D.local/scrapes/hblazer_substack"),
    )
    parser.add_argument("--min-delay-sec", type=int, default=60)
    parser.add_argument("--max-delay-sec", type=int, default=180)
    parser.add_argument("--backoff-min-sec", type=int, default=900)
    parser.add_argument("--backoff-max-sec", type=int, default=3600)
    parser.add_argument(
        "--retry-http-codes",
        default="403,429",
        help="Comma-separated HTTP status codes that trigger backoff+retry.",
    )
    parser.add_argument(
        "--max-retries-per-url",
        type=int,
        default=8,
        help="Maximum retries for URLs that hit retryable HTTP codes before marking failed.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum pages to fetch this run (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Optional RNG seed (0 = system random).")
    args = parser.parse_args()

    if args.min_delay_sec < 60:
        raise SystemExit("--min-delay-sec must be >= 60.")
    if args.max_delay_sec < args.min_delay_sec:
        raise SystemExit("--max-delay-sec must be >= --min-delay-sec.")
    if args.backoff_min_sec < args.min_delay_sec:
        raise SystemExit("--backoff-min-sec must be >= --min-delay-sec.")
    if args.backoff_max_sec < args.backoff_min_sec:
        raise SystemExit("--backoff-max-sec must be >= --backoff-min-sec.")

    output_dir = args.output_dir
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    index_jsonl = output_dir / "index.jsonl"

    state = _load_state(state_path)
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    include_urls = [u.strip() for u in args.include_url if u.strip()]

    if not state or not isinstance(state.get("pending_urls"), list):
        pending = _build_initial_queue(
            args.sitemap_url,
            keywords,
            include_urls,
            bool(args.all_posts),
        )
        state = {
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "sitemap_url": args.sitemap_url,
            "keywords": keywords,
            "all_posts": bool(args.all_posts),
            "include_urls": include_urls,
            "min_delay_sec": args.min_delay_sec,
            "max_delay_sec": args.max_delay_sec,
            "backoff_min_sec": args.backoff_min_sec,
            "backoff_max_sec": args.backoff_max_sec,
            "pending_urls": pending,
            "completed": [],
            "failed": [],
            "retry_counts": {},
        }
        _save_state(state_path, state)
        print(f"[init] queued_urls={len(pending)}")
    else:
        print(
            f"[resume] pending={len(state.get('pending_urls', []))} "
            f"completed={len(state.get('completed', []))} failed={len(state.get('failed', []))}"
        )

    retry_http_codes = {
        int(token.strip())
        for token in str(args.retry_http_codes).split(",")
        if token.strip().isdigit()
    }
    state.setdefault("retry_counts", {})

    if args.seed:
        rng = random.Random(args.seed)
    else:
        rng = random.SystemRandom()

    processed_this_run = 0
    while state["pending_urls"]:
        if args.max_pages > 0 and processed_this_run >= args.max_pages:
            print(f"[stop] reached --max-pages={args.max_pages}")
            break

        if processed_this_run > 0:
            delay = rng.randint(args.min_delay_sec, args.max_delay_sec)
            print(f"[sleep] delay_sec={delay}")
            time.sleep(delay)
        else:
            delay = 0

        url = state["pending_urls"][0]
        print(f"[fetch] url={url}")
        started_at = _now_iso()
        try:
            html = _fetch_text(url, timeout=90.0)
            title = _extract_title(html)
            text = _html_to_text(html)
            stem = _safe_name(url)
            html_path = pages_dir / f"{stem}.html"
            txt_path = pages_dir / f"{stem}.txt"
            meta_path = pages_dir / f"{stem}.meta.json"

            html_path.write_text(html, encoding="utf-8", errors="ignore")
            txt_path.write_text(text, encoding="utf-8", errors="ignore")
            meta = {
                "url": url,
                "title": title,
                "fetched_at": _now_iso(),
                "started_at": started_at,
                "delay_before_sec": delay,
                "html_path": str(html_path),
                "txt_path": str(txt_path),
                "text_len": len(text),
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
            with index_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=True) + "\n")

            state["completed"].append(meta)
            state["pending_urls"].pop(0)
            state["updated_at"] = _now_iso()
            _save_state(state_path, state)
            processed_this_run += 1
            print(
                f"[ok] title={title[:120]!r} "
                f"remaining={len(state['pending_urls'])} completed={len(state['completed'])}"
            )
        except urllib.error.HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            if code in retry_http_codes:
                retry_key = url
                retries = int(state["retry_counts"].get(retry_key, 0)) + 1
                state["retry_counts"][retry_key] = retries
                if retries <= int(args.max_retries_per_url):
                    backoff = rng.randint(args.backoff_min_sec, args.backoff_max_sec)
                    state["updated_at"] = _now_iso()
                    _save_state(state_path, state)
                    print(
                        f"[retry] code={code} retries={retries}/{args.max_retries_per_url} "
                        f"backoff_sec={backoff} url={url}"
                    )
                    time.sleep(backoff)
                    continue
            err = {
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": _now_iso(),
                "delay_before_sec": delay,
            }
            state["failed"].append(err)
            state["pending_urls"].pop(0)
            state["updated_at"] = _now_iso()
            _save_state(state_path, state)
            print(f"[fail] {err['error']}")
            processed_this_run += 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            err = {
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": _now_iso(),
                "delay_before_sec": delay,
            }
            state["failed"].append(err)
            state["pending_urls"].pop(0)
            state["updated_at"] = _now_iso()
            _save_state(state_path, state)
            print(f"[fail] {err['error']}")
            processed_this_run += 1

    print(
        f"[done] pending={len(state.get('pending_urls', []))} "
        f"completed={len(state.get('completed', []))} failed={len(state.get('failed', []))}"
    )
    print(f"[state] {state_path}")
    print(f"[index] {index_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
