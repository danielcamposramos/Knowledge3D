"""
Build a tiny alias table for dataset labels using Wikipedia redirects.

Usage
  python3 -m knowledge3d.tools.build_aliases --labels labels.txt --out viewer/public/aliases.json

labels.txt: one label per line (e.g., from dataset metadata labels)
Output JSON format: { "items": [ {"alias": "Gates of Paradise", "label": "Fine art"}, ... ] }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import urllib.parse
import urllib.request


API = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=redirects&rdlimit=50&titles={title}&origin=*"


def fetch_redirects(title: str) -> List[str]:
    url = API.format(title=urllib.parse.quote(title))
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    aliases: List[str] = []
    for _, pg in pages.items():
        for r in pg.get("redirects", []) or []:
            a = str(r.get("title") or "").strip()
            if a and a.lower() != title.lower():
                aliases.append(a)
    return aliases


def build(labels: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for i, lab in enumerate(labels):
        lab = lab.strip()
        if not lab:
            continue
        try:
            aliases = fetch_redirects(lab)
        except Exception:
            aliases = []
        if aliases:
            out[lab] = aliases
        # polite pacing
        time.sleep(0.1)
    return out


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Build alias mapping from Wikipedia redirects")
    p.add_argument("--labels", required=True, help="Path to file with labels (one per line)")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()
    labels = [ln.strip() for ln in Path(args.labels).read_text(encoding="utf-8").splitlines() if ln.strip()]
    mapping = build(labels)
    items = []
    for lab, aliases in mapping.items():
        for a in aliases:
            items.append({"alias": a, "label": lab})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(items)} items")


if __name__ == "__main__":
    main()

