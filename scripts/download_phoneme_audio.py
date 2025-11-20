#!/usr/bin/env python3
"""
Download open phoneme and letter-name audio from Lingua Libre (Wikidata),
and optionally pull pronunciation files from a Wikimedia Commons category
for languages that are better covered there (e.g., Japanese kana syllables).

Queries the Wikidata SPARQL endpoint for lexeme forms (not items) that
carry pronunciation audio (P443) for a target language, then downloads
the audio files to:

    /K3D/K3D_llama_cpp/datasets/audio/phoneme_external/<lang>/<id>.ogg

Defaults:
    langs: en, es, pt, zh
    max-per-lang: 200

Examples:
    # Wikidata/Lingua Libre pull
    PYTHONPATH=. python3 scripts/download_phoneme_audio.py --langs en es pt zh --max-per-lang 200

    # Commons pull (Japanese kana syllables, recursive into subcategories)
    PYTHONPATH=. python3 scripts/download_phoneme_audio.py \\
        --skip-wikidata \\
        --commons-category \"Category:Pronunciation of Japanese syllables\" \\
        --commons-recursive \\
        --commons-output /K3D/K3D_llama_cpp/datasets/audio/phoneme_external/ja_kana
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

OUTPUT_ROOT = Path("/K3D/K3D_llama_cpp/datasets/audio/phoneme_external")
SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_CATEGORY_DEFAULT = "Category:Pronunciation of Japanese syllables"
COMMONS_USER_AGENT = "Knowledge3D/1.0"
# Language QIDs for filtering; override with CLI if needed.
# Include dialects the users highlighted.
LANG_QID = {
    "en": "Q1860",
    "es": "Q1321",
    "pt": "Q5146",
    "pt-br": "Q750553",
    "ja": "Q5287",
    "zh": "Q9192",  # Mandarin Chinese (generic)
    "zh-cn": "Q24841726",  # Simplified Chinese
}


def build_query(lang: str, limit: int) -> str:
    # Fetch lexeme forms with pronunciation audio (P443) for the target language.
    qid = LANG_QID.get(lang, "Q1860")
    return f"""
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?lex ?form ?formLabel ?audio WHERE {{
  ?lex dct:language wd:{qid} ;
       ontolex:lexicalForm ?form .
  ?form ontolex:representation ?formLabel ;
        wdt:P443 ?audio .
  FILTER(STRLEN(?formLabel) < 3)
}}
LIMIT {limit}"""


def fetch_entries(lang: str, limit: int) -> List[Dict[str, str]]:
    headers = {"Accept": "application/sparql-results+json", "User-Agent": COMMONS_USER_AGENT}
    r = requests.get(SPARQL_URL, params={"query": build_query(lang, limit)}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    entries: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        form = b["form"]["value"]
        lex = b["lex"]["value"]
        audio = b["audio"]["value"]
        label = b.get("formLabel", {}).get("value", "")
        form_id = form.split("/")[-1]
        lex_id = lex.split("/")[-1]
        entries.append({"form_id": form_id, "lex_id": lex_id, "audio": audio, "label": label})
    return entries


def download_file(url: str, dest: Path, session: Optional[requests.Session] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    sess = session or requests.Session()
    headers = {"User-Agent": COMMONS_USER_AGENT}
    with sess.get(url, stream=True, timeout=30, headers=headers) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def list_commons_category_files(
    category: str,
    allowed_exts: Sequence[str],
    max_files: Optional[int] = None,
    session: Optional[requests.Session] = None,
    recursive: bool = False,
) -> List[Tuple[str, str]]:
    """Return (filename, download_url) for all files in a Commons category (and optionally subcategories)."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": COMMONS_USER_AGENT})

    normalized_exts = {ext.lower() for ext in allowed_exts}
    results: List[Tuple[str, str]] = []
    seen_files = set()
    to_visit = [category]
    visited = set()

    def fetch_members(cat: str) -> Dict[str, List[Dict[str, str]]]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmtype": "subcat|file" if recursive else "file",
            "cmlimit": "max",
            "format": "json",
        }
        items: List[Dict[str, str]] = []
        while True:
            resp = sess.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get("query", {}).get("categorymembers", []))
            if "continue" not in data:
                break
            params.update(data["continue"])
        subcats = [i["title"] for i in items if i.get("ns") == 14] if recursive else []
        files = [i for i in items if i.get("ns") == 6 or not recursive]
        return {"subcats": subcats, "files": files}

    while to_visit:
        cat = to_visit.pop()
        if cat in visited:
            continue
        visited.add(cat)
        members = fetch_members(cat)
        if recursive:
            for sub in members["subcats"]:
                if sub not in visited:
                    to_visit.append(sub)
        for item in members["files"]:
            filename = item["title"].replace("File:", "")
            ext = Path(filename).suffix.lower()
            if normalized_exts and ext not in normalized_exts:
                continue
            if filename in seen_files:
                continue
            safe_name = urllib.parse.quote(filename)
            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{safe_name}"
            results.append((filename, url))
            seen_files.add(filename)
            if max_files is not None and len(results) >= max_files:
                return results
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt", "zh"], help="Languages (ISO 639-1) to fetch")
    ap.add_argument("--max-per-lang", type=int, default=200, help="Max items per language")
    ap.add_argument(
        "--commons-category",
        help=(
            "Optional Wikimedia Commons category to download audio files from "
            "(e.g., 'Category:Pronunciation of Japanese syllables' for kana)."
        ),
    )
    ap.add_argument(
        "--commons-output",
        type=Path,
        help="Destination directory for Commons downloads (default: ja_kana under phoneme_external).",
    )
    ap.add_argument(
        "--commons-max",
        type=int,
        default=None,
        help="Optional cap on Commons downloads (useful for quick verification).",
    )
    ap.add_argument(
        "--commons-recursive",
        action="store_true",
        help="Recurse into Commons subcategories (needed for full kana coverage).",
    )
    ap.add_argument(
        "--commons-exts",
        nargs="+",
        default=[".ogg", ".oga", ".wav"],
        help="Allowed extensions when pulling from Commons categories.",
    )
    ap.add_argument(
        "--skip-wikidata",
        action="store_true",
        help="Skip the Wikidata SPARQL fetch and only run Commons category downloads.",
    )
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": COMMONS_USER_AGENT})

    if not args.skip_wikidata:
        for lang in args.langs:
            try:
                entries = fetch_entries(lang, args.max_per_lang)
            except Exception as exc:
                print(f"[warn] failed to query {lang}: {exc}", file=sys.stderr)
                continue
            print(f"[info] {lang}: {len(entries)} entries")
            for e in entries:
                audio_url = e["audio"]
                # Preserve server extension if possible.
                ext = Path(audio_url.split("?")[0]).suffix or ".ogg"
                fname = f"{e['form_id']}{ext}"
                dest = OUTPUT_ROOT / lang / fname
                try:
                    download_file(audio_url, dest, session=session)
                except Exception as exc:
                    print(f"[warn] download failed {audio_url}: {exc}", file=sys.stderr)
                    continue

    if args.commons_category:
        commons_dir = args.commons_output or OUTPUT_ROOT / "ja_kana"
        try:
            files = list_commons_category_files(
                args.commons_category,
                allowed_exts=args.commons_exts,
                max_files=args.commons_max,
                session=session,
                recursive=args.commons_recursive,
            )
        except Exception as exc:
            print(f"[warn] Commons discovery failed: {exc}", file=sys.stderr)
            return
        print(f"[info] commons {args.commons_category}: {len(files)} files")
        for filename, url in files:
            dest = commons_dir / filename
            try:
                download_file(url, dest, session=session)
            except Exception as exc:
                print(f"[warn] Commons download failed {url}: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
