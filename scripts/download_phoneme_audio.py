#!/usr/bin/env python3
"""
Download open phoneme and letter-name audio from Lingua Libre (Wikidata).

Queries the Wikidata SPARQL endpoint for items that are instance of
phoneme (Q708031) or letter (Q9779) with audio and a given language,
then downloads the audio files to:

    /K3D/K3D_llama_cpp/datasets/audio/phoneme_external/<lang>/<id>.ogg

Defaults:
    langs: en, es, pt, zh
    max-per-lang: 200

Usage:
    PYTHONPATH=. python3 scripts/download_phoneme_audio.py --langs en es pt zh --max-per-lang 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

import requests

OUTPUT_ROOT = Path("/K3D/K3D_llama_cpp/datasets/audio/phoneme_external")
SPARQL_URL = "https://query.wikidata.org/sparql"
# Language QIDs for filtering; override with CLI if needed.
LANG_QID = {
    "en": "Q1860",
    "es": "Q1321",
    "pt": "Q5146",
    "zh": "Q9192",  # Mandarin Chinese
}


def build_query(lang: str, limit: int) -> str:
    # Fetch any item (usually lexemes) with pronunciation audio (P443) and language (P407).
    qid = LANG_QID.get(lang, "Q1860")
    return f"""
SELECT ?item ?itemLabel ?audio WHERE {{
  ?item wdt:P443 ?audio .
  ?item wdt:P407 wd:{qid} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
  FILTER(STRLEN(?itemLabel) < 4)
}}
LIMIT {limit}
"""


def fetch_entries(lang: str, limit: int) -> List[Dict[str, str]]:
    headers = {"Accept": "application/sparql-results+json"}
    r = requests.get(SPARQL_URL, params={"query": build_query(lang, limit)}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    entries = []
    for b in data.get("results", {}).get("bindings", []):
        item = b["item"]["value"]
        item_id = item.split("/")[-1]
        audio = b["audio"]["value"]
        label = b.get("itemLabel", {}).get("value", "")
        entries.append({"id": item_id, "audio": audio, "label": label})
    return entries


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with requests.get(url, stream=True, timeout=30, headers={"User-Agent": "K3D-pho-loader"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "es", "pt", "zh"], help="Languages (ISO 639-1) to fetch")
    ap.add_argument("--max-per-lang", type=int, default=200, help="Max items per language")
    args = ap.parse_args()

    for lang in args.langs:
        try:
            entries = fetch_entries(lang, args.max_per_lang)
        except Exception as exc:
            print(f"[warn] failed to query {lang}: {exc}", file=sys.stderr)
            continue
        print(f"[info] {lang}: {len(entries)} entries")
        for e in entries:
            audio_url = e["audio"]
            fname = f"{e['id']}.ogg"
            dest = OUTPUT_ROOT / lang / fname
            try:
                download_file(audio_url, dest)
            except Exception as exc:
                print(f"[warn] download failed {audio_url}: {exc}", file=sys.stderr)
                continue
    print("Done.")


if __name__ == "__main__":
    main()
