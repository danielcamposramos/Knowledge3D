"""Build a tablet-friendly Wikipedia corpus JSONL from the local AI topics text.

Reads data/ai_wiki_corpus.txt and writes viewer/public/galaxy/working/wikipedia_corpus.jsonl
in the `{question, answer, source}` JSONL format expected by the fused head.

Usage
  conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.build_wikipedia_corpus \
      --max-lines 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .fetch_wiki_corpus import OUT_TXT as _OUT_TXT, DEFAULT_TOPICS, fetch_plain, iter_lines


ROOT = Path(__file__).resolve().parents[2]
OUT_JSONL = ROOT / "viewer/public/galaxy/working/wikipedia_corpus.jsonl"


def ensure_ai_corpus(topics: List[str]) -> Path:
    if _OUT_TXT.exists() and _OUT_TXT.stat().st_size > 0:
        return _OUT_TXT
    _OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with _OUT_TXT.open("w", encoding="utf-8") as fh:
        for t in topics:
            try:
                text = fetch_plain(t)
            except Exception:
                continue
            for ln in iter_lines(text):
                s = ln.strip()
                if len(s) < 80 or len(s) > 600 or s in seen:
                    continue
                fh.write(s + "\n")
                seen.add(s)
    return _OUT_TXT


def build(max_lines: int = 0) -> Path:
    src = ensure_ai_corpus(DEFAULT_TOPICS)
    lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_lines and max_lines > 0:
        lines = lines[: int(max_lines)]
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for ln in lines:
            rec = {
                "question": f"Summarize: {ln}",
                "answer": ln,
                "source": {"title": "Wikipedia (AI Topics)"},
                "source_file": str(src),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(lines)} lines → {OUT_JSONL}")
    return OUT_JSONL


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build wikipedia_corpus.jsonl from local AI topics text")
    ap.add_argument("--max-lines", type=int, default=0, help="Cap number of lines (0=unlimited)")
    args = ap.parse_args()
    build(max_lines=int(args.max_lines))


if __name__ == "__main__":  # pragma: no cover
    main()

