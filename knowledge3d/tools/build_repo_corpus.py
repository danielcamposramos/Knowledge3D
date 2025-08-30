"""
Extract a text corpus from cloned repos (../Knowledge3D.local/repos).

Outputs
- data/ai_repos_corpus.txt (one line per normalized paragraph/sentence)

Usage
  python3 -m knowledge3d.tools.build_repo_corpus --max-lines 15000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT.parent / f"{ROOT.name}.local" / "repos"
OUT = ROOT / "data" / "ai_repos_corpus.txt"


def normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def iter_lines(md: str) -> Iterable[str]:
    in_code = False
    for raw in md.split("\n"):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        ln = raw.strip()
        ln = re.sub(r"^[#>*\-\d\.\)\s]+", "", ln).strip()
        if len(ln) >= 8:
            yield ln


def build(max_lines: int) -> int:
    lines: List[str] = []
    for p in REPOS.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".rst", ".txt", ".json"}:
            continue
        if any(s in p.name.lower() for s in [".png", ".jpg", ".pdf", ".mpga", ".docx", ".xlsx", ".pptx"]):
            continue
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        title = p.stem.replace('_', ' ')
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(txt)
                snippet = json.dumps(list(obj.keys())[:10]) if isinstance(obj, dict) else p.name
                lines.append(f"{title} — {snippet}")
            except Exception:
                continue
            if len(lines) >= max_lines:
                break
            continue
        for ln in iter_lines(txt):
            lines.append(f"{title} — {normalize(ln)}")
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding='utf-8')
    return len(lines)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build corpus from cloned repos")
    ap.add_argument("--max-lines", type=int, default=15000)
    args = ap.parse_args()
    n = build(args.max_lines)
    print(f"Wrote {n} lines -> {OUT}")


if __name__ == "__main__":
    main()

