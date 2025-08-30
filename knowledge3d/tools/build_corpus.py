"""
Merge multiple text sources into a deduplicated corpus with a target number of lines.

Sources (if present):
- data/ai_compendium_full.txt
- data/ai_repos_corpus.txt
- data/ai_wiki_corpus.txt
- data/ai_compendium.txt

Usage
  python3 -m knowledge3d.tools.build_corpus --target 120000 --out data/ai_compendium_120k.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build merged deduped corpus to target lines")
    ap.add_argument("--target", type=int, required=True, help="Target number of lines")
    ap.add_argument("--out", required=True, help="Output text file")
    args = ap.parse_args()
    sources = [
        Path("data/ai_compendium_full.txt"),
        Path("data/ai_repos_corpus.txt"),
        Path("data/ai_wiki_corpus.txt"),
        Path("data/ai_compendium.txt"),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in sources:
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= args.target:
                break
        if len(out) >= args.target:
            break
    # Top-up by cycling if below target
    if out and len(out) < args.target:
        i = 0
        base = list(out)
        while len(out) < args.target:
            out.append(base[i % len(base)])
            i += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out[: args.target]) + "\n", encoding="utf-8")
    print(f"Wrote {min(len(out), args.target)} -> {args.out}")


if __name__ == "__main__":
    main()

