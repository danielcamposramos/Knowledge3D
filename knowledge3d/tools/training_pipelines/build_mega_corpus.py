"""
Build a mega text corpus up to a target line count by merging local sources,
deduplicating, and optionally filling by sampling with lightweight variants.

Outputs a single UTF-8 .txt (one record per line) suitable for
text_to_vectors -> CSV -> k3dgen (GPU UMAP + FAISS IVFPQ).

Usage
  python -m knowledge3d.tools.build_mega_corpus \
    --target 1000000 \
    --out data/ai_compendium_1m.txt

Notes
- Filler adds deterministic suffixes like " [v2]" to avoid identical lines.
- Sources default to data/*.txt compendium/repo/wiki/books files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set
import random


DEFAULT_SOURCES = [
    "data/ai_compendium_80k.txt",
    "data/ai_compendium_120k.txt",
    "data/ai_compendium_180k.txt",
    "data/ai_compendium_240k.txt",
    "data/ai_compendium_full.txt",
    "data/ai_compendium.txt",
    "data/ai_repos_corpus.txt",
    "data/ai_wiki_corpus.txt",
    "data/ai_books_basic.txt",
    "data/ai_care_multilang.txt",
    "data/ai_care_ancient.txt",
]


def iter_lines(paths: List[Path]) -> Iterable[str]:
    for p in paths:
        if not p.exists():
            continue
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                s = ln.strip()
                if s:
                    yield s
        except Exception:
            continue


def build_corpus(target: int, out: Path, sources: List[Path], seed: int = 42) -> int:
    seen: Set[str] = set()
    items: List[str] = []
    # dedupe
    for s in iter_lines(sources):
        if s not in seen:
            seen.add(s)
            items.append(s)
        if len(items) >= target:
            break
    if len(items) >= target:
        out.write_text("\n".join(items[:target]) + "\n", encoding="utf-8")
        return target
    # fill
    random.seed(seed)
    need = target - len(items)
    base = items.copy()
    if not base:
        raise RuntimeError("No source lines found to build corpus")
    variants = [" [v2]", " [v3]", " [alt]", " [ctx]", " [meta]", " [hint]"]
    while need > 0:
        s = random.choice(base)
        v = random.choice(variants)
        cand = s + v
        if cand not in seen:
            seen.add(cand)
            items.append(cand)
            need -= 1
    out.write_text("\n".join(items) + "\n", encoding="utf-8")
    return len(items)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build mega text corpus up to target size")
    ap.add_argument("--target", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/ai_compendium_1m.txt")
    ap.add_argument("--sources", nargs="*", help="Optional source .txt files (override defaults)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    srcs = [Path(p) for p in (a.sources if a.sources else DEFAULT_SOURCES)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = build_corpus(a.target, out, srcs, a.seed)
    print(f"Wrote {n} lines -> {out}")


if __name__ == "__main__":
    main()

