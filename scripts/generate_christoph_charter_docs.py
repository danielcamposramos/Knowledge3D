#!/usr/bin/env python3
"""Generate one-by-one docs artifacts for Christoph's Substack mission corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MISSION_KEYWORDS = [
    "sovereign",
    "sovereignty",
    "boundary",
    "boundaries",
    "charter",
    "autonomous",
    "intelligent",
    "llm",
    "ai",
    "privacy",
    "transparency",
    "responsible",
    "ethic",
    "moral",
    "control",
    "governance",
    "systems",
]


def _load_index(index_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out


def _keyword_counts(text: str, keywords: list[str]) -> dict[str, int]:
    lowered = text.lower()
    counts: dict[str, int] = {}
    for kw in keywords:
        c = lowered.count(kw)
        if c > 0:
            counts[kw] = c
    return counts


def _extractive_summary(text: str, keywords: list[str]) -> tuple[str, list[str]]:
    sents = _sentences(text)
    if not sents:
        return "", []

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sents):
        low = sent.lower()
        keyword_hits = sum(low.count(kw) for kw in keywords)
        length_bonus = min(len(sent) / 220.0, 1.0)
        first_bonus = 0.5 if idx < 4 else 0.0
        score = keyword_hits * 2.0 + length_bonus + first_bonus
        scored.append((score, idx, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = sorted(scored[:5], key=lambda x: x[1])
    key_sentences = [c[2] for c in chosen]
    summary = " ".join(key_sentences[:3]).strip()

    # Fallback to opening sentences if low-signal text.
    if not summary:
        summary = " ".join(sents[:3]).strip()
        key_sentences = sents[:5]
    return summary, key_sentences


def _safe_slug(url: str, fallback: str) -> str:
    slug = url.rstrip("/").split("/")[-1].lower().strip()
    slug = re.sub(r"[^a-z0-9\-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or fallback


def _render_markdown(
    *,
    idx: int,
    title: str,
    url: str,
    fetched_at: str,
    summary: str,
    key_sentences: list[str],
    keyword_counts: dict[str, int],
    text_path: Path,
) -> str:
    keywords_repr = ", ".join(f"{k}:{v}" for k, v in sorted(keyword_counts.items()))
    if not keywords_repr:
        keywords_repr = "none"
    lines = [
        f"# Post {idx:04d}: {title}",
        "",
        f"- URL: {url}",
        f"- Fetched At (UTC): {fetched_at}",
        f"- Keyword Signals: {keywords_repr}",
        f"- Full Text File: `{text_path}`",
        "",
        "## Summary",
        summary or "(empty)",
        "",
        "## Key Sentences",
    ]
    if key_sentences:
        for sent in key_sentences:
            lines.append(f"- {sent}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-jsonl",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/scrapes/hblazer_substack_full/index.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/Sovereign_Systems_Charter"),
    )
    args = parser.parse_args()

    rows = _load_index(args.index_jsonl)
    output_root = args.output_root
    posts_root = output_root / "posts_full"
    posts_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip() or f"Post {i:04d}"
        fetched_at = str(row.get("fetched_at", "")).strip()
        txt_path = Path(str(row.get("txt_path", "")).strip())
        if txt_path.exists():
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raw_text = ""
        cleaned = _clean_text(raw_text)
        summary, key_sentences = _extractive_summary(cleaned, MISSION_KEYWORDS)
        kw_counts = _keyword_counts(cleaned, MISSION_KEYWORDS)

        slug = _safe_slug(url, f"post-{i:04d}")
        base = f"{i:04d}_{slug}"
        out_txt = posts_root / f"{base}.txt"
        out_md = posts_root / f"{base}.md"

        out_txt.write_text(cleaned, encoding="utf-8")
        out_md.write_text(
            _render_markdown(
                idx=i,
                title=title,
                url=url,
                fetched_at=fetched_at,
                summary=summary,
                key_sentences=key_sentences,
                keyword_counts=kw_counts,
                text_path=out_txt,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "index": i,
                "title": title,
                "url": url,
                "fetched_at": fetched_at,
                "txt_file": str(out_txt),
                "summary_file": str(out_md),
                "text_len": len(cleaned),
                "keyword_counts": kw_counts,
            }
        )

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Christoph Sovereign Systems Charter Mission Corpus",
                "",
                "Generated from the completed `hblazer_substack_full` scrape.",
                "",
                "## Contents",
                "- `posts_full/*.txt`: normalized one-by-one text corpus.",
                "- `posts_full/*.md`: per-post metadata + extractive summary + key sentences.",
                "- `manifest.json`: machine-readable index for downstream synthesis.",
                "",
                "## Generation Scope",
                f"- Source index: `{args.index_jsonl}`",
                f"- Total posts generated: `{len(manifest)}`",
                "",
                "## Note",
                "- This stage intentionally excludes the final merged single report.",
                "- Final consolidated synthesis should be generated in a separate step.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_index_jsonl": str(args.index_jsonl),
                "total_posts": len(manifest),
                "posts": manifest,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    print(f"[done] generated_posts={len(manifest)} output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
