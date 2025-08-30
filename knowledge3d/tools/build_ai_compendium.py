"""
Build an AI Compendium (training-format) from local K3D repo docs.

Outputs
- data/ai_compendium.json  (entries: id,title,language,source,text,tags)
- data/ai_compendium.txt   (one line per fact/snippet)

Usage
  python3 -m knowledge3d.tools.build_ai_compendium --target-lines 4000

Notes
- Scans markdown and text files under docs/ and spec/. Skips binaries.
- Splits paragraphs into lines, normalizes whitespace, and prefixes with title.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "ai_compendium.json"
OUT_TXT = ROOT / "data" / "ai_compendium.txt"


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_md_lines(content: str) -> Iterable[str]:
    # Keep headings and sentences; drop code fences
    in_code = False
    for raw in content.split("\n"):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        ln = raw.strip()
        if not ln:
            continue
        # remove leading bullets/hash
        ln = re.sub(r"^[#>*\-\d\.\)\s]+", "", ln).strip()
        if not ln:
            continue
        yield ln


def gather_sources() -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for folder in [ROOT / "docs", ROOT / "spec"]:
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            # Skip known large/binary-like or irrelevant
            if any(s in p.name.lower() for s in [".png", ".jpg", ".pdf", ".mpga", ".docx", ".xlsx", ".pptx"]):
                continue
            title = p.stem.replace('_', ' ')
            files.append((title, p))
    return files


def build_entries(target_lines: int) -> Tuple[dict, List[str]]:
    files = gather_sources()
    entries = []
    lines: List[str] = []
    i = 0
    for title, p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if p.suffix.lower() == ".json":
            # include minimal JSON strings (e.g., specs) – keep short
            try:
                obj = json.loads(text)
                snippet = json.dumps(list(obj.keys())[:10]) if isinstance(obj, dict) else p.name
                entries.append({
                    "id": f"ai:{i}", "title": title, "language": "en",
                    "source": {"type": "k3d_repo", "origin": str(p.relative_to(ROOT))},
                    "text": snippet,
                    "tags": ["ai_compendium", "json"],
                })
                lines.append(f"{title} — {snippet}")
                i += 1
            except Exception:
                continue
            continue
        # markdown/txt
        for ln in iter_md_lines(text):
            n = normalize_whitespace(ln)
            if len(n) < 8:
                continue
            entries.append({
                "id": f"ai:{i}", "title": title, "language": "en",
                "source": {"type": "k3d_repo", "origin": str(p.relative_to(ROOT))},
                "text": n,
                "tags": ["ai_compendium"],
            })
            lines.append(f"{title} — {n}")
            i += 1
            if len(lines) >= target_lines:
                break
        if len(lines) >= target_lines:
            break
    meta = {
        "name": "AI Compendium",
        "version": 1,
        "schema": "ai-compendium.v1",
        "generator": "knowledge3d.tools.build_ai_compendium",
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "notes": "Compiled from local K3D repo docs/specs; normalized into training lines.",
    }
    return {"meta": meta, "entries": entries}, lines


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build AI Compendium dataset from local docs/specs")
    ap.add_argument("--target-lines", type=int, default=4000)
    args = ap.parse_args()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    dataset, lines = build_entries(args.target_lines)
    OUT_JSON.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset['entries'])} JSON entries -> {OUT_JSON}")
    print(f"Wrote {len(lines)} lines -> {OUT_TXT}")


if __name__ == "__main__":
    main()
