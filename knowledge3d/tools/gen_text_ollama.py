from __future__ import annotations

"""
Generate topic‑coherent text lines via a local Ollama endpoint and save to a
plain text file (one line per sample). Designed to seed balanced text corpora
for K3D with consistent subjects across modalities.

Usage
  python -m knowledge3d.tools.gen_text_ollama \
    --ollama http://192.168.0.4:11434 \
    --model exaone3.5:latest \
    --topics "animals,sports,vehicles,gardens,tools" \
    --n 80 \
    --out ../Knowledge3D.local/datasets/exaone_text_v1.txt
"""

import argparse
import json
from pathlib import Path
from typing import List

import requests  # type: ignore


def _prompt_for_topics(topics: List[str], n: int) -> str:
    per = max(1, n // max(1, len(topics)))
    tlist = ", ".join(topics)
    return (
        "You are generating a compact, diverse set of short, self-contained text lines "
        "for an embedding-based spatial memory. Keep each line under 180 characters.\n\n"
        f"Topics to cover uniformly: {tlist}.\n"
        f"Generate about {per} lines per topic.\n\n"
        "Each line must:\n"
        "- start with the topic tag in square brackets, e.g., [animals] Wolves howl to coordinate hunts.\n"
        "- be factually grounded or an explanatory snippet; avoid opinions, fluff, or unsafe content.\n"
        "- avoid colons after the tag.\n\n"
        "Output ONLY the lines, one per line, no numbering, no JSON."
    )


def generate_lines(ollama: str, model: str, topics: List[str], n: int, temperature: float = 0.6, keep_alive: str = "0s") -> List[str]:
    url = ollama.rstrip("/") + "/api/generate"
    data = {
        "model": model,
        "prompt": _prompt_for_topics(topics, n),
        "stream": False,
        "options": {"temperature": float(temperature)},
        "keep_alive": str(keep_alive),
    }
    r = requests.post(url, json=data, timeout=240)
    r.raise_for_status()
    out = r.json()
    txt = out.get("response") or out.get("data") or ""
    lines: List[str] = []
    for raw in txt.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Strip bullets/numbering
        s = s.lstrip("-*0123456789. ")
        # Keep only lines with a topic tag
        if not (s.startswith("[") and "]" in s and len(s) > 3):
            continue
        lines.append(s)
        if len(lines) >= n:
            break
    # Dedup minimally
    seen, out_lines = set(), []
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        out_lines.append(s)
    return out_lines


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate topic-coherent text via Ollama")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="exaone3.5:latest")
    ap.add_argument("--topics", default="animals,sports,vehicles,gardens,tools")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    topics = [t.strip() for t in str(args.topics).split(",") if t.strip()]
    lines = generate_lines(str(args.ollama), str(args.model), topics, int(args.n))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines -> {out}")


if __name__ == "__main__":
    main()
