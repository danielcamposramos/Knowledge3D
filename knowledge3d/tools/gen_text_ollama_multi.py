from __future__ import annotations

"""
Generate topic-coherent text files via multiple local Ollama models sequentially.
Each model is loaded for the request and set to keep_alive=0s for fast unload.

Usage
  python -m knowledge3d.tools.gen_text_ollama_multi \
    --ollama http://192.168.0.4:11434 \
    --models "exaone3.5:latest,exaone-deep:latest,granite3.3:8b,deepseek-r1:14b" \
    --topics "architecture,furniture,rooms,doors,windows,kitchen,bedroom,bathroom,living room,materials,textures" \
    --n 60 \
    --out-dir ../Knowledge3D.local/datasets
"""

import argparse
from pathlib import Path
from typing import List

from .gen_text_ollama import generate_lines


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate via multiple Ollama models")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--models", required=True, help="Comma-separated model list")
    ap.add_argument("--topics", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    topics = [t.strip() for t in str(args.topics).split(",") if t.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    combined: List[str] = []
    for m in [s.strip() for s in str(args.models).split(",") if s.strip()]:
        lines = generate_lines(args.ollama, m, topics, int(args.n), temperature=0.5, keep_alive="0s")
        p = out_dir / f"text_house_{m.replace(':','_').replace('/','_')}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        combined.extend(lines)
        print(f"wrote {len(lines)} lines -> {p}")
    all_p = out_dir / "text_house_all.txt"
    all_p.write_text("\n".join(combined) + "\n", encoding="utf-8")
    print(f"combined -> {all_p} ({len(combined)} lines)")


if __name__ == "__main__":
    main()

