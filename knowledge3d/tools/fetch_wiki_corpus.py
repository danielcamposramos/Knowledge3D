"""
Fetch plain-text pages from Wikipedia for curated AI topics and emit a corpus.

Outputs
- data/ai_wiki_corpus.txt (one line per paragraph/sentence)

Usage
  python3 -m knowledge3d.tools.fetch_wiki_corpus --topics-file data/topics_ai.txt --min-len 80 --max-lines 3000
  python3 -m knowledge3d.tools.fetch_wiki_corpus --default-topics --max-lines 3000

Notes
- Uses REST API page/plain/{title} for plaintext content.
- Respects a simple length filter and de-duplicates lines.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Set

import requests
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
OUT_TXT = ROOT / "data" / "ai_wiki_corpus.txt"


DEFAULT_TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Transformer (machine learning)",
    "Large language model",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Knowledge graph",
    "Graph neural network",
    "Embeddings",
    "Word embedding",
    "Cosine similarity",
    "Dimensionality reduction",
    "UMAP",
    "Principal component analysis",
    "Information retrieval",
    "Vector database",
    "Retrieval-augmented generation",
    "Ontology (information science)",
    "Semantic Web",
    "glTF",
    "WebXR",
    "Augmented reality",
    "3D computer graphics",
    "Data visualization",
    "Explainable artificial intelligence",
    "AI safety",
    "AI alignment",
    "AI ethics",
    "Explainable artificial intelligence",
    "Foundation model",
    "Self-supervised learning",
    "Contrastive learning",
    "Diffusion model",
    "Generative adversarial network",
    "Prompt engineering",
    "Reinforcement learning from human feedback",
    "Knowledge representation and reasoning",
    "Ontology learning",
    "Vector search",
    "FAISS",
    "Hierarchical navigable small world",
    "Annoy (software)",
    "Milvus (software)",
    "Qdrant",
    "Pinecone (company)",
    "OpenGL",
    "WebGL",
    "WebGPU",
    "Three.js",
    "glTF",
    "JSON",
    "Graph database",
    "Neo4j",
    "Cypher (query language)",
    "SPARQL",
    "Resource Description Framework",
]


def fetch_plain(title: str) -> str:
    safe = urllib.parse.quote(title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/plain/{safe}"
    headers = {
        "accept": "text/plain",
        "user-agent": "Knowledge3D/1.0 (+https://github.com/danielcamposramos/Knowledge3D)"
    }
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 200 and r.text:
        return r.text
    # Fallback: action=query extracts
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
        "origin": "*",
    }
    rq = requests.get(api, params=params, headers=headers, timeout=20)
    if rq.status_code != 200:
        return ""
    data = rq.json()
    pages = data.get("query", {}).get("pages", {})
    for _, pg in pages.items():
        ext = pg.get("extract")
        if ext:
            return ext
    return ""


def iter_lines(text: str) -> Iterable[str]:
    # split on blank lines and periods to get concise facts
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for b in blocks:
        # Normalize whitespace
        b = re.sub(r"\s+", " ", b).strip()
        # Further split long blocks by sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", b)
        for p in parts:
            ln = p.strip()
            if ln:
                yield ln


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Fetch AI topics from Wikipedia into a training corpus")
    ap.add_argument("--topics-file", help="Path to a file with one topic per line")
    ap.add_argument("--default-topics", action="store_true", help="Use curated default AI topics")
    ap.add_argument("--min-len", type=int, default=80, help="Minimum line length")
    ap.add_argument("--max-lines", type=int, default=3000, help="Maximum lines to write")
    args = ap.parse_args()

    topics: List[str] = []
    if args.topics_file:
        topics = [ln.strip() for ln in Path(args.topics_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.default_topics or not topics:
        topics = DEFAULT_TOPICS

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    lines: List[str] = []
    for title in topics:
        txt = fetch_plain(title)
        if not txt:
            continue
        count_before = len(lines)
        for ln in iter_lines(txt):
            if len(ln) < args.min_len:
                continue
            # prefix with title for context
            pref = f"{title} — {ln}"
            if pref in seen:
                continue
            seen.add(pref)
            lines.append(pref)
            if len(lines) >= args.max_lines:
                break
        print(f"{title}: +{len(lines)-count_before} lines")
        if len(lines) >= args.max_lines:
            break

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines -> {OUT_TXT}")


if __name__ == "__main__":
    main()
