#!/usr/bin/env python3
"""
Analyze and build prototype tables from RPN embeddings.

Usage example:
    python3 scripts/analyze_prototypes.py \
        --tokens-file data/ai_books_basic.sample.txt \
        --embedding-dim 2048 \
        --num-prototypes 512 \
        --max-tokens 50000 \
        --max-embeddings 40000 \
        --output validation_results/prototype_analysis.md \
        --json-output validation_results/prototype_analysis.json \
        --table-path validation_cache/prototype_table_2048d_512.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from knowledge3d.cranium.procedural_compiler import PrototypeTable
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def _load_tokens(paths: Iterable[Path], limit: int | None = None) -> List[str]:
    tokens: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                for tok in line.strip().split():
                    if tok:
                        tokens.append(tok)
                        if limit and len(tokens) >= limit:
                            return tokens
    return tokens


def _build_embeddings(tokens: List[str], embedding_dim: int, max_embeddings: int | None = None) -> np.ndarray:
    engine = RPNEmbeddingEngine(embedding_dim=embedding_dim)
    for token in tokens:
        engine.embed_word(token)
    table = engine.get_embedding_table()
    if max_embeddings and table.shape[0] > max_embeddings:
        table = table[:max_embeddings]
    return table.astype(np.float32)


def _write_markdown(path: Path, metrics: dict, table_path: Path) -> None:
    lines = [
        "# Prototype Table Analysis",
        "",
        f"**Source tokens:** {metrics['tokens_processed']}",
        f"**Embedding dim:** {metrics['embedding_dim']}",
        f"**Embeddings sampled:** {metrics['num_embeddings']}",
        f"**Prototypes:** {metrics['num_prototypes']}",
        f"**Average distance:** {metrics['avg_distance']:.4f}",
        f"**Max distance:** {metrics['max_distance']:.4f}",
        f"**Basis rank:** {metrics.get('basis_rank', 0)}",
        "",
        "## Files",
        f"- Prototype table: `{table_path}`",
        f"- JSON metrics: `{path.with_suffix('.json')}`",
        "",
        "## Metadata",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prototype table from RPN embeddings.")
    parser.add_argument("--tokens-file", action="append", default=None, help="Path(s) to text files.")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--num-prototypes", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=50000, help="Limit number of tokens to ingest.")
    parser.add_argument("--max-embeddings", type=int, default=40000, help="Limit number of trigram embeddings.")
    parser.add_argument("--output", type=Path, default=Path("validation_results/prototype_analysis.md"))
    parser.add_argument("--json-output", type=Path, default=Path("validation_results/prototype_analysis.json"))
    parser.add_argument("--table-path", type=Path, default=Path("validation_cache/prototype_table_2048d_512.npz"))
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basis-rank", type=int, default=32)
    args = parser.parse_args()

    token_paths = [Path(p) for p in args.tokens_file] if args.tokens_file else [Path("data/ai_books_basic.sample.txt")]
    for path in token_paths:
        if not path.exists():
            raise FileNotFoundError(f"Token source not found: {path}")

    tokens = _load_tokens(token_paths, limit=args.max_tokens)
    embeddings = _build_embeddings(tokens, embedding_dim=args.embedding_dim, max_embeddings=args.max_embeddings)
    table, table_metrics = PrototypeTable.build_from_embeddings(
        embeddings,
        args.num_prototypes,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        seed=args.seed,
        basis_rank=args.basis_rank,
    )
    eval_metrics = table.evaluate_embeddings(embeddings)

    table_path = table.save(args.table_path)

    metrics = {
        **table_metrics,
        **{f"coverage_{k}": v for k, v in eval_metrics.items()},
        "tokens_processed": len(tokens),
        "source_files": [str(p) for p in token_paths],
        "table_path": str(table_path),
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_markdown(args.output, metrics, table_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
