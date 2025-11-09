#!/usr/bin/env python3
"""
Train a dictionary for sparse procedural encoding.

Usage:
PYTHONPATH=. python3 scripts/train_dictionary.py \
    --tokens-file data/ai_compendium.txt \
    --embedding-dim 2048 \
    --dimensions 64,128,512,2048 \
    --num-samples 60000 \
    --components 512 \
    --output-dir validation_cache \
    --report validation_results/dictionary_training.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import DictionaryLearning

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def _load_tokens(paths: List[Path], limit: int) -> List[str]:
    tokens: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                for token in line.strip().split():
                    if not token:
                        continue
                    tokens.append(token)
                    if len(tokens) >= limit:
                        return tokens
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dictionary for sparse procedural codec.")
    parser.add_argument("--tokens-file", action="append", default=None)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=60000)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--dimensions", type=str, default=None, help="Comma-separated list of target dimensions (default: embedding_dim).")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_cache"))
    parser.add_argument("--report", type=Path, default=Path("validation_results/dictionary_training.md"))
    args = parser.parse_args()

    token_paths = [Path(p) for p in args.tokens_file] if args.tokens_file else [Path("data/ai_compendium.txt")]
    for path in token_paths:
        if not path.exists():
            raise FileNotFoundError(f"Token source not found: {path}")

    dims = (
        [int(part) for part in args.dimensions.split(",")]
        if args.dimensions
        else [args.embedding_dim]
    )
    dims = sorted(set(dims))
    max_dim = max(dims + [args.embedding_dim])

    tokens = _load_tokens(token_paths, args.num_samples)
    engine = RPNEmbeddingEngine(embedding_dim=max_dim)
    base_embeddings = np.vstack([engine.embed_word(tok) for tok in tokens]).astype(np.float32)

    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dim in dims:
        data = base_embeddings[:, :dim]
        n_components = min(dim, args.components)
        learner = DictionaryLearning(
            n_components=n_components,
            alpha=args.alpha,
            max_iter=args.max_iter,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
        )
        learner.fit(data)
        atoms = learner.components_.astype(np.float32)
        norms = np.linalg.norm(atoms, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        atoms = atoms / norms

        metadata = {
            "tokens_used": len(tokens),
            "embedding_dim": dim,
            "components": n_components,
            "alpha": args.alpha,
            "max_iter": args.max_iter,
            "source_files": [str(p) for p in token_paths],
        }
        output_path = args.output_dir / f"dictionary_{dim}d_{n_components}.npz"
        np.savez_compressed(output_path, atoms=atoms, metadata=np.array([json.dumps(metadata)]))
        print(f"[train_dictionary] Saved dictionary for {dim}D → {output_path}")
        results.append({"dim": dim, "components": n_components, "path": output_path, "metadata": metadata})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Dictionary Training Results", ""]
    lines.append(f"Total tokens: {len(tokens)}")
    lines.append(f"Dimensions trained: {', '.join(str(r['dim']) for r in results)}")
    lines.append("")
    for res in results:
        lines.extend(
            [
                f"## Dimension {res['dim']} (atoms={res['components']})",
                "",
                f"- Output: `{res['path']}`",
                "",
                "```json",
                json.dumps(res["metadata"], indent=2),
                "```",
                "",
            ]
        )
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
