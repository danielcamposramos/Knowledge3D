#!/usr/bin/env python3
"""
Validate prototype-delta compression fidelity at scale.

Example:
PYTHONPATH=. python3 scripts/validate_prototype_delta.py \
    --prototype-table validation_cache/prototype_table_2048d_512.npz \
    --tokens-file data/ai_compendium.txt \
    --sample-size 2000 \
    --embedding-dim 2048 \
    --topk 16 \
    --output validation_results/prototype_delta_compression.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from knowledge3d.cranium.fidelity_validator import ProceduralFidelityValidator
from knowledge3d.cranium.procedural_compiler import ProceduralCompiler, PrototypeTable
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine


def _load_tokens(paths: Iterable[Path], limit: int | None = None) -> List[str]:
    tokens: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                for token in line.strip().split():
                    if not token:
                        continue
                    tokens.append(token)
                    if limit and len(tokens) >= limit:
                        return tokens
    return tokens


def _write_markdown(path: Path, summary: dict, metrics: dict) -> None:
    lines = [
        "# Prototype-Delta Compression Validation",
        "",
        f"**Tokens evaluated:** {metrics['tokens_evaluated']}",
        f"**Embedding dim:** {metrics['embedding_dim']}",
        f"**Target dim:** {metrics['target_dimension']}",
        f"**Prototype table:** `{metrics['prototype_table']}`",
        f"**Top-K range:** min {metrics['min_topk']} / max {metrics['max_topk']}",
        "",
        "## Aggregate Metrics",
        f"- Average compression ratio: **{summary['average_compression']:.2f}:1**",
        f"- Min / Max compression ratio: {metrics['compression_ratio_min']:.2f} / {metrics['compression_ratio_max']:.2f}",
        f"- Average cosine similarity: **{summary['average_similarity']:.5f}**",
        f"- Valid samples (≥ threshold): {summary['valid_ratio'] * 100:.2f}%",
        "",
        "## Additional Statistics",
        f"- Average nnz (top-k corrections): {metrics['average_topk']:.1f}",
        f"- Prototype distance (avg/max): {metrics['proto_distance_avg']:.4f} / {metrics['proto_distance_max']:.4f}",
        "",
        "## JSON Metrics",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prototype-delta compression fidelity.")
    parser.add_argument("--prototype-table", type=Path, required=True)
    parser.add_argument("--tokens-file", action="append", default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--target-dim", type=int, default=None, help="Optional truncation dimension (Matryoshka prefix).")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--topk-step", type=int, default=8)
    parser.add_argument("--topk-cap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--codec", choices=["dense", "sparse", "simple", "multi", "dict"], default="dense")
    parser.add_argument("--max-prototypes", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--residual-topk", type=int, default=256)
    parser.add_argument("--dictionary-path", type=Path, default=None)
    parser.add_argument("--dict-max-coeffs", type=int, default=8)
    parser.add_argument("--dict-residual-topk", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("validation_results/prototype_delta_compression.md"))
    parser.add_argument("--json-output", type=Path, default=Path("validation_results/prototype_delta_compression.json"))
    parser.add_argument("--disable-basis", action="store_true", help="Force sparse delta encoding even if basis is available.")
    args = parser.parse_args()

    token_paths = [Path(p) for p in args.tokens_file] if args.tokens_file else [Path("data/ai_compendium.txt")]
    for path in token_paths:
        if not path.exists():
            raise FileNotFoundError(f"Token source not found: {path}")
    if not args.prototype_table.exists():
        raise FileNotFoundError(f"Prototype table not found: {args.prototype_table}")

    tokens = _load_tokens(token_paths, limit=args.sample_size)
    target_dim = args.target_dim or args.embedding_dim
    engine_dim = max(args.embedding_dim, target_dim)
    engine = RPNEmbeddingEngine(embedding_dim=engine_dim)

    if target_dim != engine_dim:
        class _TruncateEngine:
            def __init__(self, base_engine, dim):
                self._base = base_engine
                self._dim = dim

            def embed_word(self, text: str) -> np.ndarray:
                return self._base.embed_word(text)[: self._dim]

        embedding_engine = _TruncateEngine(engine, target_dim)
    else:
        embedding_engine = engine
    table = PrototypeTable.load(args.prototype_table)
    dictionary_atoms = None
    if args.dictionary_path:
        if not args.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {args.dictionary_path}")
        payload = np.load(args.dictionary_path, allow_pickle=False)
        dictionary_atoms = payload["atoms"].astype(np.float32)
    elif args.codec == "dict":
        raise RuntimeError("Dictionary codec requested but --dictionary-path not provided.")

    compiler = ProceduralCompiler(
        prototype_table=table,
        prototype_topk=args.topk,
        prototype_topk_step=args.topk_step,
        prototype_topk_cap=args.topk_cap,
        prototype_cosine_threshold=args.threshold,
        use_prototype_basis=(not args.disable_basis) and args.codec == "sparse",
        multi_max_prototypes=args.max_prototypes,
        multi_candidate_count=args.candidate_count,
        multi_residual_topk=args.residual_topk,
        multi_similarity_threshold=args.threshold,
        dictionary_atoms=dictionary_atoms,
        dictionary_max_coeffs=args.dict_max_coeffs,
        dictionary_residual_topk=args.dict_residual_topk,
        dictionary_similarity_threshold=args.threshold,
    )
    validator = ProceduralFidelityValidator(
        rpn_engine=embedding_engine,
        compiler=compiler,
        similarity_threshold=args.threshold,
    )

    if args.codec == "dense":
        mode = "prototype_dense"
    elif args.codec == "sparse":
        mode = "prototype_sparse"
    elif args.codec == "multi":
        mode = "prototype_multi"
    elif args.codec == "dict":
        mode = "dictionary_sparse"
    else:
        mode = "simple"

    results = validator.batch_validate(tokens, mode=mode)
    summary = validator.summarize(results)

    ratios = [item.compression_ratio for item in results]
    topks = [
        item.extra.get("nnz", item.extra.get("sparse_count", 0)) if item.extra else 0
        for item in results
    ]
    proto_distances = [item.extra.get("proto_distance", 0.0) if item.extra else 0.0 for item in results]

    metrics = {
        **summary,
        "tokens_evaluated": len(results),
        "embedding_dim": args.embedding_dim,
        "target_dimension": target_dim,
        "prototype_table": str(args.prototype_table),
        "compression_ratio_min": float(min(ratios)) if ratios else 0.0,
        "compression_ratio_max": float(max(ratios)) if ratios else 0.0,
        "average_topk": float(np.mean(topks)) if topks else 0.0,
        "min_topk": int(min(topks)) if topks else 0,
        "max_topk": int(max(topks)) if topks else 0,
        "proto_distance_avg": float(np.mean(proto_distances)) if proto_distances else 0.0,
        "proto_distance_max": float(np.max(proto_distances)) if proto_distances else 0.0,
        "threshold": args.threshold,
        "topk": args.topk,
        "topk_step": args.topk_step,
        "topk_cap": args.topk_cap,
        "tokens_file": [str(p) for p in token_paths],
        "use_basis": (not args.disable_basis) and args.codec == "sparse",
        "codec": args.codec,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_markdown(args.output, summary, metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
