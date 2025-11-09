#!/usr/bin/env python3
"""
Compress Phase G character embeddings with the adaptive dictionary codec.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor


def _format_ratio(value: float) -> str:
    return f"{value:.2f}"


def _load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    embeddings = data["embeddings"].astype(np.float32)
    char_ids = data["char_ids"].astype(np.int32)
    return embeddings, char_ids


def _char_label(char_id: int) -> str:
    try:
        return chr(int(char_id))
    except Exception:
        return f"#{char_id}"


def _write_markdown(path: Path, context: Dict, rows: List[str]) -> None:
    lines = [
        "# Character Embedding Compression Validation",
        "",
        f"**Dataset**: `{context['dataset']}`",
        f"**Embeddings evaluated**: {context['count']}",
        f"**Embedding dimension**: {context['dimension']}D",
        f"**Compression quality**: {context['quality']}",
        f"**Dictionary**: `{context['dictionary_path']}`",
        "",
        "## Aggregate Metrics",
        f"- Average compression ratio (vs. 2048D baseline): **{_format_ratio(context['average_compression'])}:1**",
        f"- Per-dimension compression (128D): {context['per_dim_compression']:.2f}:1",
        f"- Min / Max compression ratio: {context['compression_min']:.2f} / {context['compression_max']:.2f}",
        f"- Average cosine similarity: **{context['average_similarity']:.6f}**",
        f"- Min / Max similarity: {context['similarity_min']:.6f} / {context['similarity_max']:.6f}",
        f"- Valid samples (≥ {context['threshold']:.2f} threshold): {context['valid_ratio'] * 100:.2f}%",
        "",
        "## Codec Usage",
        f"- PD04 (dictionary): {context['codec_usage']['pd04'] * 100:.2f}% ({context['codec_counts']['pd04']} embeddings)",
        f"- PD02 (dense fallback): {context['codec_usage']['pd02'] * 100:.2f}% ({context['codec_counts']['pd02']} embeddings)",
        f"- Simple fallback: {context['codec_usage']['simple'] * 100:.2f}% ({context['codec_counts']['simple']} embeddings)",
        "",
        "## Comparison to Text Corpus",
        "",
        "| Dataset | Dimension | Compression | Fidelity | Notes |",
        "|---------|-----------|-------------|----------|-------|",
        "| ai_compendium.txt | 128D | 69.4:1 | 0.99998 | Text embeddings |",
        f"| Character glyphs | 128D | {_format_ratio(context['average_compression'])}:1 | {context['average_similarity']:.6f} | Visual embeddings |",
        "",
        "## Character Samples",
        "",
        "| Character | Embeddings | Avg Compression | Avg Fidelity | PD04 Usage |",
        "|-----------|------------|-----------------|--------------|-----------|",
    ]
    lines.extend(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress Phase G character embeddings.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/galaxy_character_embeddings.npz"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of embeddings for quick tests.")
    parser.add_argument("--quality", choices=["ultrafast", "fast", "balanced", "maximum"], default="fast")
    parser.add_argument("--output-md", type=Path, default=Path("validation_results/character_compression_128d.md"))
    parser.add_argument("--output-json", type=Path, default=Path("validation_results/character_compression_128d.json"))
    args = parser.parse_args()

    embeddings, char_ids = _load_embeddings(args.dataset)
    if args.limit:
        embeddings = embeddings[: args.limit]
        char_ids = char_ids[: args.limit]

    compressor = AdaptiveDimensionCompressor()
    programs: List[bytes] = []
    per_dim_ratios: List[float] = []
    overall_ratios: List[float] = []
    similarities: List[float] = []
    valid_flags: List[bool] = []
    codec_counts = {"pd04": 0, "pd02": 0, "simple": 0}
    char_stats: Dict[int, Dict[str, float]] = {}

    for idx, (embedding, char_id) in enumerate(zip(embeddings, char_ids)):
        program, metadata = compressor.compress(embedding, quality=args.quality, return_metadata=True)
        programs.append(program)
        per_ratio = metadata["actual_compression"]
        per_dim_ratios.append(per_ratio)
        scale = 2048 / metadata["target_dim"]
        overall_ratios.append(per_ratio * scale)
        similarities.append(metadata["actual_fidelity"])
        valid_flags.append(metadata["actual_fidelity"] >= metadata["threshold"])
        magic = program[:4]
        if magic == b"PD04":
            codec_label = "pd04"
        elif magic == b"PD02":
            codec_label = "pd02"
        elif magic == b"PD01":
            codec_label = "simple"
        else:
            codec_label = metadata.get("codec", "pd04")

        if codec_label == "pd04":
            codec_counts["pd04"] += 1
        elif codec_label == "pd02":
            codec_counts["pd02"] += 1
        else:
            codec_counts["simple"] += 1

        stats = char_stats.setdefault(int(char_id), {"count": 0, "compression_sum": 0.0, "overall_sum": 0.0, "fidelity_sum": 0.0, "pd04": 0})
        stats["count"] += 1
        stats["compression_sum"] += metadata["actual_compression"]
        stats["overall_sum"] += per_ratio * scale
        stats["fidelity_sum"] += metadata["actual_fidelity"]
        if codec_label == "pd04":
            stats["pd04"] += 1

        if (idx + 1) % 500 == 0:
            print(f"Compressed {idx + 1}/{len(embeddings)} embeddings...")

    count = len(embeddings)
    codec_usage = {k: v / count for k, v in codec_counts.items()}
    avg_comp = float(np.mean(overall_ratios)) if overall_ratios else 0.0
    per_dim_avg = float(np.mean(per_dim_ratios)) if per_dim_ratios else 0.0
    avg_sim = float(np.mean(similarities)) if similarities else 0.0
    context = {
        "dataset": str(args.dataset),
        "count": count,
        "dimension": compressor.dimension_map[args.quality],
        "quality": args.quality,
        "dictionary_path": str(compressor._locate_dictionary_file(compressor.dimension_map[args.quality])),
        "average_compression": avg_comp,
        "per_dim_compression": per_dim_avg,
        "compression_min": float(np.min(overall_ratios)) if overall_ratios else 0.0,
        "compression_max": float(np.max(overall_ratios)) if overall_ratios else 0.0,
        "average_similarity": avg_sim,
        "similarity_min": float(np.min(similarities)) if similarities else 0.0,
        "similarity_max": float(np.max(similarities)) if similarities else 0.0,
        "valid_ratio": float(np.mean(valid_flags)) if valid_flags else 0.0,
        "codec_usage": codec_usage,
        "codec_counts": codec_counts,
        "threshold": compressor.fidelity_thresholds[args.quality],
    }

    top_chars = sorted(char_stats.items(), key=lambda item: -item[1]["count"])[:10]
    rows = []
    for char_id, stats in top_chars:
        avg_c = stats["overall_sum"] / stats["count"]
        avg_f = stats["fidelity_sum"] / stats["count"]
        pd04_ratio = stats["pd04"] / stats["count"]
        rows.append(
            f"| {_char_label(char_id)} | {stats['count']} | {avg_c:.2f}:1 | {avg_f:.6f} | {pd04_ratio * 100:.1f}% |"
        )

    _write_markdown(args.output_md, context, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "character_embeddings",
                "path": str(args.dataset),
                "count": count,
                "dimension": compressor.dimension_map[args.quality],
                "quality": args.quality,
                "average_compression": context["average_compression"],
                "per_dimension_compression": context["per_dim_compression"],
                "compression_min": context["compression_min"],
                "compression_max": context["compression_max"],
                "average_similarity": context["average_similarity"],
                "similarity_min": context["similarity_min"],
                "similarity_max": context["similarity_max"],
                "valid_ratio": context["valid_ratio"],
                "codec_usage": codec_usage,
                "threshold": context["threshold"],
                "dictionary_path": context["dictionary_path"],
            },
            handle,
            indent=2,
        )

    print("Character compression complete.")
    print(f"Average compression: {context['average_compression']:.2f}:1")
    print(f"Average fidelity   : {context['average_similarity']:.6f}")


if __name__ == "__main__":
    main()
