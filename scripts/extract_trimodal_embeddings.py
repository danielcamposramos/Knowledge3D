#!/usr/bin/env python3
"""
Phase G.2 - Extract Multi-Modal Embeddings

Consumes the combined dataset prepared by `prepare_trimodal_dataset.py`,
derives modality-specific embeddings, and stores curated views for specialist
training (OCR, speech, multi-modal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.training.multimodal.trimodal_dataset import (
    compute_embeddings,
    save_embeddings,
)


DEFAULT_DATASET = Path("/K3D/Knowledge3D.local/datasets/trimodal_phase_g.jsonl")
DEFAULT_OUTPUT_ALL = Path("/K3D/Knowledge3D.local/datasets/trimodal_embeddings.jsonl")
DEFAULT_CHARACTER = Path("/K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl")
DEFAULT_SPEECH = Path("/K3D/Knowledge3D.local/datasets/speech_embeddings.jsonl")
DEFAULT_MULTIMODAL = Path("/K3D/Knowledge3D.local/datasets/multimodal_embeddings.jsonl")


def _save_filtered(
    embeddings: Iterable[Dict[str, object]],
    output_path: Path,
    required_modalities: Sequence[str],
    fields: Sequence[str],
) -> int:
    """
    Save a filtered view of embeddings requiring specific modalities.

    Args:
        embeddings: Iterable of embedding dictionaries.
        output_path: Destination JSONL path.
        required_modalities: Modalities that must be present.
        fields: Fields to retain in the output records.

    Returns:
        Number of records written.
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in embeddings:
            modalities = set(record.get("modalities", []))
            if not all(mod in modalities for mod in required_modalities):
                continue

            payload = {key: record.get(key) for key in fields if key in record}
            payload["id"] = record.get("id")
            payload["modalities"] = list(modalities)

            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tri-modal embeddings for Phase G specialists."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Combined tri-modal dataset (JSONL)",
    )
    parser.add_argument(
        "--output-all",
        type=Path,
        default=DEFAULT_OUTPUT_ALL,
        help="Path to save all computed embeddings (JSONL)",
    )
    parser.add_argument(
        "--character-output",
        type=Path,
        default=DEFAULT_CHARACTER,
        help="Output JSONL for OCR specialist training",
    )
    parser.add_argument(
        "--speech-output",
        type=Path,
        default=DEFAULT_SPEECH,
        help="Output JSONL for speech specialist training",
    )
    parser.add_argument(
        "--multimodal-output",
        type=Path,
        default=DEFAULT_MULTIMODAL,
        help="Output JSONL for full multi-modal specialist training",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimensionality for each modality",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    print(f"[Load] Dataset: {args.dataset}")
    embeddings = compute_embeddings(
        dataset_path=args.dataset,
        embedding_dim=args.embedding_dim,
        limit=args.limit,
    )
    print(f"[Compute] Derived embeddings for {len(embeddings)} samples")

    save_embeddings(embeddings, args.output_all)
    print(f"[Write] Saved full embedding set to {args.output_all}")

    # Filtered specialist datasets
    character_count = _save_filtered(
        embeddings,
        args.character_output,
        required_modalities=("text", "image"),
        fields=("text_embedding", "image_embedding", "fused_embedding"),
    )
    print(f"[Character] Saved {character_count} samples -> {args.character_output}")

    speech_count = _save_filtered(
        embeddings,
        args.speech_output,
        required_modalities=("text", "audio"),
        fields=("text_embedding", "audio_embedding", "fused_embedding"),
    )
    print(f"[Speech] Saved {speech_count} samples -> {args.speech_output}")

    multimodal_count = _save_filtered(
        embeddings,
        args.multimodal_output,
        required_modalities=("text", "image", "audio"),
        fields=("text_embedding", "image_embedding", "audio_embedding", "fused_embedding"),
    )
    print(f"[Multimodal] Saved {multimodal_count} samples -> {args.multimodal_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
