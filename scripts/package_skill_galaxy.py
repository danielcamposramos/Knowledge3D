#!/usr/bin/env python3
"""
Package a trained navigation checkpoint into a Skill Galaxy JSONL entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from knowledge3d.training.math_benchmarks.skill_galaxy import SkillGalaxy


def _resolve_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a navigation skill into Skill Galaxy.")
    parser.add_argument("--input", required=True, help="Path to navigation checkpoint (.pt).")
    parser.add_argument("--output", required=True, help="Path to output Skill Galaxy JSONL.")
    parser.add_argument("--skill-id", required=True, help="Skill ID for the Galaxy entry.")
    parser.add_argument("--description", required=True, help="Human-readable skill description.")
    parser.add_argument("--geometry", default="crystal", help="Geometry type (default: crystal).")
    parser.add_argument("--embedding-dim", type=int, default=0, help="Embedding dimension override.")
    parser.add_argument("--hidden-dim", type=int, default=0, help="Hidden dimension override.")
    parser.add_argument("--vocab-size", type=int, default=0, help="Vocab size override.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    checkpoint = torch.load(str(input_path), map_location="cpu")
    raw_payload = input_path.read_bytes()

    embedding_dim = _resolve_int(args.embedding_dim or checkpoint.get("embedding_dim"))
    hidden_dim = _resolve_int(args.hidden_dim or checkpoint.get("hidden_dim"))
    vocab_size = _resolve_int(args.vocab_size or checkpoint.get("vocab_size"))
    rule_registry = checkpoint.get("rule_registry") if isinstance(checkpoint, dict) else None

    if not embedding_dim or not hidden_dim or not vocab_size:
        raise ValueError("Missing model dimensions. Provide overrides or checkpoint metadata.")

    metadata: Dict[str, Any] = {
        "description": args.description,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
    }
    if rule_registry is not None:
        metadata["rule_registry"] = rule_registry

    skill_galaxy = SkillGalaxy(embedding_dim=embedding_dim)
    skill_galaxy.add_skill(
        skill_id=args.skill_id,
        description=args.description,
        payload=raw_payload,
        geometry=args.geometry,
        payload_format="torch_checkpoint",
        metadata=metadata,
    )
    skill_galaxy.to_jsonl(args.output)

    print(
        json.dumps(
            {
                "skill_id": args.skill_id,
                "output": args.output,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "vocab_size": vocab_size,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
