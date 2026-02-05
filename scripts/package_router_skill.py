#!/usr/bin/env python3
"""
Package a router checkpoint into a SkillGalaxy JSONL entry.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn

from knowledge3d.training.math_benchmarks.router_embedder import embed_text
from knowledge3d.training.math_benchmarks.skill_galaxy import SkillGalaxy


def _build_router_model(embedding_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Router checkpoint must be a dict.")
    return ckpt


def _extract_state(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    state = ckpt.get("state_dict") or ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise ValueError("Router checkpoint missing state_dict.")
    embedding_dim = int(ckpt.get("embedding_dim", 384))
    hidden_dim = int(ckpt.get("hidden_dim", 128))
    return state, embedding_dim, hidden_dim


def _sanity_check(
    *,
    model: nn.Module,
    embedding_dim: int,
    calculus_text: str,
    gsm8k_text: str,
) -> None:
    model.eval()
    with torch.no_grad():
        for label, text in (("calculus", calculus_text), ("gsm8k", gsm8k_text)):
            emb = embed_text(text, dim=embedding_dim)
            tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            logit = model(tensor).item()
            print(f"[Sanity] {label} logit={logit:.4f} text={text[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package router checkpoint into SkillGalaxy.")
    parser.add_argument("--input", required=True, help="Router checkpoint (.pt).")
    parser.add_argument("--output", required=True, help="SkillGalaxy JSONL output.")
    parser.add_argument("--skill-id", required=True, help="Skill ID for the router.")
    parser.add_argument("--description", required=True, help="Skill description.")
    parser.add_argument("--geometry", default="crystal", help="Geometry hint.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Skill embedding dim.")
    parser.add_argument("--validate", action="store_true", help="Run a sanity check before export.")
    args = parser.parse_args()

    ckpt = _load_checkpoint(args.input)
    state, embedding_dim, hidden_dim = _extract_state(ckpt)

    model = _build_router_model(embedding_dim, hidden_dim)
    model.load_state_dict(state)

    if args.validate:
        _sanity_check(
            model=model,
            embedding_dim=embedding_dim,
            calculus_text="Find the derivative of x^2 at x=3.",
            gsm8k_text="If you have 3 apples and buy 2 more, how many apples?",
        )

    payload_buffer = io.BytesIO()
    torch.save(ckpt, payload_buffer)

    galaxy = SkillGalaxy(embedding_dim=int(args.embedding_dim))
    galaxy.add_skill(
        skill_id=args.skill_id,
        description=args.description,
        payload=payload_buffer.getvalue(),
        geometry=args.geometry,
        payload_format="torch_checkpoint",
        metadata={
            "role": "router_gatekeeper",
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "labels": ckpt.get("labels", {"calculus": 1, "gsm8k": 0}),
        },
    )
    galaxy.to_jsonl(args.output)
    print(f"[RouterSkill] Wrote {args.output}")


if __name__ == "__main__":
    main()
