#!/usr/bin/env python3
"""
Sanity check for the navigation specialist checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch

from knowledge3d.training.math_benchmarks.navigation_dataset import NavigationDataset
from knowledge3d.training.math_benchmarks.navigation_model import (
    NavigationSeqModel,
    BOS_ID,
    PAD_ID,
    RULE_OFFSET,
)


def _decode_rules(rule_ids: List[int], registry: List[str]) -> List[str]:
    decoded = []
    for rule_id in rule_ids:
        if 0 <= rule_id < len(registry):
            decoded.append(registry[rule_id])
        else:
            decoded.append(f"unknown_{rule_id}")
    return decoded


def _greedy_decode(
    model: NavigationSeqModel,
    embedding: torch.Tensor,
    max_len: int,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    model.eval()
    tokens = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
    tokens[0, 0] = BOS_ID
    embedding = embedding.unsqueeze(0).to(device)
    for step in range(1, max_len):
        logits = model(embedding, tokens[:, :step])
        next_id = int(torch.argmax(logits[0, -1]).item())
        tokens[0, step] = next_id
    return tokens[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate navigation specialist checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/navigation_specialist_v1.pt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--bin",
        type=str,
        default="/K3D/Knowledge3D.local/logs/log_galaxy_microbench.bin",
        help="Log Galaxy binary file.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="/K3D/Knowledge3D.local/logs/log_galaxy_microbench.json",
        help="Log Galaxy metadata JSON.",
    )
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument("--index", type=int, default=None, help="Fixed index to evaluate.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    embedding_dim = int(checkpoint["embedding_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    vocab_size = int(checkpoint["vocab_size"])
    registry = checkpoint["rule_registry"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NavigationSeqModel(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    dataset = NavigationDataset(bin_path=args.bin, meta_path=args.meta)
    try:
        total = len(dataset)
        if total == 0:
            raise SystemExit("NavigationDataset is empty.")
        if args.index is not None:
            idx = args.index
        else:
            rng = random.Random(args.seed)
            idx = rng.randrange(0, total)

        embed, rule_ids = dataset[idx]
        target_ids = [int(v) for v in rule_ids.tolist()]
        decoded_target = _decode_rules(target_ids, registry)

        max_len = max(1, len(target_ids) + 1)
        predicted_tokens = _greedy_decode(model, embed, max_len=max_len, device=device, vocab_size=vocab_size)
        predicted_ids = [
            int(tok) - RULE_OFFSET
            for tok in predicted_tokens.tolist()
            if int(tok) >= RULE_OFFSET
        ]
        decoded_pred = _decode_rules(predicted_ids, registry)

        print(f"[Eval] index={idx}")
        print(f"Problem embedding dim: {embed.numel()}")
        print(f"Predicted IDs: {predicted_ids}")
        print(f"Predicted Path: {decoded_pred}")
        print(f"Actual IDs: {target_ids}")
        print(f"Actual Path: {decoded_target}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
