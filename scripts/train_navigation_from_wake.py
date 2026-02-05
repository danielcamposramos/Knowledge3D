#!/usr/bin/env python3
"""
Train a navigation specialist from wake-curated datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import io

import torch
from torch import nn
from torch.utils.data import DataLoader

from knowledge3d.training.math_benchmarks.navigation_dataset import NavigationDataset
from knowledge3d.training.math_benchmarks.navigation_model import (
    NavigationSeqModel,
    PAD_ID,
    BOS_ID,
    RULE_OFFSET,
)
from knowledge3d.training.math_benchmarks.skill_galaxy import SkillGalaxy


def _load_samples(dataset: NavigationDataset) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(dataset)):
        embed, rule_ids = dataset[idx]
        samples.append((embed, rule_ids))
    return samples


def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device):
    embeddings = torch.stack([item[0] for item in batch]).to(device)
    seqs = [item[1] for item in batch]
    max_len = max(len(seq) for seq in seqs)

    input_tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long, device=device)
    target_tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long, device=device)

    for i, seq in enumerate(seqs):
        if len(seq) == 0:
            continue
        shifted = seq.to(device) + RULE_OFFSET
        target_tokens[i, : len(seq)] = shifted
        input_tokens[i, 0] = BOS_ID
        if len(seq) > 1:
            input_tokens[i, 1 : len(seq)] = shifted[:-1]

    return embeddings, input_tokens, target_tokens


def _compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != PAD_ID
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds == targets) & mask
    return float(correct.sum().item()) / float(mask.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train navigation specialist from wake dataset.")
    parser.add_argument("--bin", required=True, help="Path to Log Galaxy binary file.")
    parser.add_argument("--meta", required=True, help="Path to Log Galaxy metadata JSON.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--skill-galaxy-out", default=None, help="Optional Skill Galaxy JSONL path.")
    parser.add_argument("--skill-id", default="nav_calculus_v5", help="Skill ID to store.")
    parser.add_argument(
        "--skill-description",
        default="Sleep-curated navigation specialist.",
        help="Description for the skill embedding.",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    registry = meta.get("rule_registry", [])
    embedding_dim = int(meta["counts"]["embedding_dim"])
    vocab_size = len(registry) + RULE_OFFSET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NavigationDataset(bin_path=args.bin, meta_path=args.meta)
    try:
        samples = _load_samples(dataset)
    finally:
        dataset.close()

    model = NavigationSeqModel(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    loader = DataLoader(
        samples,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, device),
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for embeddings, token_inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings, token_inputs)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_acc += _compute_accuracy(logits, targets)
        avg_loss = total_loss / max(1, len(loader))
        avg_acc = total_acc / max(1, len(loader))
        if epoch == 1 or epoch % 25 == 0 or epoch == int(args.epochs):
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} acc={avg_acc:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "embedding_dim": embedding_dim,
        "hidden_dim": int(args.hidden_dim),
        "vocab_size": vocab_size,
        "rule_registry": registry,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "rule_offset": RULE_OFFSET,
    }
    torch.save(checkpoint, output_path)
    print(f"[Saved] {output_path}")

    if args.skill_galaxy_out:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        skill_galaxy = SkillGalaxy()
        skill_galaxy.add_skill(
            skill_id=args.skill_id,
            description=args.skill_description,
            payload=buffer.getvalue(),
            geometry="crystal",
            metadata={
                "embedding_dim": embedding_dim,
                "hidden_dim": int(args.hidden_dim),
                "vocab_size": vocab_size,
                "rule_registry": registry,
            },
        )
        skill_galaxy.to_jsonl(args.skill_galaxy_out)
        print(f"[SkillGalaxy] {args.skill_galaxy_out}")


if __name__ == "__main__":
    main()
