#!/usr/bin/env python3
"""
Fine-tune navigation specialist using corrective RLWHF samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from knowledge3d.training.math_benchmarks.navigation_model import (
    NavigationSeqModel,
    PAD_ID,
    BOS_ID,
    RULE_OFFSET,
)


class CorrectiveDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        return item["embedding"], int(item["rule_id"])


def _collate(batch: List[Tuple[torch.Tensor, int]], device: torch.device):
    embeddings = torch.stack([item[0] for item in batch]).to(device)
    rule_ids = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
    token_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long, device=device)
    targets = rule_ids + RULE_OFFSET
    return embeddings, token_inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train navigation specialist with corrective samples.")
    parser.add_argument(
        "--base",
        type=str,
        default="checkpoints/navigation_specialist_v1.pt",
        help="Base navigation checkpoint.",
    )
    parser.add_argument(
        "--corrective",
        type=str,
        default="data/corrective_tuning_v1.pt",
        help="Corrective tuning dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/navigation_specialist_v3_rlwhf.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tune epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    base_ckpt = torch.load(args.base, map_location="cpu")
    if "model_state" in base_ckpt:
        model_state = base_ckpt["model_state"]
    else:
        model_state = base_ckpt

    embedding_dim = int(base_ckpt.get("embedding_dim") or 0)
    hidden_dim = int(base_ckpt.get("hidden_dim") or 0)
    vocab_size = int(base_ckpt.get("vocab_size") or 0)
    rule_registry = base_ckpt.get("rule_registry") or []

    if not embedding_dim or not hidden_dim or not vocab_size:
        raise SystemExit("Base checkpoint missing model dimensions.")

    corrective = torch.load(args.corrective, map_location="cpu")
    samples = corrective.get("samples") or []
    corrective_registry = corrective.get("rule_registry") or []
    if corrective_registry != rule_registry:
        raise SystemExit("Corrective dataset rule_registry does not match base checkpoint.")

    if not samples:
        raise SystemExit("No corrective samples available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NavigationSeqModel(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(model_state)

    dataset = CorrectiveDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        for embeddings, token_inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings, token_inputs)
            loss = loss_fn(logits[:, 0, :], targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(loader))
        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "rule_registry": rule_registry,
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "rule_offset": RULE_OFFSET,
        },
        output_path,
    )
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
