#!/usr/bin/env python3
"""
Train a lightweight binary router to classify calculus vs general problems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


def _load_dataset(path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = entry.get("text", "").strip()
            label = int(entry.get("label", 0))
            if text:
                samples.append((text, label))
    return samples


def _embed_samples(samples: List[Tuple[str, int]], dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings: List[List[float]] = []
    labels: List[int] = []
    for text, label in samples:
        embeddings.append(embed_text(text, dim=dim))
        labels.append(label)
    return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    correct = (preds == labels).float().mean().item()
    return float(correct)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calculus vs GSM8K router.")
    parser.add_argument("--input", required=True, help="Path to router_train.jsonl.")
    parser.add_argument("--output", required=True, help="Output model path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--embedding-dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    samples = _load_dataset(args.input)
    if not samples:
        raise RuntimeError(f"No samples loaded from {args.input}")

    embeddings, labels = _embed_samples(samples, dim=int(args.embedding_dim))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)
    labels = labels.to(device)

    model = nn.Sequential(
        nn.Linear(int(args.embedding_dim), int(args.hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_dim), 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            acc = _accuracy(logits.detach(), labels)
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} acc={acc:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "labels": {"calculus": 1, "gsm8k": 0},
        },
        output_path,
    )
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
