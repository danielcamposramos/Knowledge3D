#!/usr/bin/env python3
"""
Train router classifier from RouterGalaxy JSONL memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


def _load_entries(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _extract_samples(entries: List[Dict[str, Any]], *, target_dim: Optional[int] = None) -> Tuple[List[List[float]], List[int], int]:
    embeddings: List[List[float]] = []
    labels: List[int] = []
    for entry in entries:
        label = entry.get("label")
        if label not in (0, 1):
            continue
        embedding = entry.get("embedding")
        text = str(entry.get("problem_text", "")).strip()
        if not embedding:
            if not text:
                continue
            embedding = embed_text(text, dim=int(target_dim) if target_dim else 256)
        embedding = [float(v) for v in embedding]
        embeddings.append(embedding)
        labels.append(int(label))
    if not embeddings:
        return embeddings, labels, int(target_dim or 0)
    max_dim = max(len(v) for v in embeddings)
    dim = int(target_dim or max_dim)
    normalized: List[List[float]] = []
    for vec in embeddings:
        if len(vec) >= dim:
            normalized.append(vec[:dim])
        else:
            normalized.append(vec + [0.0] * (dim - len(vec)))
    return normalized, labels, dim


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    correct = (preds == labels).float().mean().item()
    return float(correct)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router from RouterGalaxy JSONL.")
    parser.add_argument("--input", required=True, help="RouterGalaxy JSONL path.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    entries = _load_entries(args.input)
    embeddings, labels, embedding_dim = _extract_samples(entries)
    if not embeddings:
        raise SystemExit(f"No labeled entries found in {args.input}")

    if not embedding_dim:
        raise SystemExit("Failed to infer embedding dimension from RouterGalaxy.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(embedding_dim, int(args.hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_dim), 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            acc = _accuracy(logits.detach(), y)
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} acc={acc:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": int(args.hidden_dim),
            "labels": {"calculus": 1, "gsm8k": 0},
            "source": args.input,
        },
        output,
    )
    print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
