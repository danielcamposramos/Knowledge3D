#!/usr/bin/env python3
"""
Train the TRM (Tiny Recursive Model) on reasoning patterns using ARC-AGI tasks.

This script keeps knowledge in the RPN embeddings and only teaches the TRM how
to transform embeddings. It runs inside the sanctioned training environment
(`k3d-cranium`) and stores artifacts under Knowledge3D.local/.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.training.reasoning.arc_dataset import (
    prepare_arc_reasoning_cache,
    load_arc_reasoning_cache,
)

DEFAULT_RPN_PATH = Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl")
DEFAULT_DATASET_ROOT = Path("/K3D/Knowledge3D.local/datasets/arc_agi")
DEFAULT_WEIGHTS_IN = Path("/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz")
DEFAULT_WEIGHTS_OUT = Path("/K3D/Knowledge3D.local/models/trm_weights_arc_reasoning.npz")
DEFAULT_LOG_DIR = Path("/K3D/Knowledge3D.local/logs/trm_reasoning")


def _device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpn-embeddings", type=Path, default=DEFAULT_RPN_PATH,
                        help="Path to persisted RPN embeddings (Galaxy/House).")
    parser.add_argument("--weights-in", type=Path, default=DEFAULT_WEIGHTS_IN,
                        help="Initial TRM weights (seeded from RPN).")
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_OUT,
                        help="Destination for trained TRM weights.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="Root directory for ARC-AGI dataset and caches.")
    parser.add_argument("--cache-path", type=Path, default=None,
                        help="Optional override for ARC reasoning cache path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="Optional cap on number of ARC reasoning pairs.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Regenerate the ARC cache even if it exists.")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip ARC dataset download (expects dataset already present).")
    parser.add_argument("--device", type=str, default=_device_default())
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tesla-steps", type=int, default=6,
                        help="Number of TRM recursions (keep aligned with Tesla 3/6/9).")
    return parser.parse_args()


class TRMPyTorch(nn.Module):
    def __init__(self, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, W4: np.ndarray):
        super().__init__()
        self.W1 = nn.Parameter(torch.from_numpy(W1))
        self.W2 = nn.Parameter(torch.from_numpy(W2))
        self.W3 = nn.Parameter(torch.from_numpy(W3))
        self.W4 = nn.Parameter(torch.from_numpy(W4))

    def forward(self, q: torch.Tensor, n_steps: int = 6) -> torch.Tensor:
        y = torch.zeros_like(q)
        z = torch.zeros_like(q)
        for _ in range(n_steps):
            combined = q + y + z
            hidden = torch.matmul(combined, self.W1.t())
            hidden = F.silu(hidden)
            z = torch.matmul(hidden, self.W2.t())

            combined2 = y + z
            hidden2 = torch.matmul(combined2, self.W3.t())
            hidden2 = F.silu(hidden2)
            y = torch.matmul(hidden2, self.W4.t())
        return y


@dataclass
class TrainingStats:
    epoch: int
    train_loss: float
    duration_sec: float


def load_initial_weights(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Initial TRM weights not found: {path}")
    payload = np.load(path)
    W1 = payload["W1"].astype(np.float32)
    W2 = payload["W2"].astype(np.float32)
    W3 = payload["W3"].astype(np.float32)
    W4 = payload["W4"].astype(np.float32)
    payload.close()
    return W1, W2, W3, W4


def save_weights(path: Path, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor, W4: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W1=W1.detach().cpu().numpy(),
        W2=W2.detach().cpu().numpy(),
        W3=W3.detach().cpu().numpy(),
        W4=W4.detach().cpu().numpy(),
    )


def ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_log(log_dir: Path, stats: dict) -> None:
    ensure_log_dir(log_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"trm_arc_training_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"📝 Training log written to {log_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"🚀 Training TRM on ARC reasoning (device={device})")

    # Load knowledge embeddings (House/Galaxy consolidation)
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings(args.rpn_embeddings)
    print(f"✅ Loaded RPN embeddings from {args.rpn_embeddings}")

    # Prepare ARC reasoning cache
    cache_path = prepare_arc_reasoning_cache(
        rpn_engine.embed_sentence,
        dataset_root=args.dataset_root,
        cache_path=args.cache_path,
        limit=args.limit_pairs,
        rebuild=args.rebuild_cache,
        download=not args.no_download,
    )
    cache = load_arc_reasoning_cache(cache_path)
    print(f"✅ Loaded ARC reasoning cache ({cache.questions.shape[0]} pairs)")

    dataset = TensorDataset(
        torch.from_numpy(cache.questions),
        torch.from_numpy(cache.answers),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Initialise TRM model
    W1, W2, W3, W4 = load_initial_weights(args.weights_in)
    model = TRMPyTorch(W1, W2, W3, W4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[TrainingStats] = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        epoch_loss = 0.0
        model.train()

        for batch_q, batch_target in dataloader:
            batch_q = batch_q.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_q, n_steps=args.tesla_steps)
            loss = F.mse_loss(y_pred, batch_target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * batch_q.size(0)

        epoch_loss /= len(dataset)
        duration = time.time() - start_time
        history.append(TrainingStats(epoch=epoch, train_loss=epoch_loss, duration_sec=duration))
        print(f"Epoch {epoch:02d} | loss={epoch_loss:.6f} | duration={duration:.1f}s")

    save_weights(args.weights_out, model.W1, model.W2, model.W3, model.W4)
    print(f"💾 Trained weights saved to {args.weights_out}")

    log_payload = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "tesla_steps": args.tesla_steps,
        "pairs": int(cache.questions.shape[0]),
        "cache_path": str(cache_path),
        "weights_in": str(args.weights_in),
        "weights_out": str(args.weights_out),
        "history": [asdict(s) for s in history],
        "metadata": cache.metadata,
    }
    write_log(args.log_dir, log_payload)


if __name__ == "__main__":
    main()
