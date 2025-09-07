from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .rssm import RSSM
from .dataset import load_graph_from_logs, TrajDataset


def train_world_model(log_dir: Path, out_dir: Path, epochs: int = 3, seq_len: int = 16, batch_size: int = 64, lr: float = 1e-3, device: str = "cuda") -> dict:
    ids, neighbors, positions = load_graph_from_logs(log_dir)
    ds = TrajDataset(ids, neighbors, positions, seq_len=seq_len, n_samples=20000)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    mdl = RSSM(hidden=64).to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    mdl.train()
    for ep in range(max(1, epochs)):
        tot = 0.0
        n = 0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = mdl(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n += 1
        print(f"epoch {ep+1}/{epochs} loss={tot/max(1,n):.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mdl.state_dict(), str(out_dir / "rssm.pt"))
    return {"epochs": epochs, "seq_len": seq_len, "batch": batch_size}


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train tiny world model (RSSM) from logs")
    ap.add_argument("--logs", default=str((Path(__file__).resolve().parents[3].parent / (Path(__file__).resolve().parents[3].name + ".local") / "logs")))
    ap.add_argument("--out", default=str((Path(__file__).resolve().parents[3].parent / (Path(__file__).resolve().parents[3].name + ".local") / "models" / "world_rssm")))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    res = train_world_model(Path(args.logs), Path(args.out), epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch, lr=args.lr, device=args.device)
    print(res)


if __name__ == "__main__":
    main()

