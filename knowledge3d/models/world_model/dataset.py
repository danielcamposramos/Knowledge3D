from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _iter_logs(log_dir: Path):
    for p in sorted(log_dir.glob("session-*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except OSError:
            continue


def load_graph_from_logs(log_dir: Path) -> Tuple[List[str], List[List[str]], List[List[float]]]:
    ids: List[str] = []
    neighbors: List[List[str]] = []
    positions: List[List[float]] = []
    for rec in _iter_logs(log_dir):
        if rec.get("type") == "event":
            ev = rec.get("event") or {}
            if ev.get("kind") == "dataset_graph":
                ids = list(ev.get("ids") or [])
                neighbors = list(ev.get("neighbors") or [[] for _ in range(len(ids))])
                positions = list(ev.get("positions") or [])
    return ids, neighbors, positions


class TrajDataset(Dataset):
    """Random-walk sequences over the dataset graph.

    Creates synthetic sequences by walking neighbors. This gives the world model
    a basic next-step prediction task in 3D.
    """

    def __init__(self, ids: List[str], neighbors: List[List[str]], positions: List[List[float]], seq_len: int = 16, n_samples: int = 10000):
        self.ids = ids
        self.seq_len = max(2, seq_len)
        self.pos = positions
        # Build id -> index map
        self.idx = {s: i for i, s in enumerate(ids)}
        # Normalize neighbors to indices
        self.nbr_idx: List[List[int]] = []
        for row in neighbors:
            self.nbr_idx.append([self.idx.get(j, -1) for j in row if j in self.idx])
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int):
        if not self.ids:
            x = np.zeros((self.seq_len, 3), dtype=np.float32)
            y = np.zeros((3,), dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)
        cur = random.randrange(0, len(self.ids))
        seq = [np.asarray(self.pos[cur], dtype=np.float32)]
        for _ in range(self.seq_len):
            nbrs = self.nbr_idx[cur]
            if not nbrs:
                cur = random.randrange(0, len(self.ids))
            else:
                cur = random.choice(nbrs)
            seq.append(np.asarray(self.pos[cur], dtype=np.float32))
        x = np.stack(seq[:-1], axis=0)
        y = seq[-1]
        return torch.from_numpy(x), torch.from_numpy(y)

