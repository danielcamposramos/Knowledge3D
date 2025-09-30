"""Train the ARC Grid Head (prototype) inside the fused head.

This trainer scans a local ARC-AGI training set and runs a simple supervised
loop to fit a fixed 10x10 output head. It is a minimal scaffold to unblock
Phase 25 ARC work; refine architecture and input encoding next.

Usage (GPU env):
  conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.phase25.arc_grid_trainer \
      --arc-root ../Knowledge3D.local/datasets/exams/arc-src/data/training --limit 200 --epochs 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore


def _iter_arc_pairs(root: Path, limit: int | None = None) -> Iterable[Tuple[List[List[int]], List[List[int]]]]:
    if not root.exists():
        return []
    n = 0
    for p in sorted(root.iterdir()):
        if p.suffix != ".json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        train = obj.get("train") if isinstance(obj, dict) else None
        if not isinstance(train, list):
            continue
        for pair in train:
            if not isinstance(pair, dict):
                continue
            i = pair.get("input"); o = pair.get("output")
            if isinstance(i, list) and isinstance(o, list):
                yield i, o
                n += 1
                if limit is not None and n >= int(limit):
                    return


def _grid_to_fused_embedding(g: List[List[int]], dim: int = 2048) -> List[float]:
    arr = np.asarray(g, dtype=np.int64)
    hist = np.bincount(arr.reshape(-1), minlength=10).astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    # Tile histogram to fill 2048
    reps = int(np.ceil(dim / hist.size))
    vec = np.tile(hist, reps)[:dim].astype(np.float32)
    return [float(x) for x in vec]


def run(arc_root: Path, limit: int, epochs: int) -> None:
    fh = AdaptedFusedHead()
    pairs = list(_iter_arc_pairs(arc_root, limit))
    if not pairs:
        print(f"⚠️  No ARC pairs found under {arc_root}")
        return
    print(f"📦 ARC train pairs: {len(pairs)}")
    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        for i, (inp, out) in enumerate(pairs, 1):
            fe = _grid_to_fused_embedding(inp)
            loss = fh.arc_grid_train_step(fe, out)
            losses.append(loss)
            if i % 100 == 0:
                print(f"  step {i}/{len(pairs)} loss={loss:.4f}")
        avg = float(sum(losses) / max(1, len(losses)))
        print(f"🧩 ARC Epoch {ep}: avg_loss={avg:.4f} ({len(losses)} samples)")
        fh._save_arc_head()
    print("✅ ARC grid training complete; checkpoint saved.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train ARC grid head (prototype)")
    ap.add_argument("--arc-root", type=str, default="../Knowledge3D.local/datasets/exams/arc-src/data/training")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    run(Path(args.arc_root), limit=int(args.limit), epochs=int(args.epochs))


if __name__ == "__main__":  # pragma: no cover
    main()

