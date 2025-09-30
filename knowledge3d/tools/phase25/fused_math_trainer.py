"""Train fused head projection+math head using local HF math datasets (offline).

Usage (GPU env):
  PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 \
  python -m knowledge3d.tools.phase25.fused_math_trainer --epochs 2 --limit 300

After training, the trainer saves `viewer/public/house/house_math_head.pt` and
packs it into the active House GLB as appliance `fused_math`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset, DownloadConfig  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    DownloadConfig = None  # type: ignore

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.tools.phase25 import math_bench_evaluator as mbe  # type: ignore
from knowledge3d.tools.weights_in_glb import pack_pt_into_glb  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
HOUSE_GLB = ROOT / "viewer/public/houses/default/memory_house.glb"


def _load_offline(repo: str, split: str, limit: int):
    if load_dataset is None or DownloadConfig is None:
        return []
    try:
        ds = load_dataset(repo, split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        return []
    total = min(limit, len(ds))
    return [ds[i] for i in range(total)]


def _extract(repo: str, limit: int) -> List[tuple[str, Optional[str]]]:
    rows: List[tuple[str, Optional[str]]] = []
    for split in ("train", "validation", "test"):
        data = _load_offline(repo, split, limit)
        for row in data:
            q = str(row.get("problem") or row.get("question") or row.get("Problem") or "")
            sol = str(row.get("solution") or row.get("Solution") or row.get("answer") or "")
            expected = mbe._normalize(mbe._coerce_answer(sol) or mbe._coerce_answer(str(row.get("answer") or "")))
            rows.append((q, expected))
        if rows:
            break
    return rows[:limit]


def run(repos: List[str], limit: int, epochs: int, lr: float) -> None:
    # Ensure strict PTX + in-head fusion by default
    os.environ.setdefault("K3D_PTX_STRICT", "1")
    os.environ.setdefault("K3D_FORCE_PTX_FUSE", "1")

    fh = AdaptedFusedHead()
    # Tweak learning rate
    for g in fh._opt.param_groups:
        g["lr"] = float(lr)

    # Build training pool
    pairs: List[tuple[str, Optional[str]]] = []
    for r in repos:
        pairs.extend(_extract(r, limit))
    # Filter to integer targets 0..999
    trainset: List[tuple[str, int]] = []
    for q, exp in pairs:
        if not q or exp is None:
            continue
        try:
            yi = int(str(exp))
            if 0 <= yi <= 999:
                trainset.append((q, yi))
        except Exception:
            continue
    if not trainset:
        print("⚠️  No suitable training examples found in local cache.")
        return
    print(f"📦 Fused math train samples: {len(trainset)}")

    # Training loop
    for ep in range(1, int(max(1, epochs)) + 1):
        n = 0
        for q, yi in trainset:
            # Build fused embedding via in-head PTX fusion
            emb = fh._build_ptx_fused_embedding(q)
            fh.train_step(emb, str(yi), lr=lr)
            n += 1
            if n % 200 == 0:
                print(f"  step {n}/{len(trainset)}")
        print(f"🧮 Fused-math Epoch {ep}: {len(trainset)} steps")
    # Save + pack into GLB
    fh._save_math_head()
    ckpt = ROOT / "viewer/public/house/house_math_head.pt"
    if HOUSE_GLB.exists() and ckpt.exists():
        try:
            pack_pt_into_glb(HOUSE_GLB, ckpt, "fused_math")
            print(f"📦 Packed fused_math into {HOUSE_GLB}")
        except Exception as e:
            print(f"⚠️  Failed to pack fused_math: {e}")
    print("✅ Fused math training complete.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train fused projection+math head (offline HF cache)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--repos", type=str, default="hendrycks/competition_math,openai/gsm8k,meta-math/MetaMathQA")
    args = ap.parse_args()
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    run(repos, limit=int(args.limit), epochs=int(args.epochs), lr=float(args.lr))


if __name__ == "__main__":
    main()

