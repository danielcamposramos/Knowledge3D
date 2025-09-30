from __future__ import annotations

"""Stable long-run driver for fused multi-trainer (PTX-only, single process).

Warms up PTX ops before starting training to avoid early teardown races on
this driver. Launch this module with nohup for overnight runs.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.phase25.long_run \
    --epochs 50 --limit 300 --dims "64,64,64,64" \
    --keys math,gsm8k,metamath,aime,amc,olympiad,algebra
"""

import argparse
import os
from typing import List

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS  # type: ignore
from knowledge3d.tools.phase25 import fused_multi_trainer as fmt  # type: ignore


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Stable long-run fused multi-trainer")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--dims", type=str, default="64,64,64,64")
    ap.add_argument("--keys", type=str, default="math,gsm8k,metamath,aime,amc,olympiad,algebra")
    ap.add_argument("--lr-math", type=float, default=5e-4)
    ap.add_argument("--lr-rpn", type=float, default=1e-3)
    args = ap.parse_args()

    os.environ.setdefault("K3D_PTX_STRICT", "1")
    os.environ.setdefault("K3D_FORCE_PTX_FUSE", "1")
    os.environ["K3D_FUSE_DIMS"] = args.dims
    # Optional: avoid modality/image/audio PTX during warmup
    os.environ.setdefault("K3D_DISABLE_MEDIA_LOOKUP", "1")
    os.environ.setdefault("K3D_DISABLE_SHAPE_GENERATION", "1")

    # Warm up PTX kernels
    try:
        PTX_OPS.text_modality("warmup")
    except Exception:
        pass
    try:
        _ = AdaptedFusedHead()
    except Exception:
        pass

    keys: List[str] = [k.strip() for k in args.keys.split(",") if k.strip()]
    fmt.run(keys, limit=int(args.limit), epochs=int(args.epochs), lr_math=float(args.lr_math), lr_rpn=float(args.lr_rpn))
    print("✅ long_run complete")


if __name__ == "__main__":
    main()

