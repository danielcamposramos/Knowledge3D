from __future__ import annotations

"""
Train an RLWHF answer policy end-to-end:
 - Ensures RLWHF dataset exists (builds from logs or offline benchmark if needed)
 - Trains a small causal LM with reward-weighted SFT

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.train_rlwhf_policy \
    --dataset docs/reports/training/rlwhf_dataset.jsonl \
    --out ../Knowledge3D.local/models/rlwhf_policy \
    --model distilgpt2 --epochs 1 --batch 4
"""

import argparse
import json
from pathlib import Path


def ensure_dataset(repo_root: Path, dataset: Path) -> None:
    if dataset.exists() and dataset.stat().st_size > 0:
        return
    # Prefer live logs; else try offline benchmark
    logs = repo_root.parent / f"{repo_root.name}.local" / "logs"
    bench = repo_root / "docs" / "reports" / "status" / "chat_benchmark_live.json"
    if logs.exists():
        from knowledge3d.tools.training_pipelines.build_rlwhf_dataset import main as build_main  # type: ignore
        import sys
        sys.argv = ["build", "--logs", str(logs), "--out", str(dataset), "--summary", str(repo_root / "docs" / "reports" / "status" / "rlwhf_summary.json")]
        build_main()
    elif bench.exists():
        from knowledge3d.tools.training_pipelines.rlwhf_from_offline_benchmark import main as off_main  # type: ignore
        import sys
        sys.argv = ["build_off", "--gltf", str(repo_root / "viewer" / "public" / "galaxy.cross.glb"), "--bench", str(bench), "--out", str(dataset)]
        off_main()
    else:
        raise SystemExit("No RLWHF sources found: run live or offline benchmark first.")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Train RLWHF policy via reward-weighted SFT")
    p.add_argument("--dataset", default=str(Path("docs/reports/training/rlwhf_dataset.jsonl")))
    p.add_argument("--out", default=str(Path("../Knowledge3D.local/models/rlwhf_policy")))
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset = Path(args.dataset)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    ensure_dataset(repo_root, dataset)
    from knowledge3d.models.rlwhf_policy import train  # type: ignore
    info = train(dataset, Path(args.out), model_id=str(args.model), epochs=int(args.epochs), batch_size=int(args.batch), max_len=int(args.max_len), lr=float(args.lr))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

