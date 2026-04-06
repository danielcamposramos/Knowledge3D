#!/usr/bin/env python3
"""Bounded cloud-model evaluation harness for the K3D knowledge proceduralizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.tools.knowledge_proceduralizer import run_model_eval_harness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="glm-5:cloud,kimi-k2-thinking:cloud,qwen3.5:397b-cloud,deepseek-v3.2:cloud",
        help="Comma-separated proceduralizer candidate models.",
    )
    parser.add_argument("--provider", default="ollama")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("../Knowledge3D.local/results/proceduralizer_model_eval"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    summary = run_model_eval_harness(
        models=models,
        capture_dir=args.capture_dir,
        timeout_seconds=float(args.timeout_seconds),
        provider=str(args.provider).strip().lower(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
