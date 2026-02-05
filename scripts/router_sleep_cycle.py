#!/usr/bin/env python3
"""
Router sleep cycle: consolidate RouterGalaxy memory into a new router skill.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print(f"[RouterSleep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Router sleep cycle orchestration.")
    parser.add_argument("--galaxy", default="data/router_galaxy_v1.jsonl", help="RouterGalaxy JSONL input.")
    parser.add_argument("--output-pt", default="data/router_v3.pt", help="Output router checkpoint.")
    parser.add_argument("--output-skill", default="data/skill_galaxy_router_v3.jsonl", help="Output router SkillGalaxy JSONL.")
    parser.add_argument("--skill-id", default="router_gatekeeper_v3", help="Skill ID for router skill.")
    parser.add_argument("--description", default="Router Gatekeeper V3 (Sleep Cycle)", help="Skill description.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--validate", action="store_true", help="Run sanity check during packaging.")
    args = parser.parse_args()

    galaxy_path = Path(args.galaxy)
    if not galaxy_path.exists():
        raise FileNotFoundError(f"RouterGalaxy not found: {galaxy_path}")

    train_cmd = [
        sys.executable,
        "scripts/train_router_from_galaxy.py",
        "--input",
        str(galaxy_path),
        "--output",
        str(args.output_pt),
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--learning-rate",
        str(args.learning_rate),
    ]
    _run(train_cmd)

    package_cmd = [
        sys.executable,
        "scripts/package_router_skill.py",
        "--input",
        str(args.output_pt),
        "--output",
        str(args.output_skill),
        "--skill-id",
        str(args.skill_id),
        "--description",
        str(args.description),
    ]
    if args.validate:
        package_cmd.append("--validate")
    _run(package_cmd)

    print(f"[RouterSleep] Completed. Skill -> {args.output_skill}")


if __name__ == "__main__":
    main()
