#!/usr/bin/env python3
"""
Evaluate sovereign ARC-AGI pipeline (Drawing + Grammar galaxies).

This script is lightweight and deterministic:
- Uses DrawingGalaxy + GrammarGalaxy
- SovereignTRMRouter (GPU-only matryoshka projection)
- ProgramComposer + DualShadowCopy

It can run on a synthetic sample when no dataset is provided.

Usage:
    /home/daniel/miniforge/bin/conda run -n k3d-cranium \\
        python scripts/evaluate_arc_sovereign_ai.py --sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline


def load_arc_tasks(path: Path) -> Dict[str, Any]:
    """
    Load ARC-AGI tasks from a single JSON file.

    Supports both dict-based aggregates {task_id: {...}} and standard ARC files
    that contain a list of {"train": [...], "test": [...]} entries.
    """
    data = json.loads(path.read_text())
    tasks: Dict[str, Any] = {}
    if isinstance(data, dict):
        # Single-task file: {"train": [...], "test": [...]}
        if "test" in data and isinstance(data["test"], list):
            tasks[path.stem] = data["test"][0]["input"]
        else:
            for task_id, payload in data.items():
                if isinstance(payload, dict) and "test" in payload:
                    tasks[str(task_id)] = payload["test"][0]["input"]
    elif isinstance(data, list):
        for idx, payload in enumerate(data):
            if isinstance(payload, dict) and "test" in payload and payload["test"]:
                tasks[f"{path.stem}_{idx}"] = payload["test"][0]["input"]
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arc-json", type=Path, help="Path to ARC-AGI JSON (expects test/input per task).")
    ap.add_argument("--sample", action="store_true", help="Run on a tiny synthetic sample.")
    ap.add_argument("--matryoshka-dim", type=int, default=128, help="Matryoshka dimension for router.")
    ap.add_argument("--top-k", type=int, default=3, help="Top-k grammar rules to consider.")
    args = ap.parse_args()

    pipeline = SovereignAIPipeline(matryoshka_dim=args.matryoshka_dim)

    tasks: Dict[str, List[List[int]]] = {}
    if args.sample or not args.arc_json:
        tasks = {
            "sample_1": [[1, 0], [0, 2]],
            "sample_2": [[0, 3, 0], [3, 0, 3], [0, 3, 0]],
        }
    else:
        tasks = load_arc_tasks(args.arc_json)

    for tid, grid in tasks.items():
        result = pipeline.process_task(tid, grid, top_k=args.top_k)
        print(f"[{tid}] score={result.score:.3f} type={result.program_type} program={result.best_program[:80]}...")

    summary = pipeline.summary()
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()
