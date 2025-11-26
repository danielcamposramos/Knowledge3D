#!/usr/bin/env python3
"""
Train sovereign ARC-AGI pipeline on a staged set before committing discoveries.

Strategy:
- Select 25 tasks (default) from provided directories (training/evaluation).
- Run 3 epochs (Tesla-inspired) before committing shadow discoveries.
- Uses staged DualShadowCopy so galaxies are updated only after epochs.

Usage:
  /home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
    python scripts/train_arc_sovereign_loop.py \\
      --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \\
      --max-tasks 25 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/arc_agi")
DRAWING_CHECKPOINT = CHECKPOINT_DIR / "drawing_galaxy.json"
GRAMMAR_CHECKPOINT = CHECKPOINT_DIR / "grammar_galaxy.json"
SHADOW_CHECKPOINT = CHECKPOINT_DIR / "shadow_copy.json"

from knowledge3d.training.arc_agi import SovereignAIPipeline
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def load_task(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_tasks(dirs: List[Path], limit: int) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    if not files:
        return []
    mid = len(files) // 2
    tail = files[mid:]
    random.shuffle(tail)
    return tail[:limit]

def save_checkpoints(pipeline: SovereignAIPipeline) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.drawing.save(DRAWING_CHECKPOINT)
    pipeline.grammar.save(GRAMMAR_CHECKPOINT)
    pipeline.shadow.save(SHADOW_CHECKPOINT)


def run_epoch(
    pipeline: SovereignAIPipeline,
    task_files: List[Path],
    executor: ARCRPNExecutor,
    epoch: int,
    args: argparse.Namespace,
) -> Dict:
    correct = 0
    for idx, path in enumerate(task_files):
        task_id = f"{path.stem}_e{epoch}"
        task = load_task(path)
        test_pairs = task.get("test", [])
        train_examples = task.get("train", [])
        if not test_pairs:
            continue
        test_input = test_pairs[0]["input"]
        test_output = test_pairs[0]["output"]

        result = pipeline.process_task(
            task_id,
            test_input,
            train_examples=train_examples,
            expected_output=test_output,
            top_k=args.top_k,
        )
        predicted = result.output_grid if result.output_grid is not None else executor.execute(test_input, result.best_program)
        if predicted == test_output:
            correct += 1
        if idx % 5 == 0:
            print(f"  [{epoch+1}:{idx+1}/{len(task_files)}] {task_id} score={result.score:.2f} type={result.program_type}")

    return {"correct": correct, "total": len(task_files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arc-dirs", nargs="+", type=Path, required=True, help="One or more ARC task directories.")
    ap.add_argument("--max-tasks", type=int, default=25, help="Number of tasks to stage.")
    ap.add_argument("--epochs", type=int, default=3, help="Number of epochs before committing shadow.")
    ap.add_argument("--cycles", type=int, default=3, help="Number of cycles (Tesla-inspired).")
    ap.add_argument(
        "--matryoshka-dim",
        type=int,
        default=512,
        help="Matryoshka dimension (64-16384, higher = more capacity).",
    )
    ap.add_argument("--top-k", type=int, default=12, help="Top-K routing candidates.")
    args = ap.parse_args()

    if args.max_tasks <= 0:
        print("max-tasks must be > 0")
        return

    print("Initializing sovereign pipeline...")
    pipeline = SovereignAIPipeline(matryoshka_dim=args.matryoshka_dim, staged_shadow=True)
    executor = ARCRPNExecutor()

    # Load persisted state
    print("\n[LOADING] Galaxy state from checkpoints...")
    pipeline.drawing.load(DRAWING_CHECKPOINT)
    pipeline.grammar.load(GRAMMAR_CHECKPOINT)
    pipeline.shadow.load(SHADOW_CHECKPOINT)
    print(f"  Drawing shapes: {len(pipeline.drawing.shapes)}")
    print(f"  Grammar rules: {len(pipeline.grammar.rules)}")
    print(f"  Shadow entries: {len(pipeline.shadow.library)}")

    print(f"Staged training: tasks per cycle={args.max_tasks}, epochs={args.epochs}, cycles={args.cycles}")
    epoch_stats = []
    for cycle in range(args.cycles):
        task_files = collect_tasks(args.arc_dirs, args.max_tasks)
        # Sort tasks by grid area (easy first, hard last)
        def _area(path: Path) -> int:
            try:
                t = load_task(path)
                grid = t.get("test", [{}])[0].get("input", [[]])
                return len(grid) * len(grid[0]) if grid and grid[0] else 0
            except Exception:
                return 0

        task_files.sort(key=_area)

        print(f"\nCycle {cycle+1}/{args.cycles} using {len(task_files)} tasks...")
        for epoch in range(args.epochs):
            stats = run_epoch(pipeline, task_files, executor, epoch + cycle * args.epochs, args)
            epoch_stats.append(stats)
            print(f"  Epoch {epoch+1} (cycle {cycle+1}): {stats['correct']}/{stats['total']} correct ({stats['correct']/max(1,stats['total']):.2%})")
            # Save after every epoch to accumulate discoveries incrementally
            print(f"  [SAVE] Saving checkpoints after epoch {epoch+1}...")
            save_checkpoints(pipeline)

        # Dedup/prune after each cycle to keep quality high
        print("\n[PRUNING] Removing low-quality duplicates...")
        prune_stats = pipeline.shadow.prune_low_quality()
        print(
            f"  Removed programs: {prune_stats['removed_programs']}, "
            f"removed_from_library={prune_stats['removed_from_library']}, "
            f"unique_remaining={prune_stats['unique_remaining']}"
        )

    # Commit staged discoveries after all epochs
    pipeline.shadow.commit_pending()
    pruned = pipeline.shadow.prune_discovered(executor)
    summary = pipeline.summary()
    print("\n[SAVING] Updated galaxy state to checkpoints...")
    save_checkpoints(pipeline)

    print("\nCommit complete.")
    print(f"Pruned: {pruned}")
    print(f"Drawing shapes: {summary['drawing_shapes']}, Grammar rules: {summary['grammar_rules']}, Shadow entries: {summary['shadow_entries']}")
    print("Epoch stats:", epoch_stats)


if __name__ == "__main__":
    main()
