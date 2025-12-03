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
from knowledge3d.training.arc_agi.sovereign_pipeline import _fuzzy_match
import pickle
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def load_task(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_tasks(dirs: List[Path], limit: int) -> List[Path]:
    """
    Mixed curriculum: sample tasks with proportional difficulty.

    Strategy (Daniel's recommendation):
    - 1/3 from training set (easy - basic patterns)
    - 1/3 from first half of evaluation (mid - moderate complexity)
    - 1/3 from second half of evaluation (hard - competition difficulty)

    This provides:
    - Easy wins to build library
    - Mid challenges to stretch capabilities
    - Hard aspirational targets for advanced patterns
    """
    training_files: List[Path] = []
    evaluation_files: List[Path] = []

    for d in dirs:
        if not d.exists():
            continue
        if "training" in str(d):
            training_files.extend(sorted(d.glob("*.json")))
        elif "evaluation" in str(d):
            evaluation_files.extend(sorted(d.glob("*.json")))

    # Calculate proportional splits (1/3 each category)
    easy_count = limit // 3
    mid_count = limit // 3
    hard_count = limit - easy_count - mid_count  # Remainder goes to hard

    selected: List[Path] = []

    # Easy: sample from training set
    if training_files:
        random.shuffle(training_files)
        selected.extend(training_files[:easy_count])
        print(f"  [CURRICULUM] Easy (training): {len(selected)} tasks")

    # Mid + Hard: sample from evaluation set
    if evaluation_files:
        mid_point = len(evaluation_files) // 2
        easy_eval = evaluation_files[:mid_point]
        hard_eval = evaluation_files[mid_point:]

        # Mid: first half of evaluation
        random.shuffle(easy_eval)
        mid_tasks = easy_eval[:mid_count]
        selected.extend(mid_tasks)
        print(f"  [CURRICULUM] Mid (easy eval): {len(mid_tasks)} tasks")

        # Hard: second half of evaluation
        random.shuffle(hard_eval)
        hard_tasks = hard_eval[:hard_count]
        selected.extend(hard_tasks)
        print(f"  [CURRICULUM] Hard (hard eval): {len(hard_tasks)} tasks")

    # Shuffle final mix so difficulty isn't sequential
    random.shuffle(selected)
    print(f"  [CURRICULUM] Total mixed: {len(selected)} tasks (easy={easy_count}, mid={mid_count}, hard={hard_count})")

    return selected


def select_108_tasks_tesla() -> List[Path]:
    """Select 108 tasks as 36 easy (training), 36 medium (mid eval), 36 hard (late eval)."""
    train_dir = Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training")
    eval_dir = Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation")
    train_tasks = sorted(train_dir.glob("*.json"))
    eval_tasks = sorted(eval_dir.glob("*.json"))

    easy = train_tasks[:36]
    n_eval = len(eval_tasks)
    start_medium = n_eval // 3
    medium = eval_tasks[start_medium : start_medium + 36]
    hard = eval_tasks[-36:]

    selected = easy + medium + hard
    print(f"[TASK SELECT] Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}, Total: {len(selected)}")
    return selected[:108]


def generate_tesla_curriculum(task_files: List[Path], pipeline: SovereignAIPipeline, total_epochs: int = 162) -> List[List[Path]]:
    """3-phase curriculum: mix easy/medium/hard each epoch (108 tasks)."""
    import random

    # Derive difficulty from shadow history.
    easy: List[Path] = []
    medium: List[Path] = []
    hard: List[Path] = []
    for p in task_files:
        tid = p.stem
        hist = pipeline.shadow.get_task_history(tid)
        sr = hist.get("success_rate", 0.5) if hist else 0.5
        if sr > 0.7:
            easy.append(p)
        elif sr < 0.3:
            hard.append(p)
        else:
            medium.append(p)

    if not easy or not medium or not hard:
        third = len(task_files) // 3
        easy = easy or task_files[:third]
        medium = medium or task_files[third : 2 * third]
        hard = hard or task_files[2 * third :]

    curriculum: List[List[Path]] = []
    for _ in range(total_epochs):
        e = easy[:]
        m = medium[:]
        h = hard[:]
        random.shuffle(e)
        random.shuffle(m)
        random.shuffle(h)
        epoch_tasks = (e[:36] + m[:36] + h[:36])
        random.shuffle(epoch_tasks)
        # Ensure exactly 108 entries
        if len(epoch_tasks) < 108:
            filler = (easy + medium + hard)
            random.shuffle(filler)
            epoch_tasks.extend(filler[: max(0, 108 - len(epoch_tasks))])
        curriculum.append(epoch_tasks[:108])
    return curriculum

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
        fuzzy_score = getattr(result, "fuzzy_score", None)
        # Fallback compute if pipeline did not supply.
        if fuzzy_score is None:
            fuzzy_score = _fuzzy_match(predicted, test_output) if test_output is not None else 0.0
        is_correct = getattr(result, "correct", False)
        if not is_correct and test_output is not None:
            is_correct = predicted == test_output or fuzzy_score >= 0.80

        if is_correct:
            correct += 1
            if fuzzy_score >= 0.80 and predicted != test_output:
                print(f"  [FUZZY EPOCH] Task {task_id}: fuzzy_score={fuzzy_score:.2f} accepted")
        elif 0.70 <= fuzzy_score < 0.80:
            print(f"  [NEAR MISS EPOCH] Task {task_id}: fuzzy_score={fuzzy_score:.2f} (70-80%)")
        if idx % 5 == 0:
            print(f"  [{epoch+1}:{idx+1}/{len(task_files)}] {task_id} score={result.score:.2f} type={result.program_type}")

    return {"correct": correct, "total": len(task_files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arc-dirs", nargs="+", type=Path, required=True, help="One or more ARC task directories.")
    ap.add_argument("--max-tasks", type=int, default=25, help="Number of tasks to stage.")
    ap.add_argument("--epochs", type=int, default=162, help="Number of epochs before committing shadow.")
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

    embedding_galaxy = None
    galaxy_path = Path("/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl")
    if galaxy_path.exists():
        try:
            with open(galaxy_path, "rb") as f:
                embedding_galaxy = pickle.load(f)
            print(f"[INIT] Loaded precomputed embeddings: {len(embedding_galaxy)} entries")
        except Exception as e:
            print(f"[INIT] Warning: could not load precomputed embeddings ({e}); continuing without cache")

    print("Initializing sovereign pipeline...")
    pipeline = SovereignAIPipeline(
        matryoshka_dim=args.matryoshka_dim,
        staged_shadow=True,
        embedding_galaxy=embedding_galaxy,
    )
    executor = ARCRPNExecutor()

    # Load persisted state
    print("\n[LOADING] Galaxy state from checkpoints...")
    pipeline.drawing.load(DRAWING_CHECKPOINT)
    pipeline.grammar.load(GRAMMAR_CHECKPOINT)
    pipeline.shadow.load(SHADOW_CHECKPOINT)
    pipeline.drawing.add_scale_invariant_primitives()
    print(f"  Drawing shapes: {len(pipeline.drawing.shapes)}")
    print(f"  Grammar rules: {len(pipeline.grammar.rules)}")
    print(f"  Shadow entries: {len(pipeline.shadow.library)}")

    print(f"Staged training: tasks per cycle={args.max_tasks}, epochs={args.epochs}, cycles={args.cycles}")
    epoch_stats = []
    base_task_files: List[Path] = []
    curriculum = None
    if args.max_tasks == 108:
        print("[TESLA SELECT] Using 36 easy + 36 medium + 36 hard (108 total)")
        base_task_files = select_108_tasks_tesla()
        curriculum = generate_tesla_curriculum(base_task_files, pipeline, total_epochs=args.epochs)

    best_epoch_accuracy = 0.0
    for cycle in range(args.cycles):
        task_files = base_task_files or collect_tasks(args.arc_dirs, args.max_tasks)
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
            epoch_tasks = curriculum[epoch] if curriculum else task_files
            stats = run_epoch(pipeline, epoch_tasks, executor, epoch + cycle * args.epochs, args)
            epoch_stats.append(stats)
            accuracy = stats['correct'] / max(1, stats['total'])
            print(f"  Epoch {epoch+1} (cycle {cycle+1}): {stats['correct']}/{stats['total']} correct ({accuracy:.2%})")

            global_epoch = epoch + 1 + cycle * args.epochs
            if (global_epoch % 10 == 0) or (accuracy > best_epoch_accuracy):
                pipeline._log_vocabulary_quality(global_epoch)
                best_epoch_accuracy = max(best_epoch_accuracy, accuracy)

            # SOVEREIGN: Update router from discoveries (closes feedback loop!)
            print(f"  [FEEDBACK] Updating router from shadow copy discoveries...")
            update_stats = pipeline.router.update_from_discoveries(pipeline.shadow, top_n=100)
            print(f"    Processed: {update_stats['processed']}, High-quality: {update_stats['high_quality']}, Pattern types: {update_stats['pattern_types']}")

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
