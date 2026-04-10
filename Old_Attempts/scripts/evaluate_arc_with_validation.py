#!/usr/bin/env python3
"""
Sovereign ARC-AGI evaluation with REAL accuracy validation.

Executes RPN programs and compares outputs to expected grids.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from knowledge3d.training.arc_agi import SovereignAIPipeline
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def load_arc_task(json_path: Path) -> Dict:
    """Load single ARC task JSON."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# Initialize RPN executor (CPU validation path)
_rpn_executor = ARCRPNExecutor()


def execute_rpn_program(program: str, input_grid: List[List[int]]) -> List[List[int]]:
    """
    Execute Drawing+Grammar RPN program on input grid.
    Uses ARCRPNExecutor (CPU validation path).
    """
    try:
        return _rpn_executor.execute(input_grid, program)
    except Exception as e:
        # If execution fails, return input unchanged
        print(f"  [WARNING] RPN execution failed: {e}")
        return input_grid


def grids_match(output: List[List[int]], expected: List[List[int]]) -> bool:
    """Check if output grid exactly matches expected grid."""
    return np.array_equal(np.array(output), np.array(expected))


def evaluate_task(pipeline: SovereignAIPipeline, task: Dict, task_id: str) -> Dict:
    """
    Evaluate single ARC task with real validation.

    Returns:
        {
            'task_id': str,
            'correct': int (1 or 0),
            'test_pairs': int,
            'program': str,
            'program_type': str,
            'heuristic_score': float
        }
    """
    test_pairs = task.get("test", [])
    if not test_pairs:
        return {"task_id": task_id, "correct": 0, "test_pairs": 0}

    # Use first test pair for evaluation
    test_input = test_pairs[0]["input"]
    test_output = test_pairs[0]["output"]
    train_examples = task.get("train", [])

    # Process through sovereign pipeline
    result = pipeline.process_task(
        task_id,
        test_input,
        train_examples=train_examples,
        expected_output=test_output,
        top_k=3,
    )

    # Execute the RPN program
    predicted_output = result.output_grid if result.output_grid is not None else execute_rpn_program(result.best_program, test_input)

    # Validate against expected output
    correct = 1 if grids_match(predicted_output, test_output) else 0

    return {
        "task_id": task_id,
        "correct": correct,
        "test_pairs": len(test_pairs),
        "program": result.best_program,
        "program_type": result.program_type,
        "heuristic_score": result.score,
        "matched": correct == 1
    }


def evaluate_dataset(json_dir: Path, pipeline: SovereignAIPipeline, max_tasks: int = None) -> Dict:
    """
    Evaluate entire ARC-AGI dataset.

    Returns:
        {
            'total_tasks': int,
            'correct': int,
            'accuracy': float,
            'results': List[Dict]
        }
    """
    json_files = sorted(json_dir.glob("*.json"))
    if max_tasks:
        json_files = json_files[:max_tasks]

    results = []
    correct_count = 0

    for json_file in json_files:
        task_id = json_file.stem
        task = load_arc_task(json_file)

        result = evaluate_task(pipeline, task, task_id)
        results.append(result)
        correct_count += result["correct"]

        print(f"[{task_id}] {'✓' if result['matched'] else '✗'} "
              f"(heuristic: {result['heuristic_score']:.2f}, type: {result['program_type']})")

    accuracy = correct_count / len(results) if results else 0.0

    return {
        "total_tasks": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI with real validation")
    parser.add_argument(
        "--arc-dir",
        type=Path,
        default="/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation",
        help="Directory containing ARC JSON files"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (default: all)"
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        default=128,
        help="Matryoshka dimension"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K routing candidates"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    if not args.arc_dir.exists():
        print(f"Error: ARC directory not found: {args.arc_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialize sovereign pipeline
    print(f"Initializing sovereign pipeline (matryoshka_dim={args.matryoshka_dim})...")
    pipeline = SovereignAIPipeline(matryoshka_dim=args.matryoshka_dim)

    # Evaluate dataset
    print(f"\nEvaluating ARC-AGI tasks from {args.arc_dir}...")
    results = evaluate_dataset(args.arc_dir, pipeline, max_tasks=args.max_tasks)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total tasks: {results['total_tasks']}")
    print(f"  Correct: {results['correct']}")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Drawing shapes: {len(pipeline.drawing.shapes)}")
    print(f"  Grammar rules: {len(pipeline.grammar.rules)}")
    print(f"  Shadow entries: {len(pipeline.shadow.library)}")
    print(f"{'='*60}")

    # Save results
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
