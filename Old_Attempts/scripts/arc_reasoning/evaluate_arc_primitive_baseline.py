"""Evaluate a simple primitive-detection baseline on ARC training tasks.

Approach:
  - Use the first train pair of each task to detect a primitive
    (rotate 90/180/270, flip H/V, or translate).
  - Apply the detected primitive to all train inputs and compare to train outputs.
  - Report per-task accuracy and aggregate stats.

This simulates a lightweight ARC-AGI evaluation using only single-primitive
transformations inferred from one example.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.reasoning.arc_dataset import ensure_arc_dataset, _iter_task_files, _load_task


def apply_primitive(processor: ARCGridProcessor, grid: List[List[int]], primitive: str, params: dict) -> List[List[int]]:
    """Apply detected primitive to a grid."""
    if primitive.endswith("_RECOLOR"):
        base = primitive.replace("_RECOLOR", "")
        transformed = grid
        if base == "ROTATE":
            angle = params.get("param", 0)
            transformed = processor._apply_rotation(grid, angle)  # type: ignore[attr-defined]
        elif base == "FLIP_H":
            transformed = processor._apply_flip_horizontal(grid)  # type: ignore[attr-defined]
        elif base == "FLIP_V":
            transformed = processor._apply_flip_vertical(grid)  # type: ignore[attr-defined]
        elif base == "TRANSLATE":
            dx, dy = params.get("param", (0, 0))
            transformed = _translate_grid(grid, dx, dy)
        elif base == "ROTATE_TRANSLATE":
            angle, dx, dy = params.get("param", (0, 0, 0))
            transformed = processor._apply_rotation(grid, angle)  # type: ignore[attr-defined]
            transformed = _translate_grid(transformed, dx, dy)
        src = params.get("src")
        dst = params.get("dst")
        return _recolor_grid(transformed, src, dst)
    if primitive.startswith("ROTATE_TRANSLATE"):
        angle = params.get("angle", 0)
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        rotated = processor._apply_rotation(grid, angle)  # type: ignore[attr-defined]
        return _translate_grid(rotated, dx, dy)
    if primitive.startswith("ROTATE_"):
        parts = primitive.split("_")
        angle = params.get("angle", int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)
        return processor._apply_rotation(grid, angle)  # type: ignore[attr-defined]
    if primitive == "FLIP_H":
        return processor._apply_flip_horizontal(grid)  # type: ignore[attr-defined]
    if primitive == "FLIP_V":
        return processor._apply_flip_vertical(grid)  # type: ignore[attr-defined]
    if primitive == "TRANSLATE":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        return _translate_grid(grid, dx, dy)
    if primitive == "RECOLOR":
        src = params.get("src")
        dst = params.get("dst")
        return _recolor_grid(grid, src, dst)
    # Unknown: return input unchanged
    return grid


def _translate_grid(grid: List[List[int]], dx: int, dy: int) -> List[List[int]]:
    """Translate grid contents by dx, dy with zero fill."""
    arr = np.array(grid, dtype=int)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for y in range(h):
        for x in range(w):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny, nx] = arr[y, x]
    return out.tolist()


def _recolor_grid(grid: List[List[int]], src: int, dst: int) -> List[List[int]]:
    arr = np.array(grid, dtype=int)
    arr[arr == src] = dst
    return arr.tolist()


def evaluate():
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))
    processor = ARCGridProcessor(matryoshka_dim=128, embedder_type="procedural")

    task_results = []
    primitive_counts = Counter()
    total_examples = 0
    total_correct = 0

    for task_path in task_files:
        task = _load_task(task_path)
        train = task.get("train", [])
        if len(train) < 1:
            continue

        ref = train[0]
        detected = processor.detect_spatial_primitive(ref["input"], ref["output"])
        primitive = detected.get("primitive", "UNKNOWN")
        primitive_counts[primitive] += 1

        examples_correct = 0
        examples_total = 0

        for ex in train:
            pred = apply_primitive(processor, ex["input"], primitive, detected.get("parameters", {}))
            if pred == ex["output"]:
                examples_correct += 1
            examples_total += 1

        acc = examples_correct / examples_total if examples_total else 0.0
        task_results.append((task_path.stem, primitive, acc, examples_correct, examples_total))
        total_examples += examples_total
        total_correct += examples_correct

    overall_acc = total_correct / total_examples if total_examples else 0.0

    print("ARC Primitive Baseline Evaluation (training split)")
    print(f"Tasks evaluated: {len(task_results)}")
    print(f"Total examples:  {total_examples}")
    print(f"Total correct:   {total_correct}")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print("\nPrimitive frequency:")
    for prim, count in primitive_counts.most_common():
        print(f"  {prim:12s}: {count}")

    top_tasks = sorted(task_results, key=lambda x: x[2], reverse=True)[:10]
    print("\nTop 10 tasks by accuracy:")
    for tid, prim, acc, c, t in top_tasks:
        print(f"  {tid}: acc={acc:.2f} ({c}/{t}) primitive={prim}")


if __name__ == "__main__":
    evaluate()
