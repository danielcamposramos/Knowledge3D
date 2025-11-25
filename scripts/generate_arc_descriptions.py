"""Generate simple visual descriptions for a subset of ARC training tasks."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from knowledge3d.training.reasoning.arc_dataset import ensure_arc_dataset, _iter_task_files, _load_task
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.grammar_executor import GrammarRPNExecutor


def dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    idx = int(np.argmax(counts))
    return int(vals[idx])


def bbox(grid: np.ndarray):
    ys, xs = np.nonzero(grid != 0)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def position_label(grid: np.ndarray, y0, y1, x0, x1) -> str:
    h, w = grid.shape
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    vert = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "center"
    horiz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    if vert == "center" and horiz == "center":
        return "at center"
    return f"at {vert}-{horiz}"


def describe_grid(grid):
    arr = np.array(grid, dtype=int)
    color_idx = dominant_color(arr)
    box = bbox(arr)
    if box is None:
        return "Empty grid"
    y0, y1, x0, x1 = box
    pos = position_label(arr, y0, y1, x0, x1)

    galaxy = GrammarGalaxy()
    rule = galaxy.get_rule("en_visual_description")
    ctx = {
        "subject": "Object",
        "is": "is",
        "color": str(color_idx),
        "shape": "patch",
        "position": pos,
    }
    executor = GrammarRPNExecutor()
    return executor.execute(rule.rpn_program, ctx)


def main():
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))[:10]
    for tp in task_files:
        task = _load_task(tp)
        first = task.get("train", [])[0]
        desc = describe_grid(first["input"])
        print(f"{tp.stem}: {desc}")


if __name__ == "__main__":
    main()
