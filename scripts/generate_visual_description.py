"""Generate a simple visual description from a grid using grammar galaxy."""

from __future__ import annotations

import argparse
import json
import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Grid JSON, e.g. '[[2,0],[0,0]]'")
    args = ap.parse_args()

    arr = np.array(json.loads(args.grid), dtype=int)
    color_idx = dominant_color(arr)
    box = bbox(arr)
    if box is None:
        print("Empty grid")
        return
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
    out = executor.execute(rule.rpn_program, ctx)
    print(out)


if __name__ == "__main__":
    main()
