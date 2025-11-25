"""Run semantic parser → compiler → executor on a provided grid and instruction.

Usage:
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
        scripts/run_semantic_instruction.py \
        --instruction "Move the red object to the bottom-right corner" \
        --grid '[ [2,0,0], [0,0,0], [0,0,0] ]'

    # or load grid from JSON file
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
        scripts/run_semantic_instruction.py \
        --instruction "Fill the rectangle with red" \
        --grid-file input_grid.json
"""

from __future__ import annotations

import argparse
import json
from typing import List

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic pipeline on a grid.")
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural language instruction to execute",
    )
    parser.add_argument(
        "--grid",
        help="Grid as JSON list of lists, e.g., '[[1,0],[0,0]]'",
    )
    parser.add_argument(
        "--grid-file",
        help="Path to JSON file containing grid (list of lists of ints)",
    )
    return parser.parse_args()


def load_grid(args: argparse.Namespace) -> List[List[int]]:
    if args.grid:
        return json.loads(args.grid)
    if args.grid_file:
        with open(args.grid_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Provide --grid or --grid-file")


def main() -> None:
    args = parse_args()
    grid = load_grid(args)

    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    print(f"Instruction: {args.instruction}")
    print(f"Input grid: {grid}")

    semantic = parser.parse(args.instruction)
    print(f"Semantic: {semantic}")

    rpn = compiler.compile(semantic)
    print(f"RPN: {rpn}")

    output = executor.execute(grid, rpn)
    print(f"Output grid: {output}")


if __name__ == "__main__":
    main()
