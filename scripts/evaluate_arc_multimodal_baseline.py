"""Evaluate ARC baseline using multimodal grammar (spatial + math + drawing)."""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from knowledge3d.training.arc_agi.multimodal_parser import MultimodalSemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.reasoning.arc_dataset import ensure_arc_dataset, _iter_task_files, _load_task
from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.arc_agi.semantic_primitives import COLOR_SEMANTICS

COLOR_BY_VALUE = {meta["value"]: name for name, meta in COLOR_SEMANTICS.items() if "value" in meta}


def infer_instruction_multimodal(input_grid: List[List[int]], output_grid: List[List[int]], processor: ARCGridProcessor) -> str:
    """Heuristic multimodal instruction inference."""
    inp = np.array(input_grid)
    out = np.array(output_grid)

    # Dimension change
    if inp.shape != out.shape:
        if out.shape[0] == 2 * inp.shape[0] and out.shape[1] == inp.shape[1]:
            return "Tile the pattern 2x vertically"
        if out.shape[1] == 2 * inp.shape[1] and out.shape[0] == inp.shape[0]:
            return "Tile the pattern 2x horizontally"
        return "unknown"

    # Rotations
    for k in [1, 2, 3]:
        if np.array_equal(out, np.rot90(inp, k=k)):
            angle = k * 90
            return f"Rotate the pattern {angle} degrees"

    # Flips
    if np.array_equal(out, np.fliplr(inp)):
        return "Flip the pattern horizontally"
    if np.array_equal(out, np.flipud(inp)):
        return "Flip the pattern vertically"

    # Translation or recolor via primitive detector
    detected = processor.detect_spatial_primitive(input_grid, output_grid)
    if detected.get("primitive") and detected["primitive"] != "UNKNOWN":
        prim = detected["primitive"]
        params = detected.get("parameters", {})
        if prim.startswith("ROTATE_"):
            try:
                ang = int(prim.split("_")[1])
                return f"Rotate the pattern {ang} degrees"
            except Exception:
                pass
        if prim == "TRANSLATE":
            dx, dy = params.get("dx", 0), params.get("dy", 0)
            if abs(dx) >= abs(dy) and dx != 0:
                dir_x = "right" if dx > 0 else "left"
                return f"Move the object {dir_x} by {abs(dx)}"
            if dy != 0:
                dir_y = "down" if dy > 0 else "up"
                return f"Move the object {dir_y} by {abs(dy)}"
        if prim == "RECOLOR":
            src = params.get("src")
            dst = params.get("dst")
            src_name = COLOR_BY_VALUE.get(src, str(src))
            dst_name = COLOR_BY_VALUE.get(dst, str(dst))
            return f"Change {src_name} to {dst_name}"

    # Conditional fills (row+col even/odd)
    even_mask = ((np.arange(inp.shape[0])[:, None] + np.arange(inp.shape[1])) % 2 == 0)
    odd_mask = ~even_mask
    change_mask = inp != out
    if np.array_equal(change_mask, even_mask):
        return "Fill cells where row + col is even"
    if np.array_equal(change_mask, odd_mask):
        return "Fill cells where row + col is odd"

    # Region fills
    filled = (out != 0) & (inp == 0)
    if filled.any() and filled.sum() > 0:
        cy, cx = np.argwhere(filled).mean(axis=0)
        h, w = inp.shape
        # pick predominant fill color
        new_vals, counts = np.unique(out[filled], return_counts=True)
        color_val = int(new_vals[counts.argmax()]) if len(new_vals) else 1
        color_name = COLOR_BY_VALUE.get(color_val, "blue")
        if cy < h / 3 and cx < w / 3:
            return f"Fill top-left region with {color_name}"
        if cy < h / 3 and cx > 2 * w / 3:
            return f"Fill top-right region with {color_name}"
        if cy > 2 * h / 3 and cx < w / 3:
            return f"Fill bottom-left region with {color_name}"
        if cy > 2 * h / 3 and cx > 2 * w / 3:
            return f"Fill bottom-right region with {color_name}"
        return f"Fill the empty region with {color_name}"

    # Movement to corners/center by centroid shift
    inp_colors = set(np.unique(inp)) - {0}
    out_colors = set(np.unique(out)) - {0}
    if inp_colors == out_colors and inp_colors:
        color = next(iter(inp_colors))
        src_mask = inp == color
        dst_mask = out == color
        if src_mask.sum() == dst_mask.sum() and src_mask.sum() > 0:
            sy, sx = np.argwhere(src_mask).mean(axis=0)
            dy, dx = np.argwhere(dst_mask).mean(axis=0)
            h, w = inp.shape
            target = None
            if dy == 0 and dx == 0:
                target = "top-left corner"
            elif dy == 0 and dx == w - 1:
                target = "top-right corner"
            elif dy == h - 1 and dx == 0:
                target = "bottom-left corner"
            elif dy == h - 1 and dx == w - 1:
                target = "bottom-right corner"
            elif dy == h // 2 and dx == w // 2:
                target = "center"
            if target:
                color_name = COLOR_BY_VALUE.get(int(color), "object")
                return f"Move the {color_name} object to the {target}"

    # Color replacement if only color changes
    if inp.shape == out.shape and np.count_nonzero(inp != out) > 0:
        for src_val in inp_colors:
            for dst_val in out_colors:
                test = inp.copy()
                test[test == src_val] = dst_val
                if np.array_equal(test, out):
                    src_name = COLOR_BY_VALUE.get(int(src_val), str(src_val))
                    dst_name = COLOR_BY_VALUE.get(int(dst_val), str(dst_val))
                    return f"Change {src_name} to {dst_name}"

    return "unknown"


def evaluate():
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))

    parser = MultimodalSemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()
    processor = ARCGridProcessor(matryoshka_dim=128, embedder_type="procedural")

    task_results = []
    domain_counts = Counter()
    instruction_counts = Counter()
    total_examples = 0
    total_correct = 0

    for task_path in task_files:
        task = _load_task(task_path)
        train = task.get("train", [])
        if len(train) < 1:
            continue

        ref = train[0]
        instruction = infer_instruction_multimodal(ref["input"], ref["output"], processor)
        instruction_counts[instruction] += 1

        if instruction == "unknown":
            continue

        semantic = parser.parse(instruction, debug=False)
        domain = semantic.get("domain", "unknown")
        domain_counts[domain] += 1
        if domain == "unknown":
            continue

        examples_correct = 0
        examples_total = 0

        for ex in train:
            try:
                rpn = compiler.compile(semantic)
                pred = executor.execute(ex["input"], rpn)
                if pred == ex["output"]:
                    examples_correct += 1
                examples_total += 1
            except Exception:
                examples_total += 1

        acc = examples_correct / examples_total if examples_total else 0.0
        task_results.append((task_path.stem, instruction, domain, acc, examples_correct, examples_total))
        total_examples += examples_total
        total_correct += examples_correct

    overall_acc = total_correct / total_examples if total_examples else 0.0

    print("ARC Multimodal Baseline Evaluation (training split)")
    print(f"Tasks evaluated: {len(task_results)}")
    print(f"Total examples:  {total_examples}")
    print(f"Total correct:   {total_correct}")
    print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")

    print("\nDomain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:12s}: {count}")

    print("\nInstruction frequency (top 15):")
    for instr, count in instruction_counts.most_common(15):
        print(f"  {instr:50s}: {count}")

    top_tasks = sorted(task_results, key=lambda x: x[3], reverse=True)[:10]
    print("\nTop 10 tasks by accuracy:")
    for tid, instr, domain, acc, c, t in top_tasks:
        print(f"  {tid}: {acc:.2f} ({c}/{t}) domain={domain} instr={instr[:40]}")


if __name__ == "__main__":
    evaluate()
