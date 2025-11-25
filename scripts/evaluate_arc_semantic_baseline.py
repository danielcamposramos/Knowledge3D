"""Evaluate ARC baseline using the semantic layer with richer inference and fallbacks."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.arc_agi.semantic_primitives import COLOR_SEMANTICS
from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.reasoning.arc_dataset import ensure_arc_dataset, _iter_task_files, _load_task

COLOR_BY_VALUE = {meta["value"]: name for name, meta in COLOR_SEMANTICS.items() if "value" in meta}


def infer_instruction_from_grids(input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
    """
    Infer a coarse instruction from a single input/output pair (heuristic).
    """
    inp = np.array(input_grid)
    out = np.array(output_grid)

    if inp.shape != out.shape:
        return "unknown"

    # Rotation check
    for k in [1, 2, 3]:
        if np.array_equal(out, np.rot90(inp, k=k)):
            angle = k * 90
            return f"Rotate the pattern {angle} degrees"

    # Flip check
    if np.array_equal(out, np.fliplr(inp)):
        return "Flip the pattern horizontally"
    if np.array_equal(out, np.flipud(inp)):
        return "Flip the pattern vertically"

    # Translation: compare centroids of non-zero mask
    inp_mask = inp != 0
    out_mask = out != 0
    if inp_mask.sum() == out_mask.sum() and inp_mask.sum() > 0:
        inp_coords = np.argwhere(inp_mask)
        out_coords = np.argwhere(out_mask)
        delta = out_coords.mean(axis=0) - inp_coords.mean(axis=0)
        dy, dx = int(round(delta[0])), int(round(delta[1]))
        if dy != 0 or dx != 0:
            if abs(dx) >= abs(dy) and dx != 0:
                dir_x = "right" if dx > 0 else "left"
                steps = abs(dx)
                return f"Move the object {dir_x} by {steps}"
            if dy != 0:
                dir_y = "down" if dy > 0 else "up"
                steps = abs(dy)
                return f"Move the object {dir_y} by {steps}"

    # Recolor check
    inp_colors = set(int(c) for c in np.unique(inp) if c != 0)
    out_colors = set(int(c) for c in np.unique(out) if c != 0)
    if len(inp_colors) == 1 and len(out_colors) == 1 and inp_colors != out_colors:
        src_val = next(iter(inp_colors))
        dst_val = next(iter(out_colors))
        src = COLOR_BY_VALUE.get(src_val)
        dst = COLOR_BY_VALUE.get(dst_val)
        if src and dst:
            return f"Change {src} to {dst}"

    # Fill: more filled cells in output
    if (out != 0).sum() > (inp != 0).sum():
        new_colors = out_colors - inp_colors
        color_val = next(iter(new_colors), None) or next(iter(out_colors), None)
        color_name = COLOR_BY_VALUE.get(color_val) if color_val is not None else None
        if color_name:
            return f"Fill the empty region with {color_name}"
        return "Fill the empty region"

    return "unknown"


def build_instruction_from_detected(detected: Dict[str, Any]) -> str | None:
    """Map detected primitive to a textual instruction the parser can handle."""
    prim = detected.get("primitive", "UNKNOWN")
    params = detected.get("parameters", {})

    if prim.startswith("ROTATE_"):
        try:
            angle = int(prim.split("_")[1])
        except Exception:
            angle = params.get("angle", 90)
        return f"Rotate the pattern {angle} degrees"

    if prim == "FLIP_H":
        return "Flip the pattern horizontally"
    if prim == "FLIP_V":
        return "Flip the pattern vertically"

    if prim == "TRANSLATE":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        if abs(dx) >= abs(dy) and dx != 0:
            dir_x = "right" if dx > 0 else "left"
            steps = abs(dx)
            return f"Move the object {dir_x} by {steps}"
        if dy != 0:
            dir_y = "down" if dy > 0 else "up"
            steps = abs(dy)
            return f"Move the object {dir_y} by {steps}"

    if prim == "RECOLOR":
        src = COLOR_BY_VALUE.get(params.get("src"))
        dst = COLOR_BY_VALUE.get(params.get("dst"))
        if src and dst:
            return f"Change {src} to {dst}"

    if prim.endswith("_RECOLOR"):
        # Try to verbalize primary transform, then recolor.
        base = prim.replace("_RECOLOR", "")
        base_instr = build_instruction_from_detected({"primitive": base, "parameters": params})
        src = COLOR_BY_VALUE.get(params.get("src"))
        dst = COLOR_BY_VALUE.get(params.get("dst"))
        if base_instr and src and dst:
            return f"{base_instr} then change {src} to {dst}"

    if prim == "ROTATE_TRANSLATE":
        angle, dx, dy = params.get("angle", 90), params.get("dx", 0), params.get("dy", 0)
        dir_part = ""
        if abs(dx) >= abs(dy) and dx != 0:
            dir_part = "right" if dx > 0 else "left"
        elif dy != 0:
            dir_part = "down" if dy > 0 else "up"
        if dir_part:
            return f"Rotate the pattern {angle} degrees then move the object {dir_part}"
        return f"Rotate the pattern {angle} degrees"

    return None


def apply_detected(processor: ARCGridProcessor, grid: List[List[int]], detected: Dict[str, Any]) -> List[List[int]]:
    """Apply a detected primitive using grid processor helpers."""
    prim = detected.get("primitive", "UNKNOWN")
    params = detected.get("parameters", {})
    # Reuse logic from primitive baseline.
    from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor as GP

    if prim.endswith("_RECOLOR"):
        base = prim.replace("_RECOLOR", "")
        transformed = apply_detected(processor, grid, {"primitive": base, "parameters": params})
        src = params.get("src")
        dst = params.get("dst")
        arr = np.array(transformed, dtype=int)
        arr[arr == src] = dst
        return arr.tolist()

    if prim.startswith("ROTATE_TRANSLATE"):
        angle = params.get("angle", 0)
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        rotated = GP._apply_rotation(grid, angle)  # type: ignore[attr-defined]
        return GP._apply_translation(rotated, dx, dy)  # type: ignore[attr-defined]

    if prim.startswith("ROTATE_"):
        try:
            angle = int(prim.split("_")[1])
        except Exception:
            angle = params.get("angle", 0)
        return GP._apply_rotation(grid, angle)  # type: ignore[attr-defined]

    if prim == "FLIP_H":
        return GP._apply_flip_horizontal(grid)  # type: ignore[attr-defined]

    if prim == "FLIP_V":
        return GP._apply_flip_vertical(grid)  # type: ignore[attr-defined]

    if prim == "TRANSLATE":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        return GP._apply_translation(grid, dx, dy)  # type: ignore[attr-defined]

    if prim == "RECOLOR":
        src = params.get("src")
        dst = params.get("dst")
        arr = np.array(grid, dtype=int)
        arr[arr == src] = dst
        return arr.tolist()

    return grid


def candidate_set(train: List[Dict[str, Any]], processor: ARCGridProcessor) -> List[Dict[str, Any]]:
    """Generate candidate instructions or direct transforms from all training examples."""
    candidates: List[Dict[str, Any]] = []
    seen_detected = set()
    detected_keys: Counter = Counter()  # type: ignore[var-annotated]
    detected_lookup: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, Any]] = {}

    # Collect detected primitives across all training examples.
    for ex in train:
        detected = processor.detect_spatial_primitive(ex["input"], ex["output"])
        prim = detected.get("primitive")
        if not prim or prim == "UNKNOWN":
            continue
        key = (prim, tuple(sorted(detected.get("parameters", {}).items())))
        if key in seen_detected:
            continue
        seen_detected.add(key)
        detected_keys[key] += 1
        detected_lookup[key] = detected
        candidates.append({"type": "detected", "detected": detected})
        instr = build_instruction_from_detected(detected)
        if instr:
            candidates.append({"type": "instruction", "instruction": instr})

    # Add majority-detected candidate (mode) if available.
    if detected_keys:
        top_key, _ = detected_keys.most_common(1)[0]
        detected = detected_lookup[top_key]
        candidates.append({"type": "detected", "detected": detected})
        instr = build_instruction_from_detected(detected)
        if instr:
            candidates.append({"type": "instruction", "instruction": instr})

    # Heuristic instruction from first example as fallback.
    if train:
        heuristic_instr = infer_instruction_from_grids(train[0]["input"], train[0]["output"])
        if heuristic_instr != "unknown":
            candidates.append({"type": "instruction", "instruction": heuristic_instr})

    # Brute-force simple transforms as last resort.
    for angle in (90, 180, 270):
        candidates.append({"type": "instruction", "instruction": f"Rotate the pattern {angle} degrees"})
    candidates.append({"type": "instruction", "instruction": "Flip the pattern horizontally"})
    candidates.append({"type": "instruction", "instruction": "Flip the pattern vertically"})

    # Identity candidate (no-op) to allow similarity fallback.
    candidates.append({"type": "instruction", "instruction": "unknown"})

    return candidates


def evaluate():
    """Evaluate ARC training split using semantic parsing + execution with fallbacks."""
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))

    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()
    processor = ARCGridProcessor(matryoshka_dim=128, embedder_type="procedural")

    task_results = []
    instruction_counts = Counter()
    total_examples = 0
    total_correct = 0

    for task_path in task_files:
        task = _load_task(task_path)
        train = task.get("train", [])
        if len(train) < 1:
            continue

        candidates = candidate_set(train, processor)
        task_correct = 0
        task_total = 0
        task_instruction = "unknown"

        for ex in train:
            best_score = -1.0
            best_label = "unknown"
            best_exact = False
            for cand in candidates:
                try:
                    if cand["type"] == "instruction":
                        semantic = parser.parse(cand["instruction"])
                        rpn = compiler.compile(semantic)
                        pred = executor.execute(ex["input"], rpn)
                        label = cand["instruction"]
                    elif cand["type"] == "detected":
                        pred = apply_detected(processor, ex["input"], cand["detected"])
                        label = cand["detected"]["primitive"]
                    else:
                        continue
                    target = ex["output"]
                    exact = pred == target
                    if exact:
                        score = 1.0
                    else:
                        pa = np.array(pred, dtype=int)
                        ta = np.array(target, dtype=int)
                        if pa.shape != ta.shape:
                            score = 0.0
                        else:
                            score = float((pa == ta).sum()) / float(pa.size)
                    if score > best_score or (score == best_score and exact and not best_exact):
                        best_score = score
                        best_label = label
                        best_exact = exact
                except Exception:
                    continue
            if best_exact:
                task_correct += 1
            task_total += 1
            task_instruction = best_label if task_instruction == "unknown" else task_instruction

        acc = task_correct / task_total if task_total else 0.0
        task_results.append((task_path.stem, task_instruction, acc, task_correct, task_total))
        total_examples += task_total
        total_correct += task_correct
        instruction_counts[task_instruction] += 1

    overall_acc = total_correct / total_examples if total_examples else 0.0

    print("ARC Semantic Baseline Evaluation (training split)")
    print(f"Tasks evaluated: {len(task_results)}")
    print(f"Total examples:  {total_examples}")
    print(f"Total correct:   {total_correct}")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print("\nInstruction frequency:")
    for instr, count in instruction_counts.most_common(10):
        print(f"  {instr:40s}: {count}")

    top_tasks = sorted(task_results, key=lambda x: x[2], reverse=True)[:10]
    print("\nTop 10 tasks by accuracy:")
    for tid, instr, acc, c, t in top_tasks:
        print(f"  {tid}: acc={acc:.2f} ({c}/{t}) instruction={instr}")


if __name__ == "__main__":
    evaluate()
