"""
Standard Output Adapter — converts sovereign results to ARC Prize format.

Internal: RPN programs, TernaryVector embeddings, procedural grids
External: JSON with 2D grid arrays (standard benchmark format)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class StandardOutputAdapter:
    """
    Converts sovereign K3D results to standard ARC evaluation format.
    """

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self.results: Dict[str, Dict[str, List[List[int]]]] = {}

    def record_attempt(self, task_id: str, attempt_number: int, grid: Sequence[Sequence[int]]) -> None:
        if task_id not in self.results:
            self.results[task_id] = {}

        attempt_key = f"attempt_{attempt_number}"
        standard_grid = [[int(cell) for cell in row] for row in grid]
        self.results[task_id][attempt_key] = standard_grid

    def record_from_rpn_result(self, task_id: str, attempt_number: int, rpn_grid: Any) -> None:
        if hasattr(rpn_grid, "tolist"):
            grid = rpn_grid.tolist()
        elif hasattr(rpn_grid, "to_python"):
            grid = rpn_grid.to_python()
        else:
            grid = rpn_grid
        self.record_attempt(task_id, attempt_number, grid)

    def to_submission_json(self) -> str:
        return json.dumps(self.results, indent=2)

    def save_submission(self, output_path: Optional[Path] = None, session_id: Optional[str] = None) -> str:
        if output_path is None:
            ts = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"submissions/arc_submission_{ts}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.to_submission_json())
        print(f"[SUBMISSION] Saved to {output_path}")
        return str(output_path)

    def get_stats(self) -> Dict[str, Any]:
        total_tasks = len(self.results)
        tasks_with_2 = sum(1 for r in self.results.values() if len(r) >= 2)
        return {
            "total_tasks": total_tasks,
            "tasks_with_2_attempts": tasks_with_2,
            "coverage": tasks_with_2 / total_tasks if total_tasks > 0 else 0.0,
        }


class ARCEvaluationBridge:
    """
    Bridge to evaluate submissions against ground truth.
    """

    def evaluate_submission(self, submission_path: Path, ground_truth_path: Path) -> Dict[str, Any]:
        with open(submission_path) as f:
            submission = json.load(f)
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        correct = 0
        total = 0
        results_by_task: Dict[str, Dict[str, Any]] = {}

        for task_id, expected in ground_truth.items():
            if task_id not in submission:
                results_by_task[task_id] = {"status": "missing", "correct": False}
                total += 1
                continue

            attempts = submission[task_id]
            expected_output = expected.get("test", [{}])[0].get("output", [])
            task_correct = False
            for attempt_key in ["attempt_1", "attempt_2"]:
                if attempt_key not in attempts:
                    continue
                attempt_output = attempts[attempt_key]
                if self._grids_match(attempt_output, expected_output):
                    task_correct = True
                    break

            results_by_task[task_id] = {"status": "correct" if task_correct else "incorrect", "correct": task_correct}
            if task_correct:
                correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "results_by_task": results_by_task,
        }

    def _grids_match(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        if len(grid1) != len(grid2):
            return False
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
            if row1 != row2:
                return False
        return True


__all__ = ["StandardOutputAdapter", "ARCEvaluationBridge"]
