"""
Sovereign pipeline extension for math benchmarks.

Hot path stays sovereign (PTX + RPN); this layer handles math-specific I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline
from knowledge3d.training.math_benchmarks.math_proceduralizer import MathDatasetLoader, MathProceduralizer
from knowledge3d.training.math_benchmarks.math_output_adapter import MathOutputAdapter
from knowledge3d.training.math_benchmarks.benchmark_evaluator import MathBenchmarkEvaluator
from knowledge3d.training.math_benchmarks.word_problem_solver import WordProblemSolver
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


class SovereignMathPipeline(SovereignAIPipeline):
    """
    Extends SovereignAIPipeline for math benchmarks.
    """

    def __init__(self, datasets: Optional[List[str]] = None, difficulty_range: Optional[range] = None, **kwargs):
        super().__init__(**kwargs)
        self._math_loader = MathDatasetLoader(datasets=datasets or ["gsm8k", "math"], difficulty_filter=difficulty_range)
        self._math_adapter = MathOutputAdapter()
        self._math_evaluator = MathBenchmarkEvaluator()
        self._proceduralizer = MathProceduralizer()
        self._word_solver = WordProblemSolver()
        self._composer = SovereignComposer()
        # Math-specific RPN engine (sovereign GPU path)
        self._rpn_engine = ModularRPNEngine()

    def _execute_sovereign(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute problem through sovereign hot path using existing pipeline.
        """
        return self.solve_problem(problem)

    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        text = problem.get("problem", problem.get("question", ""))
        metadata = dict(problem.get("metadata", {}))
        metadata.setdefault("original_text", text)
        metadata.setdefault("solution_rpn", problem.get("solution_rpn", metadata.get("solution_rpn", "")))

        result: Dict[str, Any] = {"stack": [], "answer": None, "success": False, "metadata": metadata}

        # SOVEREIGN: Compose RPN from input using Galaxy
        rpn_expression = self._composer.compose(text) if text else ""
        if rpn_expression:
            try:
                answer = self._rpn_engine.evaluate(rpn_expression)
                result.update(
                    {
                        "stack": [answer],
                        "answer": answer,
                        "success": True,
                        "method": "composer",
                        "rpn_program": rpn_expression,
                    }
                )
                return result
            except Exception as exc:  # noqa: BLE001 - propagate trace in metadata
                result["error"] = str(exc)
                result["rpn_program"] = rpn_expression

        # Ground-truth RPN fallback if provided by dataset metadata
        solution_rpn = metadata.get("solution_rpn", "")
        if solution_rpn:
            try:
                answer = self._rpn_engine.evaluate(solution_rpn)
                result.update(
                    {
                        "stack": [answer],
                        "answer": answer,
                        "success": True,
                        "method": "solution_rpn",
                        "rpn_program": solution_rpn,
                    }
                )
                return result
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(str(exc))
                result["rpn_program"] = solution_rpn

        # Fallback to word problem solver (grammar rules → RPN)
        word_solution = self._word_solver.solve(text) if text else {}
        rpn_program = word_solution.get("rpn_program", "")
        if rpn_program:
            try:
                answer = self._rpn_engine.evaluate(rpn_program)
                result.update(
                    {
                        "stack": [answer],
                        "answer": answer,
                        "success": True,
                        "method": "word_solver",
                        "rpn_program": rpn_program,
                        "matched_rules": word_solution.get("matched_rules", []),
                    }
                )
                return result
            except Exception as exc:  # noqa: BLE001
                result.setdefault("errors", []).append(str(exc))
                result["rpn_program"] = rpn_program

        return result

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a math task through the sovereign hot path.

        Args:
            task: {"input": problem_rpn, "metadata": {...}}

        Returns:
            {"stack": [...], "answer": Any, "success": bool}
        """
        metadata = task.get("metadata", {}) or {}
        problem_rpn = task.get("input", [])
        original_text = metadata.get("original_text", "")
        rpn_expression = ""

        if original_text:
            rpn_expression = self._composer.compose(original_text) or ""
        if not rpn_expression and problem_rpn:
            rpn_expression = self._convert_to_engine_format(problem_rpn)

        if not rpn_expression:
            return {"stack": [], "answer": None, "success": False, "error": "No executable RPN expression available."}

        try:
            answer = self._rpn_engine.evaluate(rpn_expression)
            return {"stack": [answer], "answer": answer, "success": True, "rpn_program": rpn_expression}
        except Exception as exc:  # noqa: BLE001
            return {"stack": [], "answer": None, "success": False, "error": str(exc), "rpn_program": rpn_expression}

    def _convert_to_engine_format(self, tokens: List[str]) -> str:
        """
        Convert MathProceduralizer tokens to ModularRPNEngine format.

        Input:  ["PUSH 5.0", "PUSH 3.0", "HINT ADD", "SOLVE"]
        Output: "5.0 3.0 +"
        """
        rpn_parts: List[str] = []
        pending_ops: List[str] = []

        op_map = {
            "ADD": "+",
            "SUB": "-",
            "MUL": "*",
            "DIV": "/",
            "SQRT": "sqrt",
            "SQR": "dup *",
            "POW": "pow",
            "PCT": "100 /",
            "EQ": "",
        }

        for token in tokens:
            if token.startswith("PUSH "):
                value = token.split(" ", 1)[1]
                rpn_parts.append(value)
            elif token.startswith("HINT "):
                op = token.split(" ", 1)[1].strip().upper()
                mapped = op_map.get(op, "")
                if mapped:
                    pending_ops.append(mapped)
            elif token == "SOLVE":
                if pending_ops:
                    rpn_parts.extend(pending_ops)
                    pending_ops = []

        return " ".join(rpn_parts)

    def train_on_math(self, epochs: int = 1, log_interval: int = 100) -> Dict[str, Any]:
        all_metrics: List[Dict[str, Any]] = []

        # Attach session reporter if present on superclass
        reporter = getattr(self, "reporter", None)
        loader_stats = self._math_loader.get_stats()
        if reporter:
            reporter.set_math_benchmarks(
                datasets=loader_stats.get("datasets_loaded", []),
                total_problems=loader_stats.get("total_problems", 0),
                by_source=loader_stats.get("by_source", {}),
            )

        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0

            for i, problem in enumerate(self._math_loader):
                result = self._execute_sovereign(problem)
                self._math_adapter.record_result(problem["problem_id"], result.get("stack", []), problem["source"])
                eval_result = self._math_evaluator.evaluate(
                    problem["problem_id"],
                    result.get("answer"),
                    problem["answer"],
                    problem["source"],
                )
                if eval_result["correct"]:
                    epoch_correct += 1
                epoch_total += 1

                if (i + 1) % log_interval == 0:
                    acc = epoch_correct / epoch_total if epoch_total else 0.0
                    print(f"[Epoch {epoch+1}] Problem {i+1}: {acc:.2%} accuracy")

            metrics = self._math_evaluator.get_metrics()
            metrics["epoch"] = epoch + 1
            all_metrics.append(metrics)

            print(f"[Epoch {epoch+1}] Overall: {metrics['overall']['accuracy']:.2%}")
            for source, data in metrics["by_source"].items():
                print(f"  {source}: {data['accuracy']:.2%}")

            if reporter:
                reporter.log_epoch(epoch + 1, {"accuracy": metrics["overall"]["accuracy"]}, math_metrics=metrics)

        # Persist a simple session report
        report_dir = Path("/K3D/Knowledge3D.local/logs/math_benchmarks/")
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "epochs": epochs,
            "datasets": self._math_loader.get_stats(),
            "final_metrics": all_metrics[-1] if all_metrics else {},
            "history": all_metrics,
        }
        import json
        with open(report_dir / f"math_training_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report


def train_multimodal(arc_epochs: int = 0, math_epochs: int = 1, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Train on both ARC-AGI and math benchmarks sequentially.
    """
    output_dir = output_dir or Path("/K3D/Knowledge3D.local/logs/multimodal/")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SovereignMathPipeline(datasets=["gsm8k", "math", "omni_math"], difficulty_range=range(1, 8))

    results = {"arc_agi": None, "math": None}

    if arc_epochs > 0:
        print("=" * 60)
        print("PHASE 1: ARC-AGI Visual Reasoning")
        print("=" * 60)
        # results["arc_agi"] = pipeline.train(epochs=arc_epochs)

    print("=" * 60)
    print("PHASE 2: Math Reasoning Benchmarks")
    print("=" * 60)
    results["math"] = pipeline.train_on_math(epochs=math_epochs)

    if hasattr(pipeline, "reporter"):
        pipeline.reporter.finalize()

    report_path = output_dir / "multimodal_training_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return results
