"""
Sovereign pipeline extension for math benchmarks.

Hot path stays sovereign (PTX + RPN); this layer handles math-specific I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline
from knowledge3d.training.math_benchmarks.math_proceduralizer import MathDatasetLoader, MathProceduralizer
from knowledge3d.training.math_benchmarks.math_output_adapter import MathOutputAdapter
from knowledge3d.training.math_benchmarks.benchmark_evaluator import MathBenchmarkEvaluator


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

    def _execute_sovereign(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute problem through sovereign hot path using existing pipeline.
        """
        return self.execute_task({"input": problem["problem_rpn"], "metadata": problem["metadata"]})

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

        return {"epochs": epochs, "final_metrics": all_metrics[-1] if all_metrics else {}, "history": all_metrics}


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
