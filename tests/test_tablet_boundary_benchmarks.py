from __future__ import annotations

import json
from pathlib import Path

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from scripts.run_headless_tablet_benchmarks import run_tablet_benchmark_suite


class _BenchmarkDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        task = dict(payload.get("task") or {})
        task_type = str(task.get("type", ""))
        if task_type == "ARC_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "visual", "galaxy_names": ["Drawing"], "domain": "visual"},
                "task_result": {"status": "ok", "output_grid": task.get("expected_output")},
            }
        if task_type == "MATH_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "math", "galaxy_names": ["Math"], "domain": "math"},
                "task_result": {"status": "success", "result": task.get("expected_answer")},
            }
        if task_type == "LHE_TASK":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar"], "domain": task.get("domain_hint", "multi")},
                "task_result": {"status": "success", "response": str(task.get("expected_answer") or "")},
            }
        return {"status": "error", "error": "unsupported_task"}


def test_arc_benchmark_can_run_via_headless_tablet_boundary(tmp_path: Path):
    dataset_dir = tmp_path / "arc_eval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    (dataset_dir / "task_arc.json").write_text(json.dumps(task), encoding="utf-8")

    boundary = HeadlessTabletMPC(command_handler=_BenchmarkDaemon(), storage_root=tmp_path / "storage")
    benchmark = ARCAGI2Benchmark(dataset_path=dataset_dir, tablet_boundary=boundary)

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_tasks"] == 1
    assert result["accuracy"] == 1.0
    assert result["results"][0]["solver"] == "tablet_boundary"


def test_math_benchmark_can_run_via_headless_tablet_boundary(tmp_path: Path):
    dataset_dir = tmp_path / "math"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "amc_problems.json").write_text(
        json.dumps([{"id": "m1", "problem_text": "What is 2 + 2?", "answer": "4"}]),
        encoding="utf-8",
    )

    boundary = HeadlessTabletMPC(command_handler=_BenchmarkDaemon(), storage_root=tmp_path / "storage")
    benchmark = MathCompetitionBenchmark(dataset_path=dataset_dir, tablet_boundary=boundary)

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total"] == 1
    assert result["overall_accuracy"] == 1.0
    assert result["results_by_competition"]["AMC"]["results"][0]["method"] == "tablet_boundary"


def test_math_benchmark_loads_real_gsm8k_and_math_layouts(tmp_path: Path):
    gsm8k_path = tmp_path / "datasets" / "GSM8K" / "grade_school_math" / "data"
    gsm8k_path.mkdir(parents=True, exist_ok=True)
    (gsm8k_path / "test.jsonl").write_text(
        json.dumps(
            {
                "question": "If Alice has 3 apples and buys 2 more, how many apples does she have?",
                "answer": "Alice has 3 + 2 = 5 apples.\\n#### 5",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    math_path = tmp_path / "datasets" / "math" / "data"
    math_path.mkdir(parents=True, exist_ok=True)
    (math_path / "train.jsonl").write_text(
        json.dumps(
            {
                "problem": "Solve x + 1 = 3.",
                "type": "Algebra",
                "solution": "We get x=2, so the answer is \\boxed{2}.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark = MathCompetitionBenchmark(dataset_path=tmp_path / "datasets", max_problems=10)

    assert len(benchmark.problems) == 1
    competitions = {row["competition"] for row in benchmark.problems}
    assert "MATH:Algebra" in competitions
    answers = {str(row["answer"]) for row in benchmark.problems}
    assert "2" in answers
    assert benchmark.dataset_mode == "present"
    assert any("math/data/train.jsonl" in source for source in benchmark.dataset_sources)


def test_lhe_benchmark_can_run_via_headless_tablet_boundary(tmp_path: Path):
    dataset_dir = tmp_path / "lhe"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": [
            {
                "id": "q1",
                "domain": "logic",
                "question_text": "Pick B.",
                "options": ["A", "B", "C"],
                "correct_answer": "B",
            }
        ]
    }
    (dataset_dir / "last_humanity_exam.json").write_text(json.dumps(payload), encoding="utf-8")

    boundary = HeadlessTabletMPC(command_handler=_BenchmarkDaemon(), storage_root=tmp_path / "storage")
    benchmark = LastHumanityExamBenchmark(dataset_path=dataset_dir, tablet_boundary=boundary)

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_questions"] == 1
    assert result["accuracy"] == 1.0
    assert result["results"][0]["tablet_contract"]["action_type"] == "UPDATE_TABLET"


def test_math_benchmark_defaults_to_synthetic_guard_set() -> None:
    benchmark = MathCompetitionBenchmark(dataset_path=None, max_problems=20)

    assert benchmark.dataset_mode == "synthetic"
    assert benchmark.dataset_path == Path("")
    assert len(benchmark.problems) == 20
    assert all(problem.get("tier") for problem in benchmark.problems)
    assert all(
        not str(problem.get("competition", "")).upper().startswith("GSM8K")
        for problem in benchmark.problems
    )
    assert all(
        "derivative" not in str(problem.get("problem_text", "")).lower()
        for problem in benchmark.problems
    )


def test_headless_tablet_runner_executes_arc_math_and_lhe(tmp_path: Path):
    arc_dir = tmp_path / "arc_eval"
    arc_dir.mkdir(parents=True, exist_ok=True)
    (arc_dir / "task_arc.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        ),
        encoding="utf-8",
    )

    math_dir = tmp_path / "math"
    math_dir.mkdir(parents=True, exist_ok=True)
    (math_dir / "amc_problems.json").write_text(
        json.dumps([{"id": "m1", "problem_text": "What is 2 + 2?", "answer": "4"}]),
        encoding="utf-8",
    )

    lhe_dir = tmp_path / "lhe"
    lhe_dir.mkdir(parents=True, exist_ok=True)
    (lhe_dir / "last_humanity_exam.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "domain": "logic",
                        "question_text": "Pick B.",
                        "options": ["A", "B", "C"],
                        "correct_answer": "B",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class _Args:
        storage_root = str(tmp_path / "storage")
        arc_dataset_path = str(arc_dir)
        math_dataset_path = str(math_dir)
        lhe_dataset_path = str(lhe_dir)
        max_arc_tasks = 1
        max_math_problems = 1
        max_lhe_questions = 1
        use_enriched = True

    result = run_tablet_benchmark_suite(_Args(), command_handler=_BenchmarkDaemon())

    assert result["mode"] == "headless_tablet_boundary"
    assert result["benchmarks"]["arc_agi_2"]["accuracy"] == 1.0
    assert result["benchmarks"]["math_competitions"]["overall_accuracy"] == 1.0
    assert result["benchmarks"]["last_humanity_exam"]["accuracy"] == 1.0


def test_headless_tablet_runner_can_skip_unselected_benchmarks(tmp_path: Path):
    arc_dir = tmp_path / "arc_eval"
    arc_dir.mkdir(parents=True, exist_ok=True)
    (arc_dir / "task_arc.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        ),
        encoding="utf-8",
    )

    class _Args:
        storage_root = str(tmp_path / "storage")
        arc_dataset_path = str(arc_dir)
        math_dataset_path = None
        lhe_dataset_path = None
        max_arc_tasks = 1
        max_math_problems = 0
        max_lhe_questions = 0
        use_enriched = True

    result = run_tablet_benchmark_suite(_Args(), command_handler=_BenchmarkDaemon())

    assert result["benchmarks"]["arc_agi_2"]["accuracy"] == 1.0
    assert result["benchmarks"]["math_competitions"]["status"] == "skipped"
    assert result["benchmarks"]["last_humanity_exam"]["status"] == "skipped"
