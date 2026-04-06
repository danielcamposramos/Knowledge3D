from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
import scripts.run_headless_tablet_benchmarks as runner
from scripts.run_headless_tablet_benchmarks import run_tablet_benchmark_suite


class _BenchmarkDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        task = dict(payload.get("task") or {})
        task_type = str(task.get("surface_kind") or task.get("type", ""))
        if task_type == "GAME_2D":
            return {
                "status": "ok",
                "route": {"specialist": "visual", "galaxy_names": ["Drawing"], "domain": "visual"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "grid",
                    "output_grid": task.get("expected_output"),
                    "answer_materialized": True,
                },
            }
        if task_type == "MATH":
            return {
                "status": "ok",
                "route": {"specialist": "math", "galaxy_names": ["Math"], "domain": "math"},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "numeric",
                    "answer_text": str(task.get("expected_answer") or ""),
                    "numeric_answer": 4.0,
                    "answer_materialized": True,
                },
            }
        if task_type == "QUESTION":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar"], "domain": task.get("domain_hint", "multi")},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "choice",
                    "answer_choice": str(task.get("expected_answer") or ""),
                    "answer_text": str(task.get("expected_answer") or ""),
                    "answer_materialized": True,
                },
            }
        if task_type == "GENERAL":
            return {
                "status": "ok",
                "route": {"specialist": "chat", "galaxy_names": ["Grammar"], "domain": task.get("domain_hint", "multi")},
                "task_result": {
                    "status": "ok",
                    "answer_kind": "text",
                    "answer_text": str(task.get("expected_answer") or ""),
                    "answer_materialized": True,
                },
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
    benchmark = ARCAGI2Benchmark(
        dataset_path=dataset_dir,
        tablet_boundary=boundary,
        knowledgeverse=_RunnerKnowledgeverse(dataset_dir),
    )

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
    benchmark = MathCompetitionBenchmark(
        dataset_path=dataset_dir,
        tablet_boundary=boundary,
        knowledgeverse=_RunnerKnowledgeverse(dataset_dir),
    )

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total"] == 1
    assert result["overall_accuracy"] == 1.0
    assert result["results_by_competition"]["AMC"]["results"][0]["method"] == "tablet_boundary"


def test_math_benchmark_loads_real_math_and_math_layouts(tmp_path: Path):
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

    benchmark = MathCompetitionBenchmark(
        dataset_path=tmp_path / "datasets",
        max_problems=10,
        knowledgeverse=_RunnerKnowledgeverse(tmp_path / "storage"),
    )

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
    benchmark = LastHumanityExamBenchmark(
        dataset_path=dataset_dir,
        tablet_boundary=boundary,
        knowledgeverse=_RunnerKnowledgeverse(dataset_dir),
    )

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_questions"] == 1
    assert result["accuracy"] == 1.0
    assert result["results"][0]["tablet_contract"]["action_type"] == "UPDATE_TABLET"


def test_math_benchmark_defaults_to_synthetic_guard_set(monkeypatch) -> None:
    monkeypatch.setattr(MathCompetitionBenchmark, "_has_present_dataset", lambda self, root: False)
    benchmark = MathCompetitionBenchmark(
        dataset_path=None,
        dataset_mode="synthetic",
        max_problems=20,
        knowledgeverse=_RunnerKnowledgeverse(""),
    )

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


class _RunnerKnowledgeverse:
    GPU_SPATIAL_TARGET_GALAXIES = ("Drawing",)

    def __init__(self, storage_root: str | Path):
        self.storage_root = Path(storage_root)
        self.logged_events: list[tuple[str, dict[str, object]]] = []

    def suspend_auto_sleep(self) -> None:
        return None

    def log_event(self, name: str, payload: dict[str, object]) -> None:
        self.logged_events.append((str(name), dict(payload)))

    def shutdown(self, *, persist: bool = True, profile: str = "service") -> dict[str, object]:
        return {"status": "completed", "persist": bool(persist), "profile": str(profile)}


def test_headless_tablet_runner_executes_arc_and_lhe(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "Knowledgeverse", _RunnerKnowledgeverse)
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

    args = argparse.Namespace(
        storage_root=str(tmp_path / "storage"),
        log_dir=str(tmp_path / "logs"),
        arc2_dataset_path=str(arc_dir),
        mmlu_dataset_path=None,
        gsm8k_dataset_path=None,
        lhe_dataset_path=str(lhe_dir),
        math_dataset_path=None,
        imo_dataset_path=None,
        arc2_count=1,
        arc3_count=0,
        mmlu_count=0,
        gsm8k_count=0,
        lhe_count=1,
        math_count=0,
        amc_aime_count=0,
        omni_math_count=0,
        imo_count=0,
        use_enriched=True,
        shutdown_timeout_s=0.1,
        output=None,
    )

    result = run_tablet_benchmark_suite(args, command_handler=_BenchmarkDaemon())

    assert result["summary"]["mode"] == "headless_tablet_wine_session"
    assert result["summary"]["benchmarks"]["arc2"]["accuracy"] == 1.0
    assert result["summary"]["benchmarks"]["lhe"]["accuracy"] == 1.0
    assert Path(result["summary"]["execution_artifacts"]["summary"]).exists()


def test_headless_tablet_runner_can_skip_unselected_benchmarks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "Knowledgeverse", _RunnerKnowledgeverse)
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

    args = argparse.Namespace(
        storage_root=str(tmp_path / "storage"),
        log_dir=str(tmp_path / "logs"),
        arc2_dataset_path=str(arc_dir),
        mmlu_dataset_path=None,
        gsm8k_dataset_path=None,
        lhe_dataset_path=None,
        math_dataset_path=None,
        imo_dataset_path=None,
        arc2_count=1,
        arc3_count=0,
        mmlu_count=0,
        gsm8k_count=0,
        lhe_count=0,
        math_count=0,
        amc_aime_count=0,
        omni_math_count=0,
        imo_count=0,
        use_enriched=True,
        shutdown_timeout_s=0.1,
        output=None,
    )

    result = run_tablet_benchmark_suite(args, command_handler=_BenchmarkDaemon())

    assert result["summary"]["benchmarks"]["arc2"]["accuracy"] == 1.0
    assert result["summary"]["benchmarks"]["mmlu"]["status"] == "skipped"
    assert result["summary"]["benchmarks"]["gsm8k"]["status"] == "skipped"
    assert result["summary"]["benchmarks"]["lhe"]["status"] == "skipped"


def test_runner_writes_execution_artifacts_before_shutdown_timeout(tmp_path: Path, monkeypatch) -> None:
    class _ProfileAwareKnowledgeverse:
        def __init__(self, storage_root: str | Path):
            self.storage_root = Path(storage_root)
            self.shutdown_calls: list[tuple[bool, str]] = []

        def suspend_auto_sleep(self) -> None:
            return None

        def shutdown(self, *, persist: bool = True, profile: str = "service") -> dict[str, object]:
            self.shutdown_calls.append((bool(persist), str(profile)))
            return {"status": "fast_exit", "persist": bool(persist), "profile": str(profile)}

    kv_instances: list[_ProfileAwareKnowledgeverse] = []

    def _factory(storage_root: str | Path) -> _ProfileAwareKnowledgeverse:
        kv = _ProfileAwareKnowledgeverse(storage_root)
        kv_instances.append(kv)
        return kv

    monkeypatch.setattr(runner, "Knowledgeverse", _factory)

    args = argparse.Namespace(
        storage_root=str(tmp_path / "storage"),
        log_dir=str(tmp_path / "logs"),
        arc2_dataset_path=None,
        mmlu_dataset_path=None,
        gsm8k_dataset_path=None,
        lhe_dataset_path=None,
        math_dataset_path=None,
        imo_dataset_path=None,
        arc2_count=0,
        arc3_count=0,
        mmlu_count=0,
        gsm8k_count=0,
        lhe_count=0,
        math_count=0,
        amc_aime_count=0,
        omni_math_count=0,
        imo_count=0,
        use_enriched=True,
        shutdown_timeout_s=0.01,
        output=None,
    )

    result = runner.run_tablet_benchmark_suite(args, command_handler=_BenchmarkDaemon())

    assert (Path(args.log_dir) / "summary.execution.json").exists()
    assert (Path(args.log_dir) / "full_results.execution.json").exists()
    assert kv_instances[0].shutdown_calls == [(False, "benchmark")]
    assert result["summary"]["sleep_consolidation"]["status"] == "fast_exit"
    assert result["summary"]["sleep_consolidation"]["profile"] == "benchmark"


def test_runner_uses_non_persistent_shutdown_when_supported(tmp_path: Path, monkeypatch) -> None:
    class _PersistAwareKnowledgeverse:
        def __init__(self, storage_root: str | Path):
            self.storage_root = Path(storage_root)
            self.shutdown_calls: list[tuple[bool, str]] = []

        def suspend_auto_sleep(self) -> None:
            return None

        def shutdown(self, *, persist: bool = True, profile: str = "service") -> dict[str, object]:
            self.shutdown_calls.append((bool(persist), str(profile)))
            return {"status": "completed", "persist": bool(persist), "profile": str(profile)}

    kv_instances: list[_PersistAwareKnowledgeverse] = []

    def _factory(storage_root: str | Path) -> _PersistAwareKnowledgeverse:
        kv = _PersistAwareKnowledgeverse(storage_root)
        kv_instances.append(kv)
        return kv

    monkeypatch.setattr(runner, "Knowledgeverse", _factory)

    args = argparse.Namespace(
        storage_root=str(tmp_path / "storage"),
        log_dir=str(tmp_path / "logs"),
        arc2_dataset_path=None,
        mmlu_dataset_path=None,
        gsm8k_dataset_path=None,
        lhe_dataset_path=None,
        math_dataset_path=None,
        imo_dataset_path=None,
        arc2_count=0,
        arc3_count=0,
        mmlu_count=0,
        gsm8k_count=0,
        lhe_count=0,
        math_count=0,
        amc_aime_count=0,
        omni_math_count=0,
        imo_count=0,
        use_enriched=True,
        shutdown_timeout_s=0.1,
        output=None,
    )

    result = runner.run_tablet_benchmark_suite(args, command_handler=_BenchmarkDaemon())

    assert kv_instances
    assert kv_instances[0].shutdown_calls == [(False, "benchmark")]
    assert result["summary"]["sleep_consolidation"]["persist"] is False
    assert result["summary"]["sleep_consolidation"]["profile"] == "benchmark"


def test_runner_preserves_execution_artifacts_on_legacy_shutdown_timeout(tmp_path: Path, monkeypatch) -> None:
    class _SlowLegacyKnowledgeverse:
        def __init__(self, storage_root: str | Path):
            self.storage_root = Path(storage_root)

        def suspend_auto_sleep(self) -> None:
            return None

        def shutdown(self) -> dict[str, object]:
            time.sleep(0.2)
            return {"status": "completed"}

    monkeypatch.setattr(runner, "Knowledgeverse", _SlowLegacyKnowledgeverse)

    args = argparse.Namespace(
        storage_root=str(tmp_path / "storage"),
        log_dir=str(tmp_path / "logs"),
        arc2_dataset_path=None,
        mmlu_dataset_path=None,
        gsm8k_dataset_path=None,
        lhe_dataset_path=None,
        math_dataset_path=None,
        imo_dataset_path=None,
        arc2_count=0,
        arc3_count=0,
        mmlu_count=0,
        gsm8k_count=0,
        lhe_count=0,
        math_count=0,
        amc_aime_count=0,
        omni_math_count=0,
        imo_count=0,
        use_enriched=True,
        shutdown_timeout_s=0.01,
        output=None,
    )

    result = runner.run_tablet_benchmark_suite(args, command_handler=_BenchmarkDaemon())

    assert (Path(args.log_dir) / "summary.execution.json").exists()
    assert (Path(args.log_dir) / "full_results.execution.json").exists()
    assert result["summary"]["sleep_consolidation"]["status"] == "timed_out"
