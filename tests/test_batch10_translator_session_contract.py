from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_headless_tablet_benchmarks.py"
ARC = REPO_ROOT / "benchmarks" / "arc_agi_2.py"
MATH = REPO_ROOT / "benchmarks" / "math_competitions.py"
MMLU = REPO_ROOT / "benchmarks" / "mmlu.py"
LHE = REPO_ROOT / "benchmarks" / "last_humanity_exam.py"


def test_headless_runner_drops_threadpool_preload_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source
    assert "as_completed" not in source


def test_benchmark_modules_use_tape_sessions_not_per_item_submit() -> None:
    for path in (ARC, MATH, MMLU, LHE):
        source = path.read_text(encoding="utf-8")
        assert "run_tape_session(" in source
