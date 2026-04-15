from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_all_benchmarks.py"
BENCHMARK_NAMES = {
    "ARCAGI2Benchmark",
    "MathCompetitionBenchmark",
    "LastHumanityExamBenchmark",
    "MMLUBenchmark",
}


def _benchmark_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in BENCHMARK_NAMES:
            calls.append(node)
    return calls


def test_benchmark_launcher_injects_tablet_boundary_into_every_benchmark_ctor() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = _benchmark_calls(tree)
    assert len(calls) >= 16
    for call in calls:
        assert any(keyword.arg == "tablet_boundary" for keyword in call.keywords)


def test_benchmark_launcher_contains_tablet_contract_assertions_for_all_families() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '_assert_tablet_boundary_contract("arc_empty", arc_empty)' in source
    assert '_assert_tablet_boundary_contract("arc_enriched", arc_enriched)' in source
    assert '_assert_tablet_boundary_contract("math_empty", math_empty)' in source
    assert '_assert_tablet_boundary_contract("math_enriched", math_enriched)' in source
    assert '_assert_tablet_boundary_contract("lhe_empty", lhe_empty)' in source
    assert '_assert_tablet_boundary_contract("lhe_enriched", lhe_enriched)' in source
    assert '_assert_tablet_boundary_contract("mmlu_empty", mmlu_empty)' in source
    assert '_assert_tablet_boundary_contract("mmlu_enriched", mmlu_enriched)' in source
