from __future__ import annotations

from pathlib import Path


FORBIDDEN_BRIDGE_CALLS = (
    "launch_k3d_canonicalize",
    "launch_k3d_pattern_match",
    "launch_k3d_rule_apply",
    "launch_k3d_expr_build",
)

LIVE_EVALUATION_FILES = (
    Path("benchmarks/arc2_local_runner.py"),
    Path("benchmarks/arc_agi_2.py"),
    Path("benchmarks/arc_agi_2_adapter.py"),
    Path("benchmarks/arc_agi_3.py"),
    Path("benchmarks/arc3_sdk_agent.py"),
    Path("benchmarks/mmlu.py"),
    Path("benchmarks/last_humanity_exam.py"),
    Path("benchmarks/math_competitions.py"),
    Path("benchmarks/imo_bench.py"),
)

ALLOWED_TEST_FILES = (
    Path("tests/test_arc_r0_surface.py"),
    Path("tests/test_sovereign_cas_benchmark_simple.py"),
)


def test_live_evaluation_path_has_no_direct_cas_bridge_launches() -> None:
    for path in LIVE_EVALUATION_FILES:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_BRIDGE_CALLS:
            assert forbidden not in text, f"{forbidden} leaked into live evaluation path: {path}"


def test_direct_cas_bridge_launches_remain_test_only() -> None:
    found = []
    for path in ALLOWED_TEST_FILES:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_BRIDGE_CALLS:
            if forbidden in text:
                found.append((str(path), forbidden))
    assert found, "expected at least one direct CAS bridge launch in test-only coverage"
