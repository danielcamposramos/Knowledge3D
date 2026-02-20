from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_hot_path_has_no_regex_or_eval_fallbacks() -> None:
    target_files = [
        "knowledge3d/knowledgeverse/trm_navigator.py",
        "benchmarks/arc_agi_2_adapter.py",
        "benchmarks/last_humanity_exam.py",
        "knowledge3d/knowledgeverse/specialist_router.py",
    ]
    forbidden = [
        "re.search(",
        "re.match(",
        "ast.parse(",
        "eval(",
    ]
    for rel_path in target_files:
        content = _read(rel_path)
        for pattern in forbidden:
            assert pattern not in content, f"sovereignty violation: {pattern} in {rel_path}"


def test_galaxy_manager_requires_ptx_query_by_default() -> None:
    content = _read("knowledge3d/knowledgeverse/galaxy_manager.py")
    assert 'self.require_ptx_query = _env_true("K3D_REQUIRE_PTX_QUERY", "true")' in content
    assert "return self._query_ptx_implementation(" in content


def test_arc_ops_blocks_cpu_fallbacks() -> None:
    content = _read("knowledge3d/cranium/ptx/arc_ops.py")
    assert 'raise RuntimeError("arc_ptx_unavailable")' in content
    assert 'raise RuntimeError("ptx_argsort_failed")' in content
    assert "cpu_passthrough" not in content


def test_full_ptx_runner_enforces_ptx_query() -> None:
    bench = _read("scripts/run_all_benchmarks.py")
    global_bench = _read("scripts/run_all_global_benchmarks.py")
    assert 'os.environ["K3D_REQUIRE_PTX_QUERY"] = "true"' in bench
    assert 'os.environ["K3D_REQUIRE_PTX_QUERY"] = "true"' in global_bench


def test_sender_runtime_gpu_enforcement_is_present() -> None:
    sender_files = [
        "benchmarks/math_sender.py",
        "benchmarks/arc_sender.py",
        "benchmarks/lhe_sender.py",
        "benchmarks/mmlu_sender.py",
    ]
    for rel_path in sender_files:
        content = _read(rel_path)
        assert "gpu_calls_this_command" in content or "assert_gpu_for_solved_command" in content
        assert "assert_gpu_for_solved_command" in content, f"{rel_path} must enforce runtime GPU usage"


def test_runner_runtime_gpu_enforcement_is_present() -> None:
    runner_files = [
        "scripts/run_all_benchmarks.py",
        "scripts/run_all_global_benchmarks.py",
    ]
    for rel_path in runner_files:
        content = _read(rel_path)
        assert "SovereigntyViolation" in content, f"{rel_path} missing sovereignty violation type"
        assert "--no-enforce-sovereignty" in content, f"{rel_path} missing debug override flag"
        assert '"runtime_usage"' in content and '"sovereignty"' in content, f"{rel_path} missing sovereignty summary"
