from __future__ import annotations

from pathlib import Path

from benchmarks.imo_bench import IMOBenchmark


class _FakeKnowledgeverse:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.events: list[tuple[str, dict]] = []

    def execute_task(self, *, task, route=None, specialist=None, domain_hint=None, **kwargs):
        self.calls.append(
            {
                "task": task,
                "route": route,
                "specialist": specialist,
                "domain_hint": domain_hint,
                "kwargs": kwargs,
            }
        )
        return {
            "predicted_answer": task.get("expected_answer"),
            "gpu_execution": True,
            "program_id": "imo_test_program",
            "reasoning_trace": ["imo test trace"],
        }

    def log_event(self, event_type, payload):
        self.events.append((str(event_type), dict(payload)))


def test_imo_benchmark_loads_answerbench_csv_and_routes_math_task(tmp_path: Path) -> None:
    csv_path = tmp_path / "answerbench_v2.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Problem ID,Problem,Short Answer,Category,Subcategory,Source",
                "imo-a,What is 2+2?,4,Algebra,Operation,synthetic",
                "imo-b,What is 3+3?,6,Geometry,Other,synthetic",
                "imo-c,What is 5+5?,10,Number Theory,Other,synthetic",
            ]
        ),
        encoding="utf-8",
    )

    kv = _FakeKnowledgeverse()
    bench = IMOBenchmark(knowledgeverse=kv, dataset_path=csv_path, max_questions=3)
    result = bench.run_benchmark(progress_every=1)

    assert result["total_questions"] == 3
    assert result["correct"] == 3
    assert result["accuracy"] == 1.0
    assert result["synthetic_fallback"] is False
    assert all(call["task"]["type"] == "MATH_TASK" for call in kv.calls)
    assert all(call["task"]["competition"] == "IMO" for call in kv.calls)
    assert any(event == "math_problem_solved" for event, _ in kv.events)
