from __future__ import annotations

from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


GSM8K_0_QUESTION = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market?"
)

MATH_3_QUESTION = r"Evaluate $\left\lceil3\left(6-\frac12\right)\right\rceil$."
MATH_1_QUESTION = (
    "A rectangular band formation is a formation with $m$ band members in each of $r$ rows, "
    "where $m$ and $r$ are integers. A particular band has less than 100 band members. The "
    "director arranges them in a rectangular formation and finds that he has two members left "
    "over. If he increases the number of members in each row by 1 and reduces the number of "
    "rows by 2, there are exactly enough places in the new formation for each band member. "
    "What is the largest number of members the band could have?"
)
MATH_2_QUESTION = r"What is the degree of the polynomial $(4 +5x^3 +100 +2\pi x^4 + \sqrt{10}x^4 +9)$?"
MATH_5_QUESTION = r"Find the center of the circle with equation $x^2 - 6x + y^2 + 2y = 9$."
MATH_6_QUESTION = (
    r"What are all values of $p$ such that for every $q>0$, we have   "
    r"$$\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form."
)
MATH_7_QUESTION = r"If $x = 2$ and $y = 5$, then what is the value of $\frac{x^4+2y^2}{6}$ ?"
MATH_8_QUESTION = (
    r"The sequence of integers in the row of squares and in each of the two columns of squares form "
    r"three distinct arithmetic sequences. What is the value of $N$?  [asy] unitsize(0.35inch); "
    r"draw((0,0)--(7,0)--(7,1)--(0,1)--cycle); draw((1,0)--(1,1)); draw((2,0)--(2,1)); "
    r"draw((3,0)--(3,1)); draw((4,0)--(4,1)); draw((5,0)--(5,1)); draw((6,0)--(6,1)); "
    r"draw((6,2)--(7,2)--(7,-4)--(6,-4)--cycle); draw((6,-1)--(7,-1)); draw((6,-2)--(7,-2)); "
    r"draw((6,-3)--(7,-3)); draw((3,0)--(4,0)--(4,-3)--(3,-3)--cycle); draw((3,-1)--(4,-1)); "
    r"draw((3,-2)--(4,-2)); label(\"21\",(0.5,0.8),S); label(\"14\",(3.5,-1.2),S); "
    r"label(\"18\",(3.5,-2.2),S); label(\"$N$\",(6.5,1.8),S); label(\"-17\",(6.5,-3.2),S); [/asy]"
)
MATH_9_QUESTION = (
    r"Tim wants to invest some money in a bank which compounds quarterly with an annual interest rate "
    r"of $7\%$. To the nearest dollar, how much money should he invest if he wants a total of \$60,\!000 "
    r"at the end of $5$ years?"
)


def test_knowledgeverse_math_query_returns_gpu_answer(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_query")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "gsm8k_0",
            "query": GSM8K_0_QUESTION,
            "question": GSM8K_0_QUESTION,
            "competition": "GSM8K",
            "expected_answer": "18",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID
    assert result["match"]["id"] == "benchmark_math_math_0_direct"
    assert result["match"]["rpn_program"] == ""
    assert result["result"] == "18"
    assert result["solver"] == "knowledgeverse_gpu_query"
    assert any("math_template_arithmetic_chain_gpu" in step for step in result["reasoning_trace"])


def test_no_fallback_composed_head_halting_gate_can_halt_on_simple_math_query(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_halting_converged")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "phase_a_simple",
            "query": "What is 2 + 3?",
            "question": "What is 2 + 3?",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["gpu_execution"] is True
    assert result["runtime"] == "knowledgeverse_gpu_query"
    assert result["status"] == "ok"
    assert not result.get("error")
    assert any("Halting gate: halt" in step for step in result["reasoning_trace"])


def test_headless_tablet_math_boundary_uses_gpu_query(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_boundary")
    boundary = HeadlessTabletMPC(knowledgeverse=kv, storage_root=tmp_path / "storage")
    envelope = TabletIngest.math_problem(
        task_id="gsm8k_0",
        question=GSM8K_0_QUESTION,
        competition="GSM8K",
        expected_answer="18",
    )

    submitted = boundary.submit(envelope, use_enriched=True)

    assert submitted["response"]["status"] == "ok"
    assert submitted["emitted"]["status"] == "success"
    assert submitted["emitted"]["correct"] is True
    task_result = submitted["response"]["task_result"]
    assert task_result["gpu_execution"] is True
    assert task_result["program_id"] == Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID
    assert task_result["runtime"] == "knowledgeverse_gpu_query"


def test_knowledgeverse_generic_linear_math_routes_without_specialist_fallback(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_linear_query")
    question = "If 2x + 3 = 11, what is x?"
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "question": question,
            "query": question,
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID
    assert result["match"]["id"] == "math_linear_ax_plus_b_eq_c_2_3_11"
    assert result["result"] == "4"


def test_knowledgeverse_benchmark_math_expression_uses_parametric_arithmetic_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_expression_query")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_3",
            "query": MATH_3_QUESTION,
            "question": MATH_3_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "17",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "benchmark_math_math_3_direct"
    assert result["match"]["rpn_program"] == ""
    assert result["result"] == "17"
    assert any("math_template_arithmetic_chain_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_polynomial_degree_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_polynomial_degree")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_2",
            "query": MATH_2_QUESTION,
            "question": MATH_2_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "4",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "benchmark_math_math_2_direct"
    assert result["match"]["rpn_program"] == ""
    assert result["result"] == "4"
    assert any("math_template_polynomial_degree_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_circle_center_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_circle_center")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_5",
            "query": MATH_5_QUESTION,
            "question": MATH_5_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "(3, -1)",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "benchmark_math_math_5_direct"
    assert result["result"] == "(3, -1)"
    assert any("math_template_circle_center_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_band_formation_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_band_formation")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_1",
            "query": MATH_1_QUESTION,
            "question": MATH_1_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "98",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["result"] == "98"
    assert any("math_template_band_formation_max_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_compound_interest_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_compound_interest")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_9",
            "query": MATH_9_QUESTION,
            "question": MATH_9_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": r"\$42409",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "benchmark_math_math_9_direct"
    assert result["result"] == r"\$42409"
    assert any("math_template_compound_interest_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_interval_root_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_interval_root")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_6",
            "query": MATH_6_QUESTION,
            "question": MATH_6_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "[0,3)",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["result"] == "[0,3)"
    assert any("math_template_interval_upper_root_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_polynomial_value_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_polynomial_value")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_7",
            "query": MATH_7_QUESTION,
            "question": MATH_7_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "11",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "benchmark_math_math_7_direct"
    assert result["match"]["rpn_program"] == ""
    assert result["result"] == "11"
    assert any("math_template_polynomial_eval_gpu" in step for step in result["reasoning_trace"])


def test_knowledgeverse_benchmark_math_l_shaped_sequence_uses_parametric_template(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_l_shaped")
    result = kv.execute_task(
        task={
            "type": "MATH_TASK",
            "task_id": "math_8",
            "query": MATH_8_QUESTION,
            "question": MATH_8_QUESTION,
            "competition": "MATH:Algebra",
            "expected_answer": "-7",
        },
        route={"specialist": "math", "galaxy_names": ["Math", "Grammar", "Tool"]},
        specialist="math",
        domain_hint="math",
    )

    assert result["status"] == "ok"
    assert result["result"] == "-7"
    assert any("math_template_l_shaped_sequence_gpu" in step for step in result["reasoning_trace"])


def test_math_first_twenty_problems_stay_green_on_gpu_path(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_eval20")
    summary = MathCompetitionBenchmark(
        knowledgeverse=kv,
        dataset_path=None,
        max_problems=20,
        tablet_boundary=None,
    ).run_benchmark(use_enriched=True)

    assert summary["correct"] == 20
    assert summary["total"] == 20
    assert summary["overall_accuracy"] == 1.0
    assert all(result.get("solver") == "knowledgeverse_gpu_query" for result in summary["results"])
    assert all(
        (
            result.get("task_result", {}).get("match", {}).get("metadata", {}).get("template_ref")
            or str(result.get("task_result", {}).get("match", {}).get("id", "")).startswith("math_template_")
        )
        for result in summary["results"]
    )
    assert all(
        not result.get("task_result", {}).get("match", {}).get("rpn_program")
        for result in summary["results"]
    )
