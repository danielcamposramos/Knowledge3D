from __future__ import annotations


def test_gsm_a_times_b_times_c_requires_markers():
    from knowledge3d.training.math_benchmarks.math_templates import get_gsm8k_templates

    rules = {r.rule_id: r for r in get_gsm8k_templates()}
    rule = rules["gsm_a_times_b_times_c"]
    import re

    assert re.search(rule.pattern, "2 times 3 times 4")
    assert re.search(rule.pattern, "2 x 3 x 4")
    assert re.search(rule.pattern, "2×3×4")
    assert re.search(rule.pattern, "2 3 4") is None


def test_gsm_total_of_two_requires_and():
    from knowledge3d.training.math_benchmarks.math_templates import get_gsm8k_templates
    import re

    rules = {r.rule_id: r for r in get_gsm8k_templates()}
    rule = rules["gsm_total_of_two"]

    assert re.search(rule.pattern, "3 and 5 total")
    assert re.search(rule.pattern, "3 + 5 total")
    assert re.search(rule.pattern, "3 5 total") is None


def test_gsm_plus_does_not_match_bare_and():
    from knowledge3d.training.math_benchmarks.math_templates import get_gsm8k_templates
    import re

    rules = {r.rule_id: r for r in get_gsm8k_templates()}
    rule = rules["gsm_plus"]
    assert re.search(rule.pattern, "3 plus 5") is not None
    assert re.search(rule.pattern, "3 + 5") is not None
    assert re.search(rule.pattern, "3 and 5") is None


def test_runner_validate_answer_integer_constraints():
    from scripts.run_sovereign_math_benchmarks import SovereignBenchmarkRunner

    runner = SovereignBenchmarkRunner(use_trm_navigator=False)
    assert runner._validate_answer(2.0, "How many items total?", "gsm8k") is True
    assert runner._validate_answer(2.5, "How many items total?", "gsm8k") is False
    assert runner._validate_answer(-1.0, "How many items total?", "gsm8k") is False


def test_composite_matcher_handles_multiple_gives_sentences():
    from scripts.run_sovereign_math_benchmarks import SovereignBenchmarkRunner

    runner = SovereignBenchmarkRunner(use_trm_navigator=False)
    problem = {
        "problem": "John has 15 apples. He gives 3 to Mary. Then he gives 5 to Tom. How many apples does he have left?",
        "answer": 7,
        "source": "gsm8k",
    }
    result, solver, trace = runner.solve_problem_with_meta(problem)
    assert result == 7.0
    assert solver in {"word", "template", "grammar", "trm"}
    if solver == "word":
        assert trace.get("rule_used") == "composite_matcher"
