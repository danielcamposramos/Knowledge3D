from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist


class _MiniGalaxy:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []


class _MiniGalaxyManager:
    def __init__(self) -> None:
        self._galaxies: dict[str, _MiniGalaxy] = {}

    def get_galaxy(self, name: str) -> _MiniGalaxy:
        if name not in self._galaxies:
            self._galaxies[name] = _MiniGalaxy()
        return self._galaxies[name]

    def add_entry(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        self.get_galaxy(galaxy_name).entries.append(dict(entry))

    def query(
        self,
        query_text: str,
        specialist: str = "math",
        top_k: int = 10,
        galaxies=None,
        preferred_pattern_type: str | None = None,
    ):
        target = list(galaxies or self._galaxies.keys())
        rows: list[dict[str, Any]] = []
        for name in target:
            for entry in self.get_galaxy(name).entries:
                rows.append({"entry": entry, "score": 1.0, "galaxy": name})
        return rows[: max(1, int(top_k))]


class _MiniKV:
    def __init__(self) -> None:
        self.galaxy_manager = _MiniGalaxyManager()
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        self.events.append((event_type, dict(event_data)))


def test_math_specialist_linear_equation_composes_rpn() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "11 3 - 2 /":
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "If 2x + 3 = 11, what is x?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["rpn_program"] == "11 3 - 2 /"
    assert out["coefficients"] == {"a": 2.0, "b": 3.0, "c": 11.0}


def test_math_specialist_backward_equation_composes_rpn() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        normalized = " ".join(program.strip().split())
        if normalized in {"11 3 - 2 /", "3 11 - -2 /"}:
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "If 11 = 2x + 3, what is x?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["pattern_type"] == "linear_equation"


def test_math_specialist_arithmetic_addition_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "7 5 +":
            return 12.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "What is 7 + 5?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 12.0
    assert out["rpn_program"] == "7 5 +"
    assert out["pattern_type"] == "arithmetic_add"


def test_math_specialist_ratio_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "8 2 /":
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "What is the ratio 8:2?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["rpn_program"] == "8 2 /"
    assert out["pattern_type"] == "ratio"


def test_math_specialist_proportion_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "3 4 * 2 /":
            return 6.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "Solve 2/3 = 4/x for x."}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 6.0
    assert out["rpn_program"] == "3 4 * 2 /"
    assert out["pattern_type"] == "proportion"


def test_math_specialist_word_problem_sequential_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "16 3 - 4 - 2 *":
            return 18.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Janet's ducks lay 16 eggs per day. "
                "She eats 3 every morning and bakes muffins with 4. "
                "She sells the remainder for 2 dollars each. "
                "How much does she make every day?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 18.0
    assert out["rpn_program"] == "16 3 - 4 - 2 *"
    assert out["pattern_type"] == "word_problem_sequential"
    assert out["template_id"] is None
    assert out["template_mode"] == "grammar_composition"
    assert out["grammar_chain"] == [
        "gsm_sequential_computation",
        "gsm_consume_from_total",
        "gsm_consume_from_total",
        "gsm_rate_application",
        "gsm_answer_final_stack",
    ]
    assert out["number_refs"] == ["num_16", "num_3", "num_4", "num_2"]
    assert out["word_refs"] == ["word_sixteen", "word_three", "word_four", "word_two"]


def test_math_specialist_word_problem_spelled_numbers_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "16 3 - 4 - 2 *":
            return 18.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Janet's ducks lay sixteen eggs per day. "
                "She eats three every morning and bakes muffins with four. "
                "She sells the remainder for two dollars each. "
                "How much does she make every day?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 18.0
    assert out["rpn_program"] == "16 3 - 4 - 2 *"
    assert out["pattern_type"] == "word_problem_sequential"
    assert out["number_refs"] == ["num_16", "num_3", "num_4", "num_2"]
    assert out["word_refs"] == ["word_sixteen", "word_three", "word_four", "word_two"]


def test_math_specialist_reference_expression_half_that_much() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "2 1 +":
            return 3.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 3.0
    assert out["rpn_program"] == "2 1 +"
    assert out["number_refs"] == ["num_2", "num_1"]
    assert out["word_refs"] == ["word_two", "word_one"]


def test_math_specialist_count_frequency_rate_chain() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "3 3 * 60 *":
            return 540.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "James decides to run 3 sprints 3 times a week. "
                "He runs 60 meters each sprint. "
                "How many total meters does he run a week?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 540.0
    assert out["rpn_program"] == "3 3 * 60 *"
    assert out["grammar_chain"] == [
        "gsm_rate_application",
        "gsm_answer_final_stack",
    ] or out["grammar_chain"] == [
        "gsm_sequential_computation",
        "gsm_rate_application",
        "gsm_answer_final_stack",
    ]


def test_math_specialist_profit_percentage_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "80000 1 150 100 / + * 80000 - 50000 -":
            return 70000.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. "
                "This increased the value of the house by 150%. How much profit did he make?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 70000.0
    assert out["rpn_program"] == "80000 1 150 100 / + * 80000 - 50000 -"


def test_math_specialist_final_meal_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "20 3 * 15 - 25 -":
            return 20.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. "
                "In the morning, she gives her flock of chickens 15 cups of feed. "
                "In the afternoon, she gives her chickens another 25 cups of feed. "
                "How many cups of feed does she need to give her chickens in the final meal of the day "
                "if the size of Wendi's flock is 20 chickens?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 20.0
    assert out["rpn_program"] == "20 3 * 15 - 25 -"


def test_math_specialist_discount_pair_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "16 2 / 5 * 16 2 / 5 60 * 100 / * +":
            return 64.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Kylar went to the store to buy glasses. One glass costs $5, "
                "but every second glass costs only 60% of the price. "
                "Kylar wants to buy 16 glasses. How much does he need to pay for them?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 64.0
    assert out["rpn_program"] == "16 2 / 5 * 16 2 / 5 60 * 100 / * +"


def test_math_specialist_comparison_chain_with_twice_as_many() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "20 4 * 20 4 * 2 * + 20 +":
            return 260.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. "
                "How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 260.0
    assert out["rpn_program"] == "20 4 * 20 4 * 2 * + 20 +"


def test_math_specialist_restart_download_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "200 40 * 100 / 2 / 20 + 200 2 / +":
            return 160.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, "
                "but 40% of the way through the download, Windows forces a restart to install updates, "
                "which takes 20 minutes. Then Carla has to restart the download from the beginning. "
                "How long does it take to download the file?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 160.0
    assert out["rpn_program"] == "200 40 * 100 / 2 / 20 + 200 2 / +"


def test_math_specialist_overtime_split_rate_composition() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "40 10 * 45 40 - 10 1.2 * * +":
            return 460.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process(
        {
            "question": (
                "Eliza's rate per hour for the first 40 hours she works each week is $10. "
                "She also receives an overtime pay of 1.2 times her regular hourly rate. "
                "If Eliza worked for 45 hours this week, how much are her earnings for this week?"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == 460.0
    assert out["rpn_program"] == "40 10 * 45 40 - 10 1.2 * * +"


def test_math_specialist_quantified_interval_inequality_foundation() -> None:
    kv = _MiniKV()
    specialist = MathSpecialist(knowledgeverse=kv, evaluator=lambda _program: None)
    out = specialist.process(
        {
            "question": (
                "What are all values of $p$ such that for every $q>0$, we have "
                "$$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ "
                "Express your answer in interval notation in decimal form."
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == "[0,3)"
    assert out["pattern_type"] == "foundational_quantified_interval_inequality"
    assert out["grammar_chain"] == [
        "math_quantified_domain_guard",
        "math_rational_cancellation",
        "math_quadratic_interval",
        "math_interval_emit",
    ]


def test_math_specialist_asy_arithmetic_sequence_foundation() -> None:
    kv = _MiniKV()
    specialist = MathSpecialist(knowledgeverse=kv, evaluator=lambda _program: None)
    out = specialist.process(
        {
            "question": (
                "The sequence of integers in the row of squares and in each of the two columns of squares "
                "form three distinct arithmetic sequences. What is the value of $N$?\n\n"
                "[asy]\n"
                "unitsize(0.35inch);\n"
                "draw((0,0)--(7,0)--(7,1)--(0,1)--cycle);\n"
                "draw((1,0)--(1,1));\n"
                "draw((2,0)--(2,1));\n"
                "draw((3,0)--(3,1));\n"
                "draw((4,0)--(4,1));\n"
                "draw((5,0)--(5,1));\n"
                "draw((6,0)--(6,1));\n"
                "draw((6,2)--(7,2)--(7,-4)--(6,-4)--cycle);\n"
                "draw((6,-1)--(7,-1));\n"
                "draw((6,-2)--(7,-2));\n"
                "draw((6,-3)--(7,-3));\n"
                "draw((3,0)--(4,0)--(4,-3)--(3,-3)--cycle);\n"
                "draw((3,-1)--(4,-1));\n"
                "draw((3,-2)--(4,-2));\n"
                "label(\"21\",(0.5,0.8),S);\n"
                "label(\"14\",(3.5,-1.2),S);\n"
                "label(\"18\",(3.5,-2.2),S);\n"
                "label(\"$N$\",(6.5,1.8),S);\n"
                "label(\"-17\",(6.5,-3.2),S);\n"
                "[/asy]"
            )
        },
        use_enriched=True,
    )

    assert out["status"] == "success"
    assert out["result"] == -7.0
    assert out["pattern_type"] == "foundational_asy_arithmetic_sequence"
    assert out["grammar_chain"] == [
        "math_asy_label_extract",
        "math_arithmetic_column_step",
        "math_arithmetic_row_step",
        "math_cross_sequence_emit",
    ]
