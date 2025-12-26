from __future__ import annotations


class _EchoEngine:
    def evaluate(self, expression: str, *_args, **_kwargs):
        stack = []
        for tok in (expression or "").split():
            if tok in {"+", "-", "*", "/"}:
                if len(stack) < 2:
                    raise ValueError("stack underflow")
                b = float(stack.pop())
                a = float(stack.pop())
                if tok == "+":
                    stack.append(a + b)
                elif tok == "-":
                    stack.append(a - b)
                elif tok == "*":
                    stack.append(a * b)
                else:
                    stack.append(a / b)
                continue
            stack.append(float(tok))
        if not stack:
            return None
        return float(stack[-1])


def test_word_galaxy_tokenize_basic(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy

    wg = WordGalaxy(storage_path=tmp_path)
    tokens = wg.tokenize("Natalia sold 48 clips altogether.")
    assert [t.normalized for t in tokens[:4]] == ["natalia", "sold", "48", "clips"]
    assert tokens[0].category == "proper_noun"
    assert tokens[2].category == "number"
    assert tokens[2].value == 48.0


def test_word_galaxy_tokenize_thousands_separator(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy

    wg = WordGalaxy(storage_path=tmp_path)
    tokens = wg.tokenize("She saved $3,000 last week.")
    nums = [t for t in tokens if t.category == "number"]
    assert any(t.value == 3000.0 for t in nums)


def test_trm_galaxy_reader_natalia_clips_compose(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    understanding, _trace = reader.read_problem(text)
    assert understanding.is_complete()
    assert understanding.quantities[0]["value"] == 48.0
    rpn = reader.compose_rpn(understanding)
    assert rpn == "48 48 2 / +"


def test_trm_math_navigator_prefers_galaxy_reader(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    nav = TRMMathNavigator(
        rule_bank=[],
        math_galaxy=MATH_GALAXY,
        rpn_engine=_EchoEngine(),
        galaxy_reader=reader,
    )

    text = "Natalia sold 48 clips. She sold half as many in May. How many altogether?"
    result, meta = nav.solve(text)
    assert meta["rule_used"] == "galaxy_read"
    assert result == 72.0
    comp = meta.get("read_composition", {})
    assert isinstance(comp, dict)
    assert comp.get("template_used") in {"extract_operate_aggregate", "simple_apply", "distribute_and_sum"}


def test_galaxy_reading_division_expression(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    understanding, _ = reader.read_problem("24 / 4")
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "24 4 /"


def test_galaxy_reading_has_gave_left(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "John has 15 apples. He gave 3 to Mary. How many apples does he have left?"
    understanding, _ = reader.read_problem(text)
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "15 3 -"


def test_galaxy_reading_gave_amount_noun_to(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "John has 15 apples. He gave 3 apples to Mary. How many apples does he have left?"
    understanding, _ = reader.read_problem(text)
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "15 3 -"


def test_galaxy_reading_percent_of(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    understanding, _ = reader.read_problem("25 % of 80")
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "80 25 100 / *"


def test_galaxy_reading_count_of_each_term(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    understanding, _ = reader.read_problem("3 bags of 5 apples. How many apples are there?")
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "3 5 *"


def test_galaxy_reading_for_every_additional_ratio(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Jennifer has 40 cans. Mark has 50 cans. "
        "Jennifer purchased 6 additional for every 5 Mark bought. "
        "How many cans does Jennifer have?"
    )
    understanding, _ = reader.read_problem(text)
    assert understanding.is_complete()
    rpn = reader.compose_rpn(understanding)
    assert rpn == "40 50 6 * 5 / +"
    assert _EchoEngine().evaluate(rpn) == 100.0


def test_galaxy_reading_there_are_each_have_term(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    understanding, _ = reader.read_problem("There are 6 people going on a trip. They each have 5 bags.")
    assert understanding.is_complete()
    rpn = reader.compose_rpn(understanding)
    assert rpn == "6 5 *"
    assert _EchoEngine().evaluate(rpn) == 30.0


def test_galaxy_reading_there_are_each_contain_term(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    understanding, _ = reader.read_problem("There are 12 crates that each contain 150 oranges.")
    assert understanding.is_complete()
    rpn = reader.compose_rpn(understanding)
    assert rpn == "12 150 *"
    assert _EchoEngine().evaluate(rpn) == 1800.0


def test_galaxy_reading_monthly_savings_years_to_months(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Ever since she was a little girl, Sheila has dreamed of traveling the world. "
        "By last week, she had saved $3,000. "
        "She has decided to continue saving $276 per month, for 4 years. "
        "Today, Sheila’s family secretly added $7,000 into the piggy bank. "
        "At the end of 4 years, how much money will be in Sheila’s piggy bank?"
    )
    understanding, _ = reader.read_problem(text)
    assert understanding.is_complete()
    rpn = reader.compose_rpn(understanding)
    assert _EchoEngine().evaluate(rpn) == 23248.0


def test_galaxy_reading_sum_of_two_terms(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "2 bags of 5 apples and 3 bags of 4 oranges altogether."
    understanding, _ = reader.read_problem(text)
    assert understanding.is_complete()
    assert reader.compose_rpn(understanding) == "2 5 * 3 4 * +"


def test_galaxy_reading_pages_multistep_template(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Julie is reading a 120-page book. Yesterday she read 12 pages. "
        "Today she read twice as many pages as yesterday. "
        "If she reads half of the remaining pages tomorrow, how many pages will she read tomorrow?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    rpn = reader.compose_rpn(understanding)
    expected = (
        "120 STORE_A 12 STORE_B RECALL_B 2 * STORE_C "
        "RECALL_A RECALL_B - RECALL_C - STORE_D RECALL_D 2 /"
    )
    assert rpn.split() == expected.split()


def test_galaxy_reading_there_are_each_has(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "There are 3 bags. Each bag has 5 apples. How many apples total?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "3 5 *"


def test_galaxy_reading_gave_each_of(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "John has 20 apples. He gave 3 apples to each of 4 friends. How many apples does he have left?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "20 12 -"


def test_galaxy_reading_rate_per_unit_for(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "She earns 5 dollars per day for 7 days. How much does she earn in all?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "5 7 *"


def test_galaxy_reading_multiple_quantities_without_explicit_aggregation(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "John has 10 apples. Mary has 4 oranges. How many fruits total?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    # Must not fall back to "return first number" when multiple quantities exist.
    assert reader.compose_rpn(understanding) == "10 4 +"


def test_galaxy_reading_rate_duration_multiplies_multiple_factors(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "There are 6 birds. Each bird eats 12 beetles per day for 15 days. "
        "How many beetles do they eat in all?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "6 12 15 * *"


def test_galaxy_reading_minutes_to_hours_per_day(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "John wants to finish a show in 5 days. "
        "There are 20 episodes and they are each 30 minutes long. "
        "How many hours does he watch per day?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "20 30 * 60 / 5 /"


def test_galaxy_reading_fraction_rest_of_total(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "A craft store makes a third of its sales in the fabric section, "
        "a quarter of its sales in the jewelry section, and the rest in the stationery section. "
        "They made 36 sales today. How many sales were in the stationery section?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "36 36 3 / - 36 4 / -"


def test_galaxy_reading_percent_increase_after_add(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "The total number of cases was recorded as 2000 on a particular day. "
        "The number of cases increased by 500 on the second day. "
        "On the third day, the number of cases increased by 50%. "
        "What is the total number of cases?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "2000 500 + 50 100 / 1 + *"


def test_galaxy_reading_covid_cases_recoveries(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "After tests in California, the total number of Coronavirus cases was recorded as 2000 positive cases on a particular day. "
        "The number of cases increased by 500 on the second day, with 50 recoveries. "
        "On the third day, the total number of new cases spiked to 1500 with 200 recoveries. "
        "What's the total number of positive cases after the third day?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "2000 500 + 50 - 1500 + 200 -"


def test_galaxy_reading_weighted_average_cost(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Apple sold 100 iPhones at their New York store today for an average cost of $1000. "
        "They also sold 20 iPads for an average cost of $900 and 80 Apple TVs for an average cost of $200. "
        "What was the average cost across all products sold today?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert (
        reader.compose_rpn(understanding, trace=trace, problem_text=text)
        == "100 1000 * 20 900 * 80 200 * + + 100 20 80 + + /"
    )


def test_galaxy_reading_profit_markup_schedule(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "James creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. "
        "He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. "
        "How much profit does he make in 20 weeks?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert (
        reader.compose_rpn(understanding, trace=trace, problem_text=text)
        == "6 2.5 * 6 - 500 * 5 * 20 * 2000 -"
    )


def test_galaxy_reading_remaining_unit_cost_from_budget(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Ann's favorite store was having a summer clearance. For $75 she bought 5 pairs of shorts for $7 each and 2 pairs "
        "of shoes for $10 each. She also bought 4 tops, all at the same price. How much did each top cost?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "75 5 7 * 2 10 * + - 4 /"


def test_galaxy_reading_cost_dollar_each_terms_sum(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "At lunch, 5 friends order 5 burgers that cost $3 each and 4 fries that cost $2 each. "
        "How much is the total cost?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    rule_ids = {p.get("rule_id") for p in trace.get("patterns", [])}
    assert "galaxy_count_cost_dollar_each" in rule_ids
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "5 3 * 4 2 * +"


def test_galaxy_reading_hourly_wage_schedule_month(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Carrie works for $8 an hour and works 35 hours a week. It's been a month since she started. "
        "How much money has she earned?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "35 8 * 4 *"


def test_galaxy_reading_ratio_scale_for_every(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Jason is mixing a batch of black paint. He needs to add 2 grams of charcoal for every 30 ml of water. "
        "If he adds 900 ml of water, how much charcoal does he need?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "2 30 / 900 *"


def test_galaxy_reading_grams_to_kilograms_conversion(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "20 100 * 1000 /"


def test_galaxy_reading_fraction_part_total_accounts(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Gina has two bank accounts. Each account has a quarter of the balance in Betty's account. "
        "If Betty's account balance is $3,456, what is the total balance in Gina's accounts?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "3456 4 / 2 *"


def test_galaxy_reading_inverse_fraction_chain_jelly_beans(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "There are some jelly beans in a jar. Three fourths of the jelly beans are red, "
        "and one quarter of the red jelly beans are coconut flavored. "
        "If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "750 16 * 3 /"


def test_galaxy_reading_each_eats_chain_rate_duration(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. "
        "If there are 6 jaguars in a forest, how many beetles are eaten per day?"
    )
    understanding, trace = reader.read_problem(text)
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "12 3 * 5 * 6 *"


def test_galaxy_reading_count_that_each_have_and_with_quantity(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Jessa needs to make cupcakes for 3 fourth-grade classes that each have 30 students and a P.E. class with 50 students. "
        "How many cupcakes does she need to make?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    rule_ids = {p.get("rule_id") for p in trace.get("patterns", [])}
    assert "galaxy_count_that_each_have" in rule_ids
    assert "galaxy_with_quantity_noun" in rule_ids
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "3 30 * 50 +"


def test_galaxy_reading_daily_time_schedule_minutes_week_month(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Carolyn practices the piano for 20 minutes a day and the violin for three times as long. "
        "If she practice six days a week, how many minutes does she spend practicing in a month with four weeks?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    rule_ids = {p.get("rule_id") for p in trace.get("patterns", [])}
    assert "galaxy_minutes_a_day" in rule_ids
    assert "galaxy_times_as_long" in rule_ids
    assert "galaxy_days_a_week_no_prep" in rule_ids
    assert "galaxy_with_weeks" in rule_ids
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "20 3 * 20 + 6 * 4 *"


def test_galaxy_reading_percent_chain_of_noun(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "In the school's library, there are 2300 different books. 80% of all the books are in English, "
        "but only 60% of these books were published recently. How many books is that?"
    )
    understanding, trace = reader.read_problem(text)
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "2300 80 100 / * 60 100 / *"


def test_galaxy_reading_boys_girls_remaining(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "There were 18 students on the trip. Eight were boys. How many were girls?"
    understanding, trace = reader.read_problem(text)
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "18 8 -"


def test_galaxy_reading_times_more_multiplier(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Riku has 25 times more stickers than Kristoff. "
        "If Kristoff has 85 stickers, how many stickers does Riku have?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "85 25 1 + *"


def test_galaxy_reading_shared_among(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "12 cookies were shared among 3 friends. How many cookies did each friend get?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "12 3 /"


def test_galaxy_reading_total_minus_sum_others_missing_contributor(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Kimberley, Ela, and Houston all are sent to collect firewood by their grammy. "
        "Kimberley collects ten pounds of firewood, and Houston collects 12 pounds of firewood. "
        "If the three of them managed to collect a total of 35 pounds of firewood, how much firewood did Ela collect?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "35 10 12 + -"


def test_galaxy_reading_quit_and_new_got_in(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Leticia, Nina, and Rosalie have a total of 25 people on their dance team. "
        "If 8 people quit, but 13 new people got in, how many people are there now on the team?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "25 8 - 13 +"


def test_galaxy_reading_linear_growth_rate_every_year(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Haley grows at the rate of 3 inches every year. "
        "If she is currently 20 inches tall, what will be her height after 10 years?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "20 3 10 * +"


def test_galaxy_reading_cost_difference_each_item_cost(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Caleb bought 10 cartons of ice cream and 4 cartons of frozen yoghurt. "
        "Each carton of ice cream cost $4 and each carton of frozen yoghurt cost $1. "
        "How much more did Caleb spend on ice cream than on frozen yoghurt?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "10 4 * 4 1 * -"


def test_galaxy_reading_pages_to_friends_twice_a_week_year(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "3 2 * 2 * 52 *"


def test_galaxy_reading_pizza_large_small_slices(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Albert buys 2 large pizzas and 2 small pizzas. "
        "A large pizza has 16 slices and a small pizza has 8 slices. "
        "If he eats it all, how many pieces does he eat?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding) == "2 16 * 2 8 * +"


def test_galaxy_reading_relative_chain_total(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Bella bought 11 snowflake stamps. "
        "She bought 9 more truck stamps than snowflake stamps, "
        "and 13 fewer rose stamps than truck stamps. "
        "How many stamps did Bella buy in total?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    # base=11, +9, -13 → total = 11 + (11+9) + (11+9-13) = 38
    expected = (
        "11 STORE_A RECALL_A STORE_B RECALL_B STORE_C "
        "RECALL_B 9 + STORE_B RECALL_C RECALL_B + STORE_C "
        "RECALL_B 13 - STORE_B RECALL_C RECALL_B + STORE_C "
        "RECALL_C"
    )
    assert rpn.split() == expected.split()


def test_galaxy_reading_packs_of_post_divide(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Trail mix comes in packs of 6. Roger has 13 members plus 3 coaches and 2 helpers. "
        "How many packs does he need?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "13 3 + 2 + 6 /"


def test_galaxy_reading_total_minus_terms(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "There are 5 houses on a street, and each of the first four houses has 3 gnomes in the garden. "
        "If there are a total of 20 gnomes on the street, how many gnomes are in the fifth house?"
    )
    understanding, trace = reader.read_problem(text)
    assert understanding.is_complete(), trace
    assert reader.compose_rpn(understanding, trace=trace, problem_text=text) == "20 4 3 * -"


def test_galaxy_reading_requires_two_quantities_sum(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "To make the route, it requires a total of 4 right-hand turns and requires 6 left-hand turns. "
        "How many turns total?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 10.0


def test_galaxy_reading_recorded_as_increased_by(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "The total number was recorded as 2000 and increased by 500. What is the total number now?"
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 2500.0


def test_galaxy_reading_total_of_for_count_unit_cost(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "He paid a total of $20,700 for 150 pieces. How much did each piece cost?"
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 138.0


def test_galaxy_reading_percent_rate_without_of(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = "The total bill was 140 dollars. The sales tax is 10% tax. How much is the tax?"
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 14.0


def test_galaxy_reading_total_minus_sum_others_house_expansion(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "There is a 5,200 sq. ft. house and a 7,300 sq. ft. house next to each other. "
        "The smaller house is being expanded. If the new total square footage of both houses is 16,000 sq. ft., "
        "how much is the smaller house being expanded by, in sq. ft.?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 3500.0


def test_galaxy_reading_reimburse_overcharge(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "On Friday, Remy paid Sally’s Woodworking LLC a total of $20,700 for 150 pieces of furniture. "
        "Later that evening, the company’s accountant discovered that a new intern in the sales department had overcharged Remy. "
        "If the cost of a piece of furniture is $134, how much money will Sally’s Woodworking LLC reimburse Remy?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 600.0


def test_galaxy_reading_gratuity_from_total_bill(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "A shady restaurant is charging customers gratuities after taxes without them being aware. "
        "If my total bill was $140, the sales tax in my city is 10%, I ordered a NY Striploin for $80, "
        "and I ordered a glass of wine for $10, how much did they charge me for gratuities?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 41.0


def test_galaxy_reading_wage_from_rooms_hours_rate(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "A building has four floors with ten rooms each. Legacy has to clean each room, and it takes her 6 hours to clean one room. "
        "If she earns $15 per hour of work, calculate the total amount of money she makes from cleaning all the floors in the building."
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert _EchoEngine().evaluate(rpn) == 3600.0


def test_galaxy_reading_bags_ratio_division(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Pam has 1200 apples. "
        "Each of her bags has as many apples as 3 of Gerald's bags. "
        "Gerald's bags have 40 apples each. "
        "How many bags does Pam have?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert rpn.strip() == "1200 3 40 * /"
    assert _EchoEngine().evaluate(rpn) == 10.0


def test_galaxy_reading_daily_minutes_week(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
    )

    text = (
        "Larry spends half an hour twice a day walking and playing with his dog. "
        "He also spends a fifth of an hour every day feeding his dog. "
        "How many minutes does Larry spend on his dog each week?"
    )
    understanding, trace = reader.read_problem(text)
    rpn = reader.compose_rpn(understanding, trace=trace, problem_text=text)
    assert rpn.strip() == "30 2 * 12 + 7 *"
    assert _EchoEngine().evaluate(rpn) == 504.0
