from __future__ import annotations


class _EchoEngine:
    def evaluate(self, expression: str, *_args, **_kwargs):
        stack: list[float] = []
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
        return float(stack[-1]) if stack else None


def _make_reader(tmp_path, *, thinking_budget: int = 0):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    return TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
        thinking_budget=thinking_budget,
    )


def test_percent_complement(tmp_path):
    reader = _make_reader(tmp_path)
    text = "There are 100 students. 80% are girls. How many are not girls?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=2)
    assert result == 20.0
    assert meta.get("template_used") == "percent_complement"


def test_multi_item_cost_sum_context(tmp_path):
    reader = _make_reader(tmp_path)
    text = "35 students bought a book for $10.50 each and a notebook for $7.50 each. How much did they spend altogether?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=2)
    assert result == 630.0
    assert meta.get("template_used") in {"distribute_and_sum", "extract_operate_aggregate"}


def test_percent_cut_remaining_after_product(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Mr. Wells has a garden of flowers with 50 rows. "
        "If each row has 400 flowers and Mr. Wells cuts 60% of the flowers, "
        "how many flowers are remaining in the garden?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 8000.0
    assert meta.get("template_used") in {"test_time_compute", "percent_complement", "extract_operate_aggregate"}


def test_relative_chain_candidate_generation(tmp_path):
    reader = _make_reader(tmp_path)
    text = "There are 140 goats. There are 40 more goats than sheep. How many sheep are there?"
    understanding, trace = reader.read_problem(text)
    candidates = reader._generate_relative_chain_candidates(
        problem_text=text,
        understanding=understanding,
        trace=trace,
        question_type=reader.classify_question(text),
        max_candidates=20,
    )
    assert "140 40 -" in candidates


def test_ttc_plausibility_rejects_percent_explosion(tmp_path):
    reader = _make_reader(tmp_path)
    text = "There are 100 students. 80% are girls. How many are not girls?"
    verdict = reader.verify_plausibility(text, 8000.0, "100 80 *")
    assert verdict.get("plausible") is False
    assert verdict.get("reason") in {"percent_result_exceeds_scale", "percent_result_exceeds_total"}


def test_ttc_plausibility_rejects_incomplete_multi_step(tmp_path):
    reader = _make_reader(tmp_path)
    text = "Tom had $100. He spent $25, then spent $40 more. How much money does he have left?"
    verdict = reader.verify_plausibility(text, 75.0, "100 25 -")
    assert verdict.get("plausible") is False
    assert verdict.get("reason") == "multi_step_incomplete"


def test_ttc_plausibility_rejects_percent_out_of_bounds(tmp_path):
    reader = _make_reader(tmp_path)
    text = "How much soda is left, expressed as a percentage of a bottle?"
    verdict = reader.verify_plausibility(text, 700.0, "7 100 *")
    assert verdict.get("plausible") is False
    assert verdict.get("reason") == "percent_out_of_bounds"


def test_ttc_plausibility_rejects_no_operation(tmp_path):
    reader = _make_reader(tmp_path)
    text = "There are 3 apples and 4 apples. How many apples are there in total?"
    verdict = reader.verify_plausibility(text, 3.0, "3")
    assert verdict.get("plausible") is False
    assert verdict.get("reason") == "no_operation"


def test_extract_combined_total_number_before_keyword(tmp_path):
    reader = _make_reader(tmp_path)
    text = "Bill and Phil are firehouse Dalmatians. If they have 59 spots combined, how many spots does Bill have?"
    assert reader._extract_combined_total(text) == 59.0


def test_extract_numbers_falls_back_to_number_words(tmp_path):
    reader = _make_reader(tmp_path)
    nums = reader.extract_numbers("Gina scored two goals and Tom scored three goals.")
    assert 2.0 in nums
    assert 3.0 in nums


def test_percent_increase_more_expensive(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "Alan bought a $2000 phone online. John bought it 2% more expensive in a local store. How much did John spend?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 2040.0
    assert meta.get("template_used") == "test_time_compute"


def test_percent_discount(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Travis wants to fly to Australia. The regular tickets cost about $2000. "
        "As Travis is a student, he will get a 30% discount on this price. How much does he need to pay?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1400.0
    assert meta.get("template_used") == "test_time_compute"


def test_nested_each_capacity_increase(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "There are 4 carriages in a train and each carriage has 25 seats. "
        "If each carriage could accommodate 10 more passengers, how many passengers would fill up 3 trains?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 420.0
    assert meta.get("template_used") == "test_time_compute"


def test_remaining_equal_cost_from_partial_known(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Five shirts together cost $85. Of the 5 shirts, there are 3 shirts that cost $15 each. "
        "If the remaining shirts are each equal in value, what is the cost, in dollars, of each remaining shirt?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20.0
    assert meta.get("template_used") == "test_time_compute"


def test_ratio_ticket_revenue_total_people(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Three times as many children as adults attend a concert on Saturday. "
        "An adult ticket costs $7 and a child's ticket costs $3. "
        "The theater collected a total of $6,000. How many total people attended?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1500.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_multi_step_full_composition(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "There were 50 more female adults than male adults, and children were twice the total number of adults. "
        "If there were 100 male adults, what was the total number of people at the reunion?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 750.0
    assert meta.get("template_used") == "test_time_compute"
    rpn = str(meta.get("rpn_program") or "")
    num_ops = rpn.count("+") + rpn.count("-") + rpn.count("*") + rpn.count("/")
    assert num_ops >= 3


def test_relative_chain_multiplication_series(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sammy has 20 cookies. Gab has twice as many as Sammy. "
        "Cher has twice as many as Gab. How many cookies does Cher have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 80.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_fraction_series(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "42 students applied. 1/3 got accepted. Of those accepted, half enrolled. How many enrolled?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 7.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_half_that_amount_additional(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Peggy has 6 dolls. Her grandmother gives Peggy her own collection of 30 dolls. "
        "Over the year, Peggy receives half that amount of dolls. How many dolls does Peggy have now?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 51.0
    assert meta.get("template_used") == "test_time_compute"


def test_schedule_rate_duration_from_initial_final(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Ashley had already blown up 12 balloons for the party when Andy took over and started blowing them up "
        "at a rate of 2 every five minutes. When Andy stopped, there were 50 balloons. "
        "For how many minutes did Andy blow up balloons?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 95.0
    assert meta.get("template_used") == "test_time_compute"


def test_schedule_weekly_sum_times_weeks(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "In one week, Jake can eat 3 papayas, his brother can eat 5 papayas, and his father can eat 4 papayas. "
        "To account for 4 weeks, how many papayas does Jake need to buy from the farmer's market?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 48.0
    assert meta.get("template_used") == "test_time_compute"


def test_legs_constraint_quadrupeds_and_humans(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Mariel is a dog walker. While walking her pack of dogs, she gets tangled up in the leashes of another "
        "dog walker and their 3 dogs. There are 36 legs tangled up in leashes. How many dogs is Mariel walking?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 5.0
    assert meta.get("template_used") == "test_time_compute"


def test_tiered_hourly_pay_overtime(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Bill gets paid $20 every hour he works up to a total of 40 hours, after which he gets paid double that "
        "amount per hour. How much does Bill get paid for a 50-hour workweek?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1200.0
    assert meta.get("template_used") == "test_time_compute"


def test_fraction_partition_rest_multiple_fractions(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Out of the 120 people who attended the party, 1/3 are men while half are women. "
        "The rest are children. How many children attended the party?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20.0
    assert meta.get("template_used") == "test_time_compute"


def test_piecewise_yield_fraction_half_rate(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Last year, Jorge planted corn on all of his 60 acres of property. Typically, corn grown on good soil yields 400 "
        "bushels per acre, but in clay-rich soil, the yield is only half as much per acre as in good soil. One-third of "
        "Jorge's 60 acres of land is clay-rich soil and the rest is good soil. How many bushels of corn did Jorge's land "
        "yield last year?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20000.0
    assert meta.get("template_used") == "test_time_compute"


def test_budget_savings_percent_of_monthly_sum(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Celia is creating a budget for the next 4 weeks. She wants to spend no more than $100 a week on food. "
        "For rent for her apartment, she plans to spend $1500. She has $30 set aside for video streaming services for the month. "
        "She also has $50 planned for one month of cell phone usage. After she adds up all of her spending for the month "
        "she wants to set aside 10% of it to put into savings. How much money is Celia going to put into her savings account?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 198.0
    assert meta.get("template_used") == "test_time_compute"


def test_after_gifts_relative_comparison(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Debelyn, Christel, and Andrena collect dolls. Debelyn had 20 dolls before she gave Andrena 2 dolls. "
        "Christel had 24 dolls before giving Andrena 5 dolls. After all the gifts, Andrena now has 2 more dolls than Christel, "
        "how many more dolls does Andrena have now than Debelyn?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 3.0
    assert meta.get("template_used") == "test_time_compute"


def test_day_by_day_sum_times_multiplier(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sidney does 20 jumping jacks on Monday, 36 on Tuesday, 40 on Wednesday, and 50 on Thursday. "
        "Brooke does three times as many jumping jacks as Sidney. How many jumping jacks did Brooke do?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 438.0
    assert meta.get("template_used") == "test_time_compute"


def test_schedule_every_interval_from_bags(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. "
        "The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. "
        "How long did the grill run?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 240.0
    assert meta.get("template_used") == "test_time_compute"


def test_schedule_story_trips_roundtrip_week(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Paul lives in a 5th story apartment. He makes 3 trips out from and back to his apartment throughout the day "
        "each day of a week. How many feet does he travel vertically in total over the week if each story is 10 feet tall?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 2100.0
    assert meta.get("template_used") == "test_time_compute"


def test_schedule_packaging_fraction_morning_bed(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Basil gets 1/2 of a dog cookie in the morning and before bed. She gets 2 whole cookies during the day. "
        "Basil’s cookies are packaged with 45 cookies per box. How many boxes will she need to last her for 30 days?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 2.0
    assert meta.get("template_used") == "test_time_compute"


def test_partition_fraction_ratio_remainder(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Erin put 16 curlers in her hair. One-fourth of the curlers are small pink ones. "
        "There are twice as many medium blue curlers as there are pink ones. The rest are large green curlers. "
        "How many large green curlers does Erin have in her hair?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 4.0
    assert meta.get("template_used") == "test_time_compute"


def test_coin_change_unknown_quarters(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A jar on the family's counter contains change they've been saving a trip to the ice cream shop. "
        "There are 123 pennies, 85 nickels, 35 dimes, and a number of quarters. "
        "All five family members get a double scoop, which costs $3 each. "
        "After the trip, they have 48 cents left over. How many quarters were in the jar?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 26.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_half_total_between_two_counts(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Tina has 40 students. Maura has 98 students. "
        "Zack has half the total number of students between Tina and Maura. "
        "How many students does Zack have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 69.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_half_total_between_equal_groups_absent_total_three_classrooms(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Tina's classroom has the same amount of students as Maura's. "
        "Zack's classroom has half the amount of total students between Tina and Maura's classrooms. "
        "How many students are there in total between the 3 classrooms if when Zack was sick there were 22 students in his class?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 69.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_two_step_difference(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Suraya picked 12 apples more than Caleb, and Caleb picked 5 apples less than Kayla. "
        "If Kayla picked 20 apples, how many more apples did Suraya pick than Kayla?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 7.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_fraction_word_ratio_with_delta(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Tom has 8 more marbles than Ben. Phillip has three eighths as many marbles as Tom. "
        "If Ben has 40 marbles, how many marbles does Phillip have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 18.0
    assert meta.get("template_used") == "test_time_compute"


def test_ordinal_throw_chain_total_from_terminal(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Ellie takes her little brother down to the pond to teach him how to skip stones. "
        "After he takes his first throw, she gives him pointers to improve his technique. "
        "His second throw skips two more times across the water than his first. "
        "His third throw skips twice as many times as his second. "
        "His fourth throw skips 3 fewer times than his third throw. "
        "His fifth throw skips one more time than the fourth throw. "
        "If his fifth throw skipped 8 times across the water, how many skips did he make in total between all of his throws?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 33.0
    assert meta.get("template_used") == "test_time_compute"


def test_inventory_started_with_losses_and_gains(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Claire was in charge of passing out free balloons to all the children at the fair. "
        "She started with 50 balloons. While passing 1 balloon to a little girl, 12 balloons floated away. "
        "Over the next thirty minutes, she gave 9 more away and grabbed the last 11 from her coworker. "
        "How many balloons does Claire have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 39.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_combined_sibling_multipliers(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sue has 48 stickers. Bill has 6 times as many stickers as Sue. "
        "Harry has 8 times as many stickers as Sue. How many stickers do Bill and Harry have combined?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 672.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_combined_chain_multipliers(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Bill has 6 times as many nuts as Harry, and Harry has twice as many nuts as Sue. "
        "If Sue has 48 nuts, how many do Bill and Harry have combined?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 672.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_inverse_multiplier_chain(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Kenny played basketball last week. He ran for twice as long as he played basketball, "
        "and he practiced on the trumpet for twice as long as he ran. "
        "If he practiced the trumpet for 40 minutes, how long did he play basketball?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_inverse_multiplier_chain_hours_variant(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Kenny played basketball last week. He ran for twice as long as he played basketball, "
        "and he practiced on the trumpet for twice as long as he ran. "
        "If he practiced the trumpet for 40 hours, how many hours did Kenny play basketball?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_divide_then_add(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Elon has 10 more teslas than Sam who has half the number of teslas as Chris. "
        "Chris has 6 teslas. How many teslas does Elon have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 13.0
    assert meta.get("template_used") == "test_time_compute"


def test_time_schedule_twice_that_long_total_minutes(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Every morning Harry takes 15 minutes to buy coffee and a bagel and twice that long "
        "to read the paper and eat before going in to work. How long does Harry's morning routine take?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 45.0
    assert meta.get("template_used") == "test_time_compute"


def test_playlist_hour_long_remaining_songs(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Minnie is making a playlist of songs for a party she will be throwing. She wants the playlist to be an hour long. "
        "She has added 16 three-minute songs to the playlist. How many four-minute songs can she add before reaching one hour?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 3.0
    assert meta.get("template_used") == "test_time_compute"


def test_inverse_fraction_per_day_days_to_finish(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        'Marc bought 50 episodes of the show "Friends" online. Each day Marc watches 1/10 of the episodes he bought. '
        "How many days will Marc need to finish 50 episodes of Friends?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10.0
    assert meta.get("template_used") == "test_time_compute"


def test_average_fit_into_ratio(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "If Jim has 20 apples, and Jane has 60 apples, and Jerry has 40 apples, "
        "how many times can Jim's number of apples fit into the average amount of apples for a person?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 2.0
    assert meta.get("template_used") == "test_time_compute"


def test_cost_sum_of_products_same_count(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Ali's class wants to order 35 English textbooks and 35 geography textbooks. "
        "Knowing that a geography book costs $10.50 and that an English book costs $7.50, "
        "what is the total amount of money spent?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 630.0
    assert meta.get("template_used") == "test_time_compute"


def test_insurance_out_of_pocket_percent_complement(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Tom broke his leg and needed to go to the doctor. The visit cost $300 and the cast cost $200. "
        "If insurance covered 60% how much was Tom's out-of-pocket cost?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 200.0
    assert meta.get("template_used") == "test_time_compute"


def test_pair_count_conversion(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "A drawer contains 40 pairs of socks. How many socks are in the drawer?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 80.0
    assert meta.get("template_used") == "test_time_compute"


def test_pair_inverse_conversion(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "There are 80 socks. How many pairs of socks are there?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 40.0
    assert meta.get("template_used") == "test_time_compute"


def test_division_each_cost(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "A dozen apples cost $6. How much does each apple cost?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 0.5
    assert meta.get("template_used") == "test_time_compute"


def test_pairs_mixed_with_each_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes. How many snakes did she see total?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 36.0
    assert meta.get("template_used") == "test_time_compute"


def test_half_rate_mpg_requires_more_gallons(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Darlene's car gets 20 miles/gallon. Martha's car gets half as many miles per gallon as Darlene’s car. "
        "How many gallons does Martha’s car require to make a 300-mile trip?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 30.0
    assert meta.get("template_used") == "test_time_compute"


def test_travel_time_twice_leg_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "It takes a duck 40 days to fly to the south during winter, twice as much time to fly to the north during summer, "
        "and 60 days to travel to the East during spring. How long does it take the duck to travel in total?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 180.0
    assert meta.get("template_used") == "test_time_compute"


def test_packaging_unit_price_chain(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A carton contains 12 boxes. If each box has 10 packs of cheese cookies, "
        "what's the price of a pack of cheese cookies if a dozen cartons cost $1440?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1.0
    assert meta.get("template_used") == "test_time_compute"


def test_each_get_sum_of_products_half_teaspoon(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Layla is feeding her pet fish. She has two Goldfish which each get one teaspoon of fish food. "
        "Her 3 Swordtails each get 2 teaspoons of food. Her 8 Guppies each get half a teaspoon of food. "
        "How many teaspoons of fish food does she need in total?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 12.0
    assert meta.get("template_used") == "test_time_compute"


def test_combined_total_more_than_double(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sara and Joe have a combined height of 120 inches. "
        "Joe is 6 inches more than double Sara's height. How tall is Joe?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 82.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_nested_more_less_multipliers(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sun City has 1000 more than twice as many people as Roseville City. "
        "Roseville city has 500 less than thrice as many people as Willowdale city. "
        "If Willowdale city has 2000 people, how many people does Sun City have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 12000.0
    assert meta.get("template_used") == "test_time_compute"


def test_times_fewer_then_twice_chain(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Melanie has twice as many cats as Annie, and Annie has three times fewer cats than Jacob. "
        "If Jacob has 90 cats, how many cats does Melanie have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 60.0
    assert meta.get("template_used") == "test_time_compute"


def test_more_then_fraction_of_derived_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Billy made 49 sandwiches; Katelyn made 47 more than that. "
        "Chloe made a quarter of the amount that Katelyn made. How many sandwiches did they make in all?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 169.0
    assert meta.get("template_used") == "test_time_compute"


def test_weighted_cost_other_more_each(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Henry took 9 pills a day for 14 days. Of these 9 pills, 4 pills cost $1.50 each, "
        "and the other pills each cost $5.50 more. How much did he spend in total on the pills?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 41.0
    assert meta.get("template_used") == "test_time_compute"


def test_rest_bought_linear_customers(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A watermelon stand sold 46 watermelons. Seventeen customers bought one melon, "
        "three customers bought three melons, and the rest bought two melons. "
        "How many customers bought two melons?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10.0
    assert meta.get("template_used") == "test_time_compute"


def test_school_lunch_schedule_weeks_minus_missed(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Jackson’s mom packs him a peanut butter and jelly sandwich for his school lunch on Wednesdays and Fridays. "
        "There are 36 weeks of school and Jackson has only missed 1 Wednesday and 2 Fridays. "
        "How many peanut butter and jelly sandwiches did Jackson eat for lunch?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 69.0
    assert meta.get("template_used") == "test_time_compute"


def test_weekly_charges_pay_every_weeks_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Grace just started her own business. Each week, she charges 300 dollars. "
        "Grace's client will pay her every 2 weeks. How many weeks will it take for Grace to get 1800 dollars?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 6.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_delta_times_chain_total_sum(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Miriam has five times as many albums as Katrina. Katrina has six times the number of albums as Bridget. "
        "Bridget has 15 fewer albums than Adele. If Adele has 30 albums, "
        "how many albums do Miriam, Katrina, Bridget, and Adele have altogether?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 585.0
    assert meta.get("template_used") == "test_time_compute"


def test_cash_vs_installment_savings(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Mr. Roberts can buy a television for $400 cash or $120 down payment and $30 a month for 12 months. "
        "How much can he save by paying cash?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 80.0
    assert meta.get("template_used") == "test_time_compute"


def test_full_theatre_adult_child_revenue(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A movie theatre has 250 seats. The cost of a ticket is $6 for an adult and $4 for a child. "
        "The theatre is full and contains 188 children. What is the total ticket cost?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1124.0
    assert meta.get("template_used") == "test_time_compute"


def test_weighted_fraction_groups_nice_people(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "All people named Barry are nice, while only half of the people named Kevin are nice. "
        "Three-fourths of people named Julie are nice, while 10% of people named Joe are nice. "
        "If a crowd contains 24 people named Barry, 20 people named Kevin, 80 people named Julie, and 50 people named Joe, "
        "how many nice people are in the crowd?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 99.0
    assert meta.get("template_used") == "test_time_compute"


def test_piecewise_pay_from_average_points(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "James's favorite basketball player got paid based on how many points he averaged each week. "
        "He gets $10,000 if he averages 30 or more points a game and $8,000 if he averages under 30 points a game. "
        "For his last week, in the first game he scored 30 points and in the second game he scored 28 points. "
        "In the third game he scored 32 points. In the fourth game he scored 34 points and in the fifth game he scored 26 points. "
        "How much does he get paid for the week?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10000.0
    assert meta.get("template_used") == "test_time_compute"


def test_mixed_acreage_yield_half_rate(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Last year, Jorge planted corn on all of his 60 acres of property. Typically, corn grown on good soil yields "
        "400 bushels per acre, but in clay-rich soil, the yield is only half as much per acre as in good soil. "
        "One-third of Jorge's 60 acres of land is clay-rich soil and the rest is good soil. "
        "How many bushels of corn did Jorge's land yield last year?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20000.0
    assert meta.get("template_used") == "test_time_compute"


def test_rotten_then_kept_fraction_cascade(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Iris has a berry farm. Her brother and sister help her to pick up the berries and sell them to the market. "
        "Iris picked 30 blueberries, her sister picked 20 cranberries, and her brother was able to pick 10 raspberries. "
        "If 1/3 of the total berries they were able to pick are rotten and the remaining 1/2 of the fresh berries need to be kept, "
        "how many berries will they be able to sell?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20.0
    assert meta.get("template_used") == "test_time_compute"


def test_expedition_relative_weeks_to_days(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Jodi and Vance are researching on a deserted island and have to stay on the island for a certain number of weeks to carry out their research. "
        "On their first expedition, they stayed for three weeks on the island. They spent two weeks more on the second expedition than they spent on their first expedition. "
        "They spent twice as many weeks on their last expedition as they spent on their second expedition. "
        "Calculate the total number of days they spent on the island on all the trips."
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 126.0
    assert meta.get("template_used") == "test_time_compute"


def test_monthly_budget_percent_to_savings(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Celia is creating a budget for the next 4 weeks. She wants to spend no more than $100 a week on food. "
        "For rent for her apartment, she plans to spend $1500. She has $30 set aside for video streaming services for the month. "
        "She also has $50 planned for one month of cell phone usage. After she adds up all of her spending for the month she wants to set aside 10% of it to put into savings. "
        "How much money is Celia going to put into her savings account?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 198.0
    assert meta.get("template_used") == "test_time_compute"


def test_unknown_unit_price_from_total_minus_known_items(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Jessica is having a sweet tooth and bought 10 chocolate bars, 10 packs of gummy bears, and 20 bags of chocolate chips. "
        "Her total rang up to $150. If the cost of a pack of gummy bears is $2 and a bag of chocolate chips costs $5, "
        "how much does 1 chocolate bar cost?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 3.0
    assert meta.get("template_used") == "test_time_compute"


def test_two_entity_weekly_cost_different_weeks(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Julia has a parrot and a rabbit. She buys food for both of the animals for $30 in total a week. "
        "Julia has the rabbit for 5 weeks, and the parrot for 3 weeks. "
        "How much money did Julia already spend on food for her animals, if the weekly cost of the rabbit food is $12?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 114.0
    assert meta.get("template_used") == "test_time_compute"


def test_sum_of_fraction_of_quantities_mixture(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "In a cheese-making process, a young lady mixes 3/5th of 20 liters of water with "
        "5/6th of 18 liters of vinegar. How many liters of the mixture are obtained?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 27.0
    assert meta.get("template_used") == "test_time_compute"


def test_times_as_many_altogether(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Grant has four times as many vacations as Kelvin has classes. If Kelvin has 90 classes, "
        "how many vacations and classes do Grant and Kelvin have altogether?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 450.0
    assert meta.get("template_used") == "test_time_compute"


def test_unit_price_discount_each(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "In a bookstore, a book costs $5. When Sheryll bought 10 books, she was given a discount of $0.5 each. "
        "How much did Sheryll pay in all?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 45.0
    assert meta.get("template_used") == "test_time_compute"


def test_tiered_hourly_pay_double_after_threshold(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Bill gets paid $20 every hour he works up to a total of 40 hours, after which he gets paid double that amount per hour. "
        "How much does Bill get paid for a 50-hour week?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 1200.0
    assert meta.get("template_used") == "test_time_compute"


def test_missing_item_cost_from_total_minus_known_costs(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "John bought a tennis racket. He also bought sneakers that cost $200 and a sports outfit that cost $250. "
        "He spent a total of $750 for all those items. What was the cost of the tennis racket?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 300.0
    assert meta.get("template_used") == "test_time_compute"


def test_remaining_equal_value_from_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Five shirts together cost $85. Of the 5 shirts, there are 3 shirts that cost $15 each. "
        "If the remaining shirts are each equal in value, what is the cost of each remaining shirt?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20.0
    assert meta.get("template_used") == "test_time_compute"


def test_per_person_discount_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Ali and Leila reserve their places for a trip to Egypt. The price is $147 per person, "
        "but they were each given a discount of $14 since there are two of them. "
        "How much do they pay in total?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 266.0
    assert meta.get("template_used") == "test_time_compute"


def test_percent_discount_multi_item_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "A spiral notebook costs $15, and a personal planner costs $10. "
        "How much would it cost in total to buy 4 spiral notebooks and 8 personal planners at a 20% discount?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 112.0
    assert meta.get("template_used") == "test_time_compute"


def test_currency_conversion_fraction_of_official_rate(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Willie came back from Europe with 70 euros. Normally 5 euros is worth 1 dollar, "
        "but the money exchange at the airport will only give Willie 5/7ths of the official conversion rate. "
        "How many dollars will Willie get?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 10.0
    assert meta.get("template_used") == "test_time_compute"


def test_twice_as_many_more_in_a_month(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Jake delivers 234 newspapers a week. Miranda delivers twice as many newspapers a week. "
        "How many more newspapers does Miranda deliver than Jake in a month?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 936.0
    assert meta.get("template_used") == "test_time_compute"


def test_per_day_relative_twice_week_total(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Holly needs to take 2 insulin pills per day, 3 blood pressure pills per day, "
        "and twice as many anticonvulsants as blood pressure pills each day. "
        "How many pills does Holly take in a week?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 77.0
    assert meta.get("template_used") == "test_time_compute"


def test_inverse_half_after_addition(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Keith bought 8 new baseball trading cards to add to his collection. "
        "The next day his dog ate half of his collection. There are now only 46 cards left. "
        "How many cards did Keith have before buying the new cards?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 84.0
    assert meta.get("template_used") == "test_time_compute"


def test_nested_each_multiplication_chain(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "There are 4 trains. Each train has 4 carriages. Each carriage has 3 rows. "
        "Each row has 5 wheels. How many wheels are there in total?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 240.0
    assert meta.get("template_used") == "test_time_compute"


def test_rest_of_then_half(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "There are 42 students total. 20 are from Europe and 10 are from South America. "
        "From the rest of the students, only half signed up. How many signed up?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 6.0
    assert meta.get("template_used") == "test_time_compute"


def test_right_triangles_fit_in_square(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "How many right triangles with a height of 2 inches and a width of two inches could fit "
        "inside a square with 2-inch sides?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 2.0
    assert meta.get("template_used") == "test_time_compute"


def test_consumable_pack_cost_over_days(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Judy uses 10 pencils during her 5 day school week. A 30 pack of pencils costs $4. "
        "How much will she spend on pencils over 45 days?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 12.0
    assert meta.get("template_used") == "test_time_compute"


def test_algebraic_combined_affine_height(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sara and Joe have a combined height of 120 inches. "
        "Joe is 6 inches more than double Sara's height. How tall is Joe?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 82.0
    assert meta.get("template_used") == "test_time_compute"


def test_algebraic_ratio_altogether(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Grant has four times as many vacations as Kelvin has classes. "
        "If Kelvin has 90 classes, how many vacations and classes do "
        "Grant and Kelvin have altogether?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 450.0
    assert meta.get("template_used") == "test_time_compute"


def test_algebraic_chain_to_terminal(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Melanie has twice as many cats as Annie, and Annie has three times fewer cats than Jacob. "
        "If Jacob has 90 cats, how many cats does Melanie have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 60.0
    assert meta.get("template_used") == "test_time_compute"


def test_algebraic_remaining_unit_cost(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Five shirts together cost $85. Of the 5 shirts, there are 3 shirts that cost $15 each. "
        "If the remaining shirts are each equal in value, what is the cost, in dollars, of one of the remaining shirts?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 20.0
    assert meta.get("template_used") == "test_time_compute"


def test_entity_chain_affine_multi_hop(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Sun City has 10 more than twice as many people as Roseville City. "
        "Roseville City has 5 less than thrice as many people as Willowdale city. "
        "If Willowdale city has 100 people, how many people does Sun City have?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 600.0
    assert meta.get("template_used") == "test_time_compute"


def test_weekly_schedule_bethany_riding_horses(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Bethany loved riding horses. She rode 1 hour after school every Monday, Wednesday, and Friday. "
        "On Tuesday and Thursday, she rode for 30 min and on Saturdays, she could ride for 2 hours. "
        "How many hours in total did she ride over a 2 week period?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 12.0
    assert meta.get("template_used") == "test_time_compute"


def test_relative_chain_three_entities_altogether(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Jar A has 28 marbles. Jar B has 12 more marbles than jar A. "
        "Jar C has twice as many marbles as jar B. How many marbles are there altogether?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 148.0
    assert meta.get("template_used") == "test_time_compute"


def test_gave_away_affine_constraint(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = (
        "Nigel won $45 but gave some away. His mother gave him $80 more. "
        "If now Nigel has $10 more than twice the amount he originally had, how much money did he give away?"
    )
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 25.0
    assert meta.get("template_used") == "test_time_compute"


def test_age_years_ago_future_twice(tmp_path):
    reader = _make_reader(tmp_path, thinking_budget=8)
    text = "3 years ago James turned 27. In 5 years Matt will be twice James age. How old is Matt now?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=3)
    assert result == 65.0
    assert meta.get("template_used") == "test_time_compute"
