"""
Example math grammar rules leveraging Math Galaxy symlinks.

Rules reference canonical symbols via symbol_refs (codepoints) without duplicating
glyph data. Enables cross-domain discovery and automatic generalization.
"""

import re

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule
from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY


# Grammar rules that compose RPN via Math Galaxy lookups (sovereign path)
SOVEREIGN_MATH_RULES = [
    GrammarRule(
        rule_id="latex_frac",
        language="math",
        pattern=r"\\frac\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\frac", m.group(1), m.group(2)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "\\frac{24}{4}", "output": "6"}],
    ),
    GrammarRule(
        rule_id="latex_binom",
        language="math",
        pattern=r"\\binom\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\binom", m.group(1), m.group(2)),
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "\\binom{10}{3}", "output": "120"}],
    ),
    GrammarRule(
        rule_id="latex_sqrt",
        language="math",
        pattern=r"\\sqrt\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\sqrt", m.group(1)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "\\sqrt{16}", "output": "4"}],
    ),
    GrammarRule(
        rule_id="factorial",
        language="math",
        pattern=r"(\d+)!",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("!", m.group(1)),
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "5!", "output": "120"}],
    ),
    GrammarRule(
        rule_id="power",
        language="math",
        pattern=r"(\d+)\^(\d+)",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("^", m.group(1), m.group(2)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "2^10", "output": "1024"}],
    ),
    GrammarRule(
        rule_id="latex_gcd_braces",
        language="math",
        pattern=r"\\gcd\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\gcd", m.group(1), m.group(2)),
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "\\gcd{12}{8}", "output": "4"}],
    ),
    GrammarRule(
        rule_id="latex_gcd_parens",
        language="math",
        pattern=r"gcd\((\d+),\s*(\d+)\)",
        rpn_program="{g0} {g1} gcd",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "gcd(12, 8)", "output": "4"}],
    ),
    GrammarRule(
        rule_id="latex_lcm_braces",
        language="math",
        pattern=r"\\lcm\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\lcm", m.group(1), m.group(2)),
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "\\lcm{6}{8}", "output": "24"}],
    ),
    GrammarRule(
        rule_id="latex_abs_braces",
        language="math",
        pattern=r"\\abs\{(-?\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\abs", m.group(1)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "\\abs{-7}", "output": "7"}],
    ),
    GrammarRule(
        rule_id="math_percent_of",
        language="english",
        pattern=r"(\d+\.?\d*)\s*%\s*(?:of)\s*(\d+\.?\d*)",
        rpn_program="{g1} {g0} 100 / *",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "25% of 80", "output": "20"}],
    ),
    GrammarRule(
        rule_id="latex_log_base_numeric",
        language="math",
        pattern=r"\\log_{(\d+)}\\((\d+)\\)",
        rpn_program="{g1} log {g0} log /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\log_{2}(8)", "output": "3"}],
    ),
    GrammarRule(
        rule_id="latex_ln_numeric",
        language="math",
        pattern=r"\\ln\\((\d+)\\)",
        rpn_program="{g0} log",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\ln(1)", "output": "0"}],
    ),
    GrammarRule(
        rule_id="latex_sin_numeric",
        language="math",
        pattern=r"\\sin\\((\d+)\\)",
        rpn_program="{g0} sin",
        domain="math_trig",
        symbol_refs=[],
        examples=[{"input": "\\sin(0)", "output": "0"}],
    ),
    GrammarRule(
        rule_id="latex_cos_numeric",
        language="math",
        pattern=r"\\cos\\((\d+)\\)",
        rpn_program="{g0} cos",
        domain="math_trig",
        symbol_refs=[],
        examples=[{"input": "\\cos(0)", "output": "1"}],
    ),
    GrammarRule(
        rule_id="latex_tan_numeric",
        language="math",
        pattern=r"\\tan\\((\d+)\\)",
        rpn_program="{g0} tan",
        domain="math_trig",
        symbol_refs=[],
        examples=[{"input": "\\tan(0)", "output": "0"}],
    ),
]

# --------------------------------------------------------------------------- #
# Galaxy-aware ("reading") rules
# --------------------------------------------------------------------------- #
GALAXY_AWARE_RULES = [
    # "[entity] has/had/owns [N] [items]" → base quantity.
    GrammarRule(
        rule_id="galaxy_has_quantity",
        language="math",
        pattern="word_sequence:has_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "proper_noun", "capture": "entity"},
                {"word_in": ["has", "had", "owns", "bought", "made", "started", "starts", "collected", "collects", "collect", "gathered", "gathers", "gather"]},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "John has 15 apples", "output": "base=15"}],
    ),
    # "[entity] has/had/owns [N]" → base quantity (no explicit noun).
    GrammarRule(
        rule_id="galaxy_has_quantity_no_noun",
        language="math",
        pattern="word_sequence:has_quantity_no_noun",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "proper_noun", "capture": "entity"},
                {"word_in": ["has", "had", "owns", "bought", "made", "started", "starts", "collected", "collects", "collect", "gathered", "gathers", "gather"]},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "John has 15", "output": "base=15"}],
    ),
    # "and [N] [items]" → additional quantity in a list (common GSM8K phrasing).
    GrammarRule(
        rule_id="galaxy_and_quantity_noun",
        language="math",
        pattern="word_sequence:and_quantity_noun",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "and"},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "and 4 cartons", "output": "base=4"}],
    ),
    # "If [N] [items] ..." → capture the given quantity in conditional statements.
    GrammarRule(
        rule_id="galaxy_if_quantity_noun",
        language="math",
        pattern="word_sequence:if_quantity_noun",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "if"},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "If 750 jelly beans are coconut flavored", "output": "base=750"}],
    ),
    # "For $[N] <pronoun> bought/paid/spent ..." → capture a budget/total spend anchor.
    GrammarRule(
        rule_id="galaxy_for_dollar_bought_total",
        language="math",
        pattern="word_sequence:for_dollar_bought_total",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "for"},
                {"word": "$"},
                {"category": "number", "capture": "total"},
                {"word_in": ["she", "he", "they", "we", "i"]},
                {"word_in": ["bought", "buy", "buys", "purchased", "spent", "paid"]},
            ],
        },
        examples=[{"input": "For $75 she bought ...", "output": "total=75"}],
    ),
    # "also bought/purchased [N] [items]" → base quantity (used to pick up extra unpriced items).
    GrammarRule(
        rule_id="galaxy_bought_quantity_noun",
        language="math",
        pattern="word_sequence:bought_quantity_noun",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "also"},
                {"word_in": ["bought", "buy", "buys", "purchased", "purchases"]},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "She also bought 4 tops", "output": "base=4"}],
    ),
    # "[N] <items> ... for $[P] each" → capture a priced item term (count*price).
    GrammarRule(
        rule_id="galaxy_count_for_dollar_each",
        language="math",
        pattern="word_sequence:count_for_dollar_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["for", "at"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "5 pairs of shorts for $7 each", "output": "count=5 price=7"}],
    ),
    # Multi-item cost aggregation:
    # "<c1> ... for $<p1> each ... and <c2> ... for $<p2> each" → enables summing cost terms.
    GrammarRule(
        rule_id="multi_item_cost_sum",
        language="math",
        pattern="word_sequence:multi_item_cost_sum",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 18,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation", "verb"],
            "word_pattern": [
                {"category": "number", "capture": "count_a"},
                {"word": "$"},
                {"category": "number", "capture": "price_a"},
                {"word_in": ["each", "apiece"]},
                {"word": "and"},
                {"category": "number", "capture": "count_b"},
                {"word": "$"},
                {"category": "number", "capture": "price_b"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[
            {
                "input": "35 items for $10.50 each and 35 items for $7.50 each",
                "output": "terms=35*10.5 + 35*7.5",
            }
        ],
    ),
    # Multi-item cost aggregation with shared count:
    # "<count> ... for $<p1> each ... and ... for $<p2> each" → implies count*p1 + count*p2.
    GrammarRule(
        rule_id="multi_item_cost_sum_context",
        language="math",
        pattern="word_sequence:multi_item_cost_sum_context",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 22,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation", "verb"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"word": "$"},
                {"category": "number", "capture": "price_a"},
                {"word_in": ["each", "apiece"]},
                {"word": "and"},
                {"word": "$"},
                {"category": "number", "capture": "price_b"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[
            {
                "input": "35 students bought a book for $10.50 each and a notebook for $7.50 each",
                "output": "terms=35*10.5 + 35*7.5",
            }
        ],
    ),
    # "bags have [N] <noun> each" → capture per-bag quantity (used with "as many ... as K of ... bags").
    GrammarRule(
        rule_id="galaxy_bags_have_each",
        language="math",
        pattern="word_sequence:bags_have_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["bag", "bags"]},
                {"word": "have"},
                {"category": "number", "capture": "each"},
                {"category": "noun"},
                {"word": "each"},
            ],
        },
        examples=[{"input": "bags have 40 apples each", "output": "each=40"}],
    ),
    # "[N] <items> ... cost $[P] each" → capture a priced item term (count*price).
    # Common GSM8K phrasing: "5 hamburgers that cost $3 each".
    GrammarRule(
        rule_id="galaxy_count_cost_dollar_each",
        language="math",
        pattern="word_sequence:count_cost_dollar_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["cost", "costs", "priced", "price", "prices"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "5 hamburgers that cost $3 each", "output": "count=5 price=3"}],
    ),
    # "there are [N] [items]" → base quantity.
    GrammarRule(
        rule_id="galaxy_there_are_quantity",
        language="math",
        pattern="word_sequence:there_are_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["are", "were", "is", "was"]},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "There are 12 cookies", "output": "base=12"}],
    ),
    # "has as many <noun> as [K]" → capture multiplier for comparative bundle problems.
    # Example: "Each of her bags has as many apples as 3 of Gerald's bags."
    GrammarRule(
        rule_id="galaxy_has_as_many_as_count",
        language="math",
        pattern="word_sequence:has_as_many_as_count",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "proper_noun"],
            "word_pattern": [
                {"word_in": ["has", "have"]},
                {"word": "as"},
                {"word": "many"},
                {"category": "noun"},
                {"word": "as"},
                {"category": "number", "capture": "multiplier"},
            ],
        },
        examples=[{"input": "has as many apples as 3 ...", "output": "multiplier=3"}],
    ),
    # "[pct]% of <noun phrase>" → percent-of operation applied to an implicit total (when a total quantity is known).
    # Example: "80% of all the books are in English" (total from earlier "there are 2300 books").
    GrammarRule(
        rule_id="galaxy_percent_of_noun",
        language="math",
        pattern="word_sequence:percent_of_noun",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "pct"},
                {"word_in": ["%", "percent"]},
                {"word": "of"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "80% of all the books", "output": "op=percent_of 80"}],
    ),
    # "[N] were boys" → capture a labeled subgroup count (used for remaining/girls problems).
    GrammarRule(
        rule_id="galaxy_were_boys",
        language="math",
        pattern="word_sequence:were_boys",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "boys"},
                {"word_in": ["were", "was", "are", "is"]},
                {"word_in": ["boys", "boy"]},
            ],
        },
        examples=[{"input": "Eight were boys.", "output": "boys=8"}],
    ),
    # "each [noun] eats/consumes [N] [noun]" → capture the per-entity rate/count number.
    # Used for multiplicative chains like "each snake eats 3 birds per day".
    GrammarRule(
        rule_id="galaxy_each_eats_quantity",
        language="math",
        pattern="word_sequence:each_eats_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol"],
            "word_pattern": [
                {"word": "each"},
                {"category": "noun"},
                {"word_in": ["eats", "eat", "consumes", "consume"]},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "each bird eats 12 beetles per day", "output": "base=12"}],
    ),
    # "... balance is $ [N]" → extract monetary balance amounts (e.g., "$3,456").
    GrammarRule(
        rule_id="galaxy_balance_is_dollar",
        language="math",
        pattern="word_sequence:balance_is_dollar",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["balance", "balances"]},
                {"word_in": ["is", "was", "equals", "equaled"]},
                {"word": "$"},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "account balance is $3,456", "output": "base=3456"}],
    ),
    # "There is a [N] ... and a [M] ..." → capture two quantities (e.g., adjacent house sizes).
    GrammarRule(
        rule_id="galaxy_there_is_a_and_a",
        language="math",
        pattern="word_sequence:there_is_a_and_a",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["is", "are", "was", "were"]},
                {"category": "number", "capture": "a"},
                {"word": "and"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "There is a 5,200 sq ft house and a 7,300 sq ft house", "output": "a=5200 b=7300"}],
    ),
    # "[ProperNoun] sold [Number] [Noun]" → capture base quantity.
    GrammarRule(
        rule_id="galaxy_sold_quantity",
        language="math",
        pattern="word_sequence:sold_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "proper_noun"},
                {"word": "sold"},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "Natalia sold 48 clips", "output": "base=48"}],
    ),
    # "[ProperNoun] sold [Noun] to [Number]" → capture base quantity (common GSM8K phrasing).
    GrammarRule(
        rule_id="galaxy_sold_to_quantity",
        language="math",
        pattern="word_sequence:sold_to_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "proper_noun"},
                {"word": "sold"},
                {"category": "noun"},
                {"word": "to"},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "Natalia sold clips to 48 friends", "output": "base=48"}],
    ),
    # "sold [N] ... average cost/price ... $ [P]" → capture (count, avg_price) for weighted averages.
    GrammarRule(
        rule_id="galaxy_sold_avg_cost_dollar",
        language="math",
        pattern="word_sequence:sold_avg_cost_dollar",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 16,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["sold", "sell", "sells"]},
                {"category": "number", "capture": "count"},
                {"word_in": ["average", "avg"]},
                {"word_in": ["cost", "price"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
            ],
        },
        examples=[
            {
                "input": "Apple sold 100 iPhones for an average cost of $1000.",
                "output": "count=100 price=1000",
            }
        ],
    ),
    # "and [N] ... average cost/price ... $ [P]" → tail clause for weighted average prompts.
    GrammarRule(
        rule_id="galaxy_and_avg_cost_dollar",
        language="math",
        pattern="word_sequence:and_avg_cost_dollar",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 18,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "verb", "aggregation"],
            "word_pattern": [
                {"word": "and"},
                {"category": "number", "capture": "count"},
                {"word_in": ["average", "avg"]},
                {"word_in": ["cost", "price"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
            ],
        },
        examples=[
            {
                "input": "..., and 80 Apple TVs at an average cost of $200.",
                "output": "count=80 price=200",
            }
        ],
    ),
    # "each ... cost $ [P]" / "each ... costs $ [P]" → capture per-item cost for later composition.
    GrammarRule(
        rule_id="galaxy_each_item_cost_dollar",
        language="math",
        pattern="word_sequence:each_item_cost_dollar",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "each"},
                {"word_in": ["cost", "costs", "priced", "price"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
            ],
        },
        examples=[{"input": "Each carton of ice cream cost $4", "output": "price=4"}],
    ),
    # "creates/makes ... for $ [N]" → fixed upfront cost (profit problems).
    GrammarRule(
        rule_id="galaxy_creates_for_money",
        language="math",
        pattern="word_sequence:creates_for_money",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["creates", "create", "created", "makes", "make", "made"]},
                {"word": "for"},
                {"word": "$"},
                {"category": "number", "capture": "fixed_cost"},
            ],
        },
        examples=[{"input": "He creates a movie for $2000.", "output": "fixed_cost=2000"}],
    ),
    # "sells ... for [M] times that much" → markup multiplier.
    GrammarRule(
        rule_id="galaxy_sells_for_times_that_much",
        language="math",
        pattern="word_sequence:sells_for_times_that_much",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["sells", "sell", "sold"]},
                {"word": "for"},
                {"category": "number", "capture": "multiplier"},
                {"word": "times"},
            ],
        },
        examples=[{"input": "He sells it for 2.5 times that much.", "output": "multiplier=2.5"}],
    ),
    # "sells [N] ... a day" / "sells [N] ... per day" → units-per-day rate.
    GrammarRule(
        rule_id="galaxy_sells_per_day",
        language="math",
        pattern="word_sequence:sells_per_day",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["sells", "sell", "sold"]},
                {"category": "number", "capture": "per_day"},
                {"word_in": ["a", "per"]},
                {"word_in": ["day", "days"]},
            ],
        },
        examples=[{"input": "He sells 500 movies a day.", "output": "per_day=500"}],
    ),
    # "for [N] days a week" / "[N] days per week" → days-per-week factor.
    GrammarRule(
        rule_id="galaxy_days_a_week",
        language="math",
        pattern="word_sequence:days_a_week",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["for", "in"]},
                {"category": "number", "capture": "days_per_week"},
                {"word_in": ["day", "days"]},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "for 5 days a week", "output": "days_per_week=5"}],
    ),
    # "[N] days a week" (no explicit "for/in") → days-per-week factor.
    GrammarRule(
        rule_id="galaxy_days_a_week_no_prep",
        language="math",
        pattern="word_sequence:days_a_week_no_prep",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "days_per_week"},
                {"word_in": ["day", "days"]},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "she practices 6 days a week", "output": "days_per_week=6"}],
    ),
    # "[N] hours a week" / "[N] hours per week" → weekly hours workload.
    GrammarRule(
        rule_id="galaxy_hours_a_week",
        language="math",
        pattern="word_sequence:hours_a_week",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "hours"},
                {"word_in": ["hour", "hours"]},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "she works 35 hours a week", "output": "hours=35"}],
    ),
    # "a month" (implicit 1 month) → duration hint.
    GrammarRule(
        rule_id="galaxy_a_month",
        language="math",
        pattern="word_sequence:a_month",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 0,
            "skip_categories": ["symbol"],
            "word_pattern": [
                {"word_in": ["a", "an", "one"]},
                {"word_in": ["month", "months"]},
            ],
        },
        examples=[{"input": "It's been a month since she started.", "output": "months=1"}],
    ),
    # "in/for/over [N] weeks" → week duration/count.
    GrammarRule(
        rule_id="galaxy_in_weeks",
        language="math",
        pattern="word_sequence:in_weeks",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 0,
            "skip_categories": ["stopword", "symbol", "aggregation"],
            "word_pattern": [
                {"word_in": ["in", "for", "over"]},
                {"category": "number", "capture": "weeks"},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "in 20 weeks", "output": "weeks=20"}],
    ),
    # "with [N] weeks" → week duration/count ("a month with four weeks").
    GrammarRule(
        rule_id="galaxy_with_weeks",
        language="math",
        pattern="word_sequence:with_weeks",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "with"},
                {"category": "number", "capture": "weeks"},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "a month with four weeks", "output": "weeks=4"}],
    ),
    # "saved $ [N]" / "saved [N]" → base quantity (money accumulation).
    GrammarRule(
        rule_id="galaxy_saved_amount",
        language="math",
        pattern="word_sequence:saved_amount",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"word_in": ["saved", "save"]},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "she had saved $3000", "output": "base=3000"}],
    ),
    # "added $ [N]" / "added [N]" → additional quantity (money accumulation).
    GrammarRule(
        rule_id="galaxy_added_amount",
        language="math",
        pattern="word_sequence:added_amount",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"word_in": ["added", "add", "adds"]},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "family added $7000", "output": "base=7000"}],
    ),
    # "ordered ... for $ [N]" → line-item price (subtotal term).
    GrammarRule(
        rule_id="galaxy_ordered_for_money",
        language="math",
        pattern="word_sequence:ordered_for_money",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["ordered", "order"]},
                {"word": "for"},
                {"word": "$"},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "ordered a steak for $80", "output": "amount=80"}],
    ),
    # "cost/price ... is $ [N]" → unit cost label (e.g., reimburse/overcharge).
    GrammarRule(
        rule_id="galaxy_unit_cost_is",
        language="math",
        pattern="word_sequence:unit_cost_is",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["cost", "price"]},
                {"word_in": ["is", "was", "are", "were"]},
                {"word": "$"},
                {"category": "number", "capture": "unit_cost"},
            ],
        },
        examples=[{"input": "the cost of a piece is $134", "output": "unit_cost=134"}],
    ),
    # "[N] / [M]" → direct division expression.
    GrammarRule(
        rule_id="galaxy_divide_symbol",
        language="math",
        pattern="word_sequence:divide_symbol",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "/"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "24 / 4", "output": "24 4 /"}],
    ),
    # "[N] divided by [M]" → division expression.
    GrammarRule(
        rule_id="galaxy_divided_by",
        language="math",
        pattern="word_sequence:divided_by",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "divided"},
                {"word": "by"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "24 divided by 4", "output": "24 4 /"}],
    ),
    # "[N] * [M]" → multiplication expression.
    GrammarRule(
        rule_id="galaxy_times_symbol",
        language="math",
        pattern="word_sequence:times_symbol",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word_in": ["*", "×", "x"]},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "3 * 5", "output": "3 5 *"}],
    ),
    # "[N] times [M]" → multiplication expression.
    GrammarRule(
        rule_id="galaxy_times",
        language="math",
        pattern="word_sequence:times",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "times"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "3 times 5", "output": "3 5 *"}],
    ),
    # "[N] + [M]" / "[N] plus [M]" → addition expression.
    GrammarRule(
        rule_id="galaxy_plus_symbol",
        language="math",
        pattern="word_sequence:plus_symbol",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "+"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "3 + 5", "output": "3 5 +"}],
    ),
    GrammarRule(
        rule_id="galaxy_plus",
        language="math",
        pattern="word_sequence:plus",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "plus"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "3 plus 5", "output": "3 5 +"}],
    ),
    # "[N] - [M]" / "[N] minus [M]" → subtraction expression.
    GrammarRule(
        rule_id="galaxy_minus_symbol",
        language="math",
        pattern="word_sequence:minus_symbol",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "-"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "10 - 3", "output": "10 3 -"}],
    ),
    GrammarRule(
        rule_id="galaxy_minus",
        language="math",
        pattern="word_sequence:minus",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "minus"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "10 minus 3", "output": "10 3 -"}],
    ),
    # "[pct] % of [N]" → arithmetic percent-of.
    GrammarRule(
        rule_id="galaxy_percent_of",
        language="math",
        pattern="word_sequence:percent_of",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "pct"},
                {"word": "%"},
                {"word": "of"},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "25% of 80", "output": "80 25 100 / *"}],
    ),
    # Percent complement (building block): "[pct]% are ..." → percent-of signal without explicit "of".
    # Used with a known total (e.g., "There are 2300 books. 80% are in English. How many are not?")
    GrammarRule(
        rule_id="percent_complement_subtract",
        language="math",
        pattern="word_sequence:percent_complement_subtract",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "pct"},
                {"word_in": ["%", "percent"]},
                {"word_in": ["are", "is", "were", "was"]},
            ],
        },
        examples=[{"input": "80% are in English", "output": "op=percent_of 80"}],
    ),
    # Percent complement question cue: "how many ... not" → treat as a complement/difference request.
    GrammarRule(
        rule_id="percent_complement_direct",
        language="math",
        pattern="word_sequence:percent_complement_direct",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "how"},
                {"word": "many"},
                {"word": "not"},
            ],
        },
        examples=[{"input": "How many are not in English?", "output": "goal=percent_complement"}],
    ),
    # "[a]/[b] of [N]" → fraction-of expression (e.g., "5/6 of the nuts").
    GrammarRule(
        rule_id="galaxy_fraction_of",
        language="math",
        pattern="word_sequence:fraction_of",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "num"},
                {"word": "/"},
                {"category": "number", "capture": "den"},
                {"word": "of"},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "If 5/6 of 30 nuts were eaten", "output": "eaten=30*5/6"}],
    ),
    # "[a]/[b] ... eaten/used/spent/lost" → indicates subtraction of that fraction-of-total.
    GrammarRule(
        rule_id="galaxy_fraction_eaten",
        language="math",
        pattern="word_sequence:fraction_eaten",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "num"},
                {"word": "/"},
                {"category": "number", "capture": "den"},
                {"word_in": ["eaten", "used", "spent", "lost"]},
            ],
        },
        examples=[{"input": "5/6 of the nuts were eaten", "output": "op=subtract(total*5/6)"}],
    ),
    # "half as many" → flag an operation (resolved during composition).
    GrammarRule(
        rule_id="galaxy_half_as_many",
        language="math",
        pattern="word_sequence:half_as_many",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "half"},
                {"word": "as"},
                {"word": "many"},
            ],
        },
        examples=[{"input": "half as many", "output": "op=half"}],
    ),
    # "a/one third of ..." → fraction-part signal (used for "the rest" composition).
    GrammarRule(
        rule_id="galaxy_a_third_of",
        language="math",
        pattern="word_sequence:a_third_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["a", "one"]},
                {"word_in": ["third", "thirds"]},
                {"word": "of"},
            ],
        },
        examples=[{"input": "a third of its sales", "output": "fraction=1/3"}],
    ),
    # "a/one quarter of ..." → fraction-part signal (used for "the rest" composition).
    GrammarRule(
        rule_id="galaxy_a_quarter_of",
        language="math",
        pattern="word_sequence:a_quarter_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["a", "one"]},
                {"word_in": ["quarter", "quarters"]},
                {"word": "of"},
            ],
        },
        examples=[{"input": "a quarter of its sales", "output": "fraction=1/4"}],
    ),
    # "<num> <denom_word> of ..." → general word-fraction signal (e.g., "three fourths of").
    GrammarRule(
        rule_id="galaxy_fraction_words_of",
        language="math",
        pattern="word_sequence:fraction_words_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "num"},
                {
                    "word_in": [
                        "half",
                        "halves",
                        "third",
                        "thirds",
                        "fourth",
                        "fourths",
                        "quarter",
                        "quarters",
                        "fifth",
                        "fifths",
                        "sixth",
                        "sixths",
                        "seventh",
                        "sevenths",
                        "eighth",
                        "eighths",
                        "ninth",
                        "ninths",
                        "tenth",
                        "tenths",
                    ],
                    "capture": "denom_word",
                },
                {"word": "of"},
            ],
        },
        examples=[{"input": "three fourths of the jelly beans", "output": "fraction=3/4"}],
    ),
    # "the rest" / "rest" → remainder goal for fraction-part problems.
    GrammarRule(
        rule_id="galaxy_the_rest",
        language="math",
        pattern="word_sequence:the_rest",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["rest", "remaining"]},
            ],
        },
        examples=[{"input": "and the rest in stationery", "output": "goal=rest"}],
    ),
    # "twice as many" → * 2 (derivation of second quantity from base).
    GrammarRule(
        rule_id="galaxy_twice_as_many",
        language="math",
        pattern="word_sequence:twice_as_many",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "twice"},
                {"word": "as"},
                {"word": "many"},
            ],
        },
        examples=[{"input": "twice as many", "output": "derive=*2"}],
    ),
    # "three times as many" → * 3 (derivation).
    GrammarRule(
        rule_id="galaxy_n_times_as_many",
        language="math",
        pattern="word_sequence:n_times_as_many",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "multiplier"},
                {"word": "times"},
                {"word": "as"},
                {"word": "many"},
            ],
        },
        examples=[{"input": "3 times as many", "output": "derive=*3"}],
    ),
    # "gave [N] to" → subtraction op on the running total.
    GrammarRule(
        rule_id="galaxy_gave_to",
        language="math",
        pattern="word_sequence:gave_to",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"word_in": ["gave", "gives", "give"]},
                {"category": "number", "capture": "amount"},
                {"word": "to"},
            ],
        },
        examples=[{"input": "gave 3 to", "output": "op=-3"}],
    ),
    # "gave [N] [items] to" → subtract.
    GrammarRule(
        rule_id="galaxy_gave_amount_noun_to",
        language="math",
        pattern="word_sequence:gave_amount_noun_to",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["gave", "gives", "give"]},
                {"category": "number", "capture": "amount"},
                {"category": "noun"},
                {"word": "to"},
            ],
        },
        examples=[{"input": "gave 3 apples to", "output": "op=-3"}],
    ),
    # "received/got [N]" → addition op.
    GrammarRule(
        rule_id="galaxy_received",
        language="math",
        pattern="word_sequence:received",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "noun", "symbol"],
            "word_pattern": [
                {"word_in": ["received", "gets", "got", "gained", "found", "earned"]},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "got 4", "output": "op=+4"}],
    ),
    # "received/got [N] [items]" → add.
    GrammarRule(
        rule_id="galaxy_received_amount_noun",
        language="math",
        pattern="word_sequence:received_amount_noun",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["received", "gets", "got", "gained", "found", "earned"]},
                {"category": "number", "capture": "amount"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "got 4 apples", "output": "op=+4"}],
    ),
    # "spent $ [N]" → subtract.
    GrammarRule(
        rule_id="galaxy_spent_money",
        language="math",
        pattern="word_sequence:spent_money",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["spent", "spends", "spend", "paid", "pays", "pay"]},
                {"word": "$"},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "spent $ 5", "output": "op=-5"}],
    ),
    # "spent [N]" → subtract (money often omits $).
    GrammarRule(
        rule_id="galaxy_spent_amount",
        language="math",
        pattern="word_sequence:spent_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["spent", "spends", "spend", "paid", "pays", "pay"]},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "spent 5", "output": "op=-5"}],
    ),
    # "[N] ... quit" → subtract (team/member counts).
    GrammarRule(
        rule_id="galaxy_quit_amount",
        language="math",
        pattern="word_sequence:quit_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word_in": ["quit", "quits", "quitters"]},
            ],
        },
        examples=[{"input": "8 people quit", "output": "op=-8"}],
    ),
    # "[N] new ... got in" → add (team/member counts).
    GrammarRule(
        rule_id="galaxy_new_got_in_amount",
        language="math",
        pattern="word_sequence:new_got_in_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun", "verb", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word": "new"},
                {"word_in": ["got", "get", "gets"]},
                {"word": "in"},
            ],
        },
        examples=[{"input": "13 new people got in", "output": "op=+13"}],
    ),
    # "lost [N]" → subtract.
    GrammarRule(
        rule_id="galaxy_lost_amount",
        language="math",
        pattern="word_sequence:lost_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["lost", "loses", "lose", "used", "uses", "use"]},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "lost 2", "output": "op=-2"}],
    ),
    # "[N] less than" → subtraction hint (usually applied after a derived/base quantity).
    GrammarRule(
        rule_id="galaxy_less_than",
        language="math",
        pattern="word_sequence:less_than",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word_in": ["less", "fewer"]},
                {"word": "than"},
            ],
        },
        examples=[{"input": "5 less than", "output": "op=-5"}],
    ),
    # "[N] more than" → addition hint.
    GrammarRule(
        rule_id="galaxy_more_than",
        language="math",
        pattern="word_sequence:more_than",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word": "more"},
                {"word": "than"},
            ],
        },
        examples=[{"input": "3 more than", "output": "op=+3"}],
    ),
    # Relative pattern building blocks (TIER-1): keep these generic and compositional.
    GrammarRule(
        rule_id="relative_more_than",
        language="math",
        pattern="word_sequence:relative_more_than",
        rpn_program="",
        domain="math_relative",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word_in": ["more", "additional", "extra"]},
                {"word": "than"},
                {"category_in": ["noun", "proper_noun"], "capture": "base"},
            ],
        },
        examples=[{"input": "40 more than sheep", "output": "relative(+40)"}],
    ),
    GrammarRule(
        rule_id="relative_less_than",
        language="math",
        pattern="word_sequence:relative_less_than",
        rpn_program="",
        domain="math_relative",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "amount"},
                {"word_in": ["less", "fewer"]},
                {"word": "than"},
                {"category_in": ["noun", "proper_noun"], "capture": "base"},
            ],
        },
        examples=[{"input": "13 less than Mary", "output": "relative(-13)"}],
    ),
    GrammarRule(
        rule_id="relative_multiple_of",
        language="math",
        pattern="word_sequence:relative_multiple_of",
        rpn_program="",
        domain="math_relative",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "aggregation"],
            "word_pattern": [
                {"word_in": ["twice", "double"]},
                {"category_in": ["noun", "proper_noun"], "capture": "base"},
            ],
        },
        examples=[{"input": "twice Mary's", "output": "relative(*2)"}],
    ),
    GrammarRule(
        rule_id="relative_times_quantity",
        language="math",
        pattern="word_sequence:relative_times_quantity",
        rpn_program="",
        domain="math_relative",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "multiplier"},
                {"word_in": ["times", "×", "x"]},
                {"category_in": ["noun", "proper_noun"], "capture": "base"},
            ],
        },
        examples=[{"input": "3 times the goats", "output": "relative(*3)"}],
    ),
    # "[N] times more" → multiplicative-plus-base hint (interpreted as base*(N+1) for GSM8K-style phrasing).
    GrammarRule(
        rule_id="galaxy_times_more",
        language="math",
        pattern="word_sequence:times_more",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "multiplier"},
                {"word": "times"},
                {"word": "more"},
            ],
        },
        examples=[{"input": "25 times more", "output": "op=times_more multiplier=25"}],
    ),
    # "plus [N]" → addition op (common list phrasing).
    GrammarRule(
        rule_id="galaxy_plus_amount",
        language="math",
        pattern="word_sequence:plus_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"word": "plus"},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "plus 3 coaches", "output": "op=+3"}],
    ),
    # "plus [N] ... and [M]" → two additions (e.g. "plus 3 coaches and 2 helpers").
    GrammarRule(
        rule_id="galaxy_plus_amount_and_amount",
        language="math",
        pattern="word_sequence:plus_amount_and_amount",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"word": "plus"},
                {"category": "number", "capture": "a"},
                {"word": "and"},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "plus 3 coaches and 2 helpers", "output": "op=+3, op=+2"}],
    ),
    # "packs of [N]" / "pack of [N]" → post-division hint (apply after totals).
    GrammarRule(
        rule_id="galaxy_packs_of",
        language="math",
        pattern="word_sequence:packs_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "noun", "symbol", "proper_noun"],
            "word_pattern": [
                {"word_in": ["pack", "packs"]},
                {"word": "of"},
                {"category": "number", "capture": "pack_size"},
            ],
        },
        examples=[{"input": "comes in packs of 6", "output": "op=post_divide 6"}],
    ),
    # "[N] [items] at $ [price] each" → multiply.
    GrammarRule(
        rule_id="galaxy_each_cost_dollar",
        language="math",
        pattern="word_sequence:each_cost_dollar",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["at", "for"]},
                {"word": "$"},
                {"category": "number", "capture": "price"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "3 apples at $ 2 each", "output": "op=*2 (repeat 3)"}],
    ),
    # "[N] [items] at [price] each" → multiply.
    GrammarRule(
        rule_id="galaxy_each_cost",
        language="math",
        pattern="word_sequence:each_cost",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["at", "for"]},
                {"category": "number", "capture": "price"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "3 apples at 2 each", "output": "op=*2 (repeat 3)"}],
    ),
    # "[rate] per [unit] [N]" → multiply.
    GrammarRule(
        rule_id="galaxy_rate_per_unit",
        language="math",
        pattern="word_sequence:rate_per_unit",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "rate"},
                {"word": "per"},
                {"category": "noun"},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "5 per day 7", "output": "op=*5 (repeat 7)"}],
    ),
    # "earns $ [rate] per hour" → multiply by hourly wage (applied after computing hours).
    GrammarRule(
        rule_id="galaxy_earns_dollars_per_hour",
        language="math",
        pattern="word_sequence:earns_dollars_per_hour",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["earns", "earn", "earned", "makes", "make", "made"]},
                {"word": "$"},
                {"category": "number", "capture": "rate"},
                {"word": "per"},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "she earns $15 per hour", "output": "op=*15"}],
    ),
    # "works for $ [rate] an hour" / "works at $ [rate] per hour" → multiply by hourly wage.
    GrammarRule(
        rule_id="galaxy_works_for_dollars_per_hour",
        language="math",
        pattern="word_sequence:works_for_dollars_per_hour",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 5,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["works", "work", "worked"]},
                {"word_in": ["for", "at"]},
                {"word": "$"},
                {"category": "number", "capture": "rate"},
                {"word_in": ["an", "a", "per"]},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "She works for $8 an hour", "output": "op=*8"}],
    ),
    # "earns [rate] per hour" (no $) → multiply by hourly wage.
    GrammarRule(
        rule_id="galaxy_earns_per_hour",
        language="math",
        pattern="word_sequence:earns_per_hour",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["earns", "earn", "earned", "makes", "make", "made"]},
                {"category": "number", "capture": "rate"},
                {"word": "per"},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "he earns 15 per hour", "output": "op=*15"}],
    ),
    # "takes [N] hours" / "it takes ... [N] hours" → multiply by duration-per-task.
    GrammarRule(
        rule_id="galaxy_takes_hours",
        language="math",
        pattern="word_sequence:takes_hours",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["takes", "take"]},
                {"category": "number", "capture": "hours"},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "it takes her 6 hours", "output": "op=*6"}],
    ),
    # "in/for/over [N] days" → duration hint (used for per-day conversions).
    GrammarRule(
        rule_id="galaxy_in_days",
        language="math",
        pattern="word_sequence:in_days",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["in", "for", "over"]},
                {"category": "number", "capture": "days"},
                {"word_in": ["day", "days"]},
            ],
        },
        examples=[{"input": "in 5 days", "output": "days=5"}],
    ),
    # "for [N] minutes a/per day" → capture daily minutes rate/quantity.
    GrammarRule(
        rule_id="galaxy_minutes_a_day",
        language="math",
        pattern="word_sequence:minutes_a_day",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["for"]},
                {"category": "number", "capture": "minutes"},
                {"word_in": ["minute", "minutes"]},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["day", "days"]},
            ],
        },
        examples=[{"input": "practices for 20 minutes a day", "output": "minutes=20"}],
    ),
    # "half an hour" / "half a hour" → duration extraction (minutes=30).
    GrammarRule(
        rule_id="galaxy_half_an_hour",
        language="math",
        pattern="word_sequence:half_an_hour",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["half"]},
                {"word_in": ["a", "an"]},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "spends half an hour", "output": "minutes=30"}],
    ),
    # "a/one fifth of an hour" (and similar denominators) → duration extraction (minutes=60/denom).
    GrammarRule(
        rule_id="galaxy_fraction_of_an_hour",
        language="math",
        pattern="word_sequence:fraction_of_an_hour",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["a", "an", "one"]},
                {
                    "word_in": [
                        "half",
                        "halves",
                        "third",
                        "thirds",
                        "fourth",
                        "fourths",
                        "quarter",
                        "quarters",
                        "fifth",
                        "fifths",
                        "sixth",
                        "sixths",
                        "seventh",
                        "sevenths",
                        "eighth",
                        "eighths",
                        "ninth",
                        "ninths",
                        "tenth",
                        "tenths",
                    ],
                    "capture": "denom_word",
                },
                {"word": "of"},
                {"word_in": ["a", "an"]},
                {"word_in": ["hour", "hours"]},
            ],
        },
        examples=[{"input": "a fifth of an hour", "output": "minutes=12"}],
    ),
    # "twice a day" → multiply by 2.
    GrammarRule(
        rule_id="galaxy_twice_a_day",
        language="math",
        pattern="word_sequence:twice_a_day",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "twice"},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["day", "days"]},
            ],
        },
        examples=[{"input": "twice a day", "output": "op=*2"}],
    ),
    # "each/per week" → multiply by 7 (days per week).
    GrammarRule(
        rule_id="galaxy_each_week",
        language="math",
        pattern="word_sequence:each_week",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["each", "per"]},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "each week", "output": "op=*7"}],
    ),
    # "[N] times as long" → capture duration multiplier (used with a previous duration).
    GrammarRule(
        rule_id="galaxy_times_as_long",
        language="math",
        pattern="word_sequence:times_as_long",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"category": "number", "capture": "multiplier"},
                {"word": "times"},
                {"word": "as"},
                {"word": "long"},
            ],
        },
        examples=[{"input": "three times as long", "output": "multiplier=3"}],
    ),
    # "currently [N] ..." → base quantity (often used for heights/amounts).
    GrammarRule(
        rule_id="galaxy_currently_quantity",
        language="math",
        pattern="word_sequence:currently_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["currently", "current"]},
                {"category": "number", "capture": "base"},
            ],
        },
        examples=[{"input": "she is currently 20 inches tall", "output": "base=20"}],
    ),
    # "at the rate of [R] ... every year" → capture rate-per-year.
    GrammarRule(
        rule_id="galaxy_rate_every_year",
        language="math",
        pattern="word_sequence:rate_every_year",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "rate"},
                {"word": "of"},
                {"category": "number", "capture": "rate"},
                {"word": "every"},
                {"word_in": ["year", "years"]},
            ],
        },
        examples=[{"input": "grows at the rate of 3 inches every year", "output": "rate=3"}],
    ),
    # "after/for [N] years" → capture duration in years.
    GrammarRule(
        rule_id="galaxy_after_years",
        language="math",
        pattern="word_sequence:after_years",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["after", "for", "over", "in"]},
                {"category": "number", "capture": "years"},
                {"word_in": ["year", "years"]},
            ],
        },
        examples=[{"input": "after 10 years", "output": "years=10"}],
    ),
    # "There are [N] [items] ... each [M] minutes" → total minutes term = N*M.
    GrammarRule(
        rule_id="galaxy_count_each_minutes",
        language="math",
        pattern="word_sequence:count_each_minutes",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["are", "were", "is", "was"]},
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word": "each"},
                {"category": "number", "capture": "minutes"},
                {"word_in": ["minute", "minutes"]},
            ],
        },
        examples=[{"input": "There are 20 episodes and they are each 30 minutes long", "output": "term_minutes=20*30"}],
    ),
    # "[rate] per [unit] for [N] [unit_plural]" → term = rate*N (more natural GSM8K phrasing).
    GrammarRule(
        rule_id="galaxy_rate_per_unit_for",
        language="math",
        pattern="word_sequence:rate_per_unit_for",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "rate"},
                {"word": "per"},
                {"category": "noun"},
                {"word_in": ["for", "in"]},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "5 dollars per day for 7 days", "output": "term=5*7"}],
    ),
    # "[rate] per month for [Y] years" → term = rate * (Y*12).
    GrammarRule(
        rule_id="galaxy_rate_per_month_for_years",
        language="math",
        pattern="word_sequence:rate_per_month_for_years",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"category": "number", "capture": "rate"},
                {"word": "per"},
                {"word_in": ["month", "months"]},
                {"word_in": ["for", "in"]},
                {"category": "number", "capture": "years"},
                {"word_in": ["year", "years"]},
            ],
        },
        examples=[{"input": "save 276 per month for 4 years", "output": "term=276*4*12"}],
    ),
    # "[N] (additional|extra|more) for every [M]" → ratio signal (handled in composition).
    # Example: "Jennifer purchased 6 additional for every 5 Mark bought" → op=ratio_add (6/5 applied to a nearby quantity).
    GrammarRule(
        rule_id="galaxy_for_every_additional",
        language="math",
        pattern="word_sequence:for_every_additional",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 5,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "numerator"},
                {"word_in": ["additional", "extra", "more"]},
                {"word": "for"},
                {"word": "every"},
                {"category": "number", "capture": "denominator"},
            ],
        },
        examples=[{"input": "6 additional for every 5", "output": "op=ratio_add 6/5"}],
    ),
    # "[A] ... for every [B] ..." → scale ratio signal (A/B applied to a nearby total).
    GrammarRule(
        rule_id="galaxy_for_every_ratio",
        language="math",
        pattern="word_sequence:for_every_ratio",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "numerator"},
                {"word": "for"},
                {"word": "every"},
                {"category": "number", "capture": "denominator"},
            ],
        },
        examples=[{"input": "2 grams for every 30 ml", "output": "op=ratio_scale 2/30"}],
    ),
    # "There are [N] [containers] ... each ... has [M] ..." → term = N*M.
    GrammarRule(
        rule_id="galaxy_there_are_each_has",
        language="math",
        pattern="word_sequence:there_are_each_has",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "proper_noun"],
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["are", "were", "is", "was"]},
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word": "each"},
                {"category": "noun"},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "each"},
            ],
        },
        examples=[{"input": "There are 3 bags. Each bag has 5 apples.", "output": "term=3*5"}],
    ),
    # "There are [N] [items] ... each has/have [M]" → term = N*M (more permissive, skips the noun after "each").
    GrammarRule(
        rule_id="galaxy_there_are_each_have",
        language="math",
        pattern="word_sequence:there_are_each_have",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["are", "were", "is", "was"]},
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word": "each"},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "each"},
            ],
        },
        examples=[{"input": "There are 6 people. They each have 5 bags.", "output": "term=6*5"}],
    ),
    # "There are [N] [containers] ... each contain(s)/hold(s) [M]" → term = N*M.
    GrammarRule(
        rule_id="galaxy_there_are_each_contain",
        language="math",
        pattern="word_sequence:there_are_each_contain",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"word": "there"},
                {"word_in": ["are", "were", "is", "was"]},
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word": "each"},
                {"word_in": ["contain", "contains", "hold", "holds"]},
                {"category": "number", "capture": "each"},
            ],
        },
        examples=[{"input": "There are 12 crates that each contain 150 oranges.", "output": "term=12*150"}],
    ),
    # "[P]-page letter to [N] friends" → term = P*N (used with frequency multipliers).
    GrammarRule(
        rule_id="galaxy_page_letter_to_friends",
        language="math",
        pattern="word_sequence:page_letter_to_friends",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"category": "number", "capture": "pages"},
                {"word_in": ["page", "pages"]},
                {"category": "noun"},
                {"word": "to"},
                {"category": "number", "capture": "friends"},
                {"word_in": ["friend", "friends"]},
            ],
        },
        examples=[{"input": "writes a 3-page letter to 2 different friends", "output": "term=3*2"}],
    ),
    # Pizza template: "[A] large pizzas and [B] small pizzas ... large has [C] slices ... small has [D] slices"
    GrammarRule(
        rule_id="galaxy_pizza_large_small_slices",
        language="math",
        pattern="word_sequence:pizza_large_small_slices",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 40,
            "skip_categories": ["stopword", "symbol", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "large_count"},
                {"word": "large"},
                {"word_in": ["pizza", "pizzas"]},
                {"category": "number", "capture": "small_count"},
                {"word": "small"},
                {"word_in": ["pizza", "pizzas"]},
                {"word": "large"},
                {"word_in": ["pizza", "pizzas"]},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "large_slices"},
                {"word_in": ["slice", "slices"]},
                {"word": "small"},
                {"word_in": ["pizza", "pizzas"]},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "small_slices"},
                {"word_in": ["slice", "slices"]},
            ],
        },
        examples=[{"input": "2 large pizzas ... 16 slices ... 2 small ... 8 slices", "output": "term=2*16+2*8"}],
    ),
    # "[A] and [B] ... total/altogether" → direct sum expression.
    GrammarRule(
        rule_id="galaxy_plus_total",
        language="math",
        pattern="word_sequence:plus_total",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            # Keep this extremely local: avoid cross-sentence matches like
            # "total ... 2000 and increased by 500. What is the total ...".
            "max_skip": 1,
            "skip_categories": ["stopword", "symbol"],
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word": "and"},
                {"category": "number", "capture": "b"},
                {"word_in": ["total", "altogether", "combined", "sum"]},
            ],
        },
        examples=[{"input": "3 and 5 total", "output": "3 5 +"}],
    ),
    # "altogether"/"total"/"combined" → aggregation hint.
    GrammarRule(
        rule_id="galaxy_altogether",
        language="math",
        pattern="word_sequence:altogether",
        rpn_program="",
        domain="math_aggregation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["altogether", "total", "combined", "sum"]},
            ],
        },
        examples=[{"input": "altogether", "output": "agg=sum"}],
    ),
    # "how many" → aggregation/question hint (used to permit sum patterns).
    GrammarRule(
        rule_id="galaxy_how_many_total",
        language="math",
        pattern="word_sequence:how_many",
        rpn_program="",
        domain="math_aggregation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "how"},
                {"word": "many"},
            ],
        },
        examples=[{"input": "How many", "output": "question"}],
    ),
    # "total of [N]" / "total is [N]" → capture a total label (useful for remainder problems).
    GrammarRule(
        rule_id="galaxy_total_of",
        language="math",
        pattern="word_sequence:total_of",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            # WordGalaxy sometimes tags prepositions like "in/of" as "aggregation".
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "total"},
                {"word_in": ["of", "is", "are", "was", "were"]},
                {"category": "number", "capture": "total"},
            ],
        },
        examples=[{"input": "a total of 20", "output": "total=20"}],
    ),
    # "new total ... is [N]" → long-distance total capture in expansion problems.
    GrammarRule(
        rule_id="galaxy_new_total_is",
        language="math",
        pattern="word_sequence:new_total_is",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "new"},
                {"word": "total"},
                {"word_in": ["is", "was", "are", "were"]},
                {"category": "number", "capture": "total"},
            ],
        },
        examples=[{"input": "If the new total square footage ... is 16,000", "output": "total=16000"}],
    ),
    # "total of [N] for [M] ..." → often a unit-rate ask ("each"/"per") downstream.
    GrammarRule(
        rule_id="galaxy_total_of_for_count",
        language="math",
        pattern="word_sequence:total_of_for_count",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word": "total"},
                {"word_in": ["of", "is", "are", "was", "were"]},
                {"category": "number", "capture": "total"},
                {"word": "for"},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "paid a total of 20,700 for 150 pieces", "output": "total=20700 count=150"}],
    ),
    # "requires/need(s) ... [N]" → capture additional quantities in "total of ... and requires ..."
    GrammarRule(
        rule_id="galaxy_requires_quantity",
        language="math",
        pattern="word_sequence:requires_quantity",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["requires", "require", "needs", "need", "takes", "take"]},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "requires 6 left turns", "output": "amount=6"}],
    ),
    # "recorded/reported as [N]" → capture base totals that later get adjusted.
    GrammarRule(
        rule_id="galaxy_recorded_as",
        language="math",
        pattern="word_sequence:recorded_as",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["recorded", "record", "reported", "reports", "registered"]},
                {"word_in": ["as", "at", "was", "were"]},
                {"category": "number", "capture": "value"},
            ],
        },
        examples=[{"input": "recorded as 2000", "output": "value=2000"}],
    ),
    # "increased by [N]" → addition op.
    GrammarRule(
        rule_id="galaxy_increased_by",
        language="math",
        pattern="word_sequence:increased_by",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["increased", "increase", "increases", "rose", "rises", "rise", "grew", "grow", "grows"]},
                {"word": "by"},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "increased by 500", "output": "op=+500"}],
    ),
    # "with [N] recoveries" → subtract from current total (COVID-style prompts).
    GrammarRule(
        rule_id="galaxy_with_recoveries",
        language="math",
        pattern="word_sequence:with_recoveries",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "with"},
                {"category": "number", "capture": "amount"},
                {"word_in": ["recovery", "recoveries"]},
            ],
        },
        examples=[{"input": "with 50 recoveries", "output": "op=-50"}],
    ),
    # "spiked to [N]" → treat as an added delta (new cases) for running totals.
    GrammarRule(
        rule_id="galaxy_spiked_to",
        language="math",
        pattern="word_sequence:spiked_to",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["spiked", "spike", "spikes"]},
                {"word": "to"},
                {"category": "number", "capture": "amount"},
            ],
        },
        examples=[{"input": "spiked to 1500", "output": "op=+1500"}],
    ),
    # "increased by [P]%" → multiplicative percent-increase on the running total.
    GrammarRule(
        rule_id="galaxy_increased_by_percent",
        language="math",
        pattern="word_sequence:increased_by_percent",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["increased", "increase", "increases", "rose", "rises", "rise", "grew", "grow", "grows"]},
                {"word": "by"},
                {"category": "number", "capture": "pct"},
                {"word_in": ["%", "percent"]},
            ],
        },
        examples=[{"input": "increased by 50%", "output": "op=percent_increase 50"}],
    ),
    # "tax/tip/discount [N] %" (without explicit "of") → percent-of operation on the current base.
    GrammarRule(
        rule_id="galaxy_percent_rate",
        language="math",
        pattern="word_sequence:percent_rate",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 3,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "pct"},
                {"word_in": ["%", "percent"]},
                {"word_in": ["tax", "tip", "discount", "interest", "commission", "markup"]},
            ],
        },
        examples=[{"input": "sales tax 10%", "output": "op=pct 10"}],
    ),
    # "tax ... is [N] %" / "taxes are [N] %" → percent label (used in gratuity/tax multi-step).
    GrammarRule(
        rule_id="galaxy_tax_is_percent",
        language="math",
        pattern="word_sequence:tax_is_percent",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["tax", "taxes"]},
                {"word_in": ["is", "was", "are", "were"]},
                {"category": "number", "capture": "pct"},
                {"word_in": ["%", "percent"]},
            ],
        },
        examples=[{"input": "sales tax is 10%", "output": "pct=10"}],
    ),
    # "bill/total was [N]" → capture totals that do not include "total of".
    GrammarRule(
        rule_id="galaxy_total_was",
        language="math",
        pattern="word_sequence:total_was",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun", "aggregation"],
            "word_pattern": [
                {"word_in": ["bill", "total", "amount", "cost"]},
                {"word_in": ["is", "was", "were"]},
                {"category": "number", "capture": "total"},
            ],
        },
        examples=[{"input": "The total bill was 140", "output": "total=140"}],
    ),
    # ------------------------------------------------------------------ #
    # Product terms (count × each)
    # ------------------------------------------------------------------ #
    # "[N] [containers] of [M] [items]" → term = N*M (e.g., "3 bags of 5 apples").
    GrammarRule(
        rule_id="galaxy_count_of_each",
        language="math",
        pattern="word_sequence:count_of_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word": "of"},
                {"category": "number", "capture": "each"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "3 bags of 5 apples", "output": "term=3*5"}],
    ),
    # "[N] [items] with [M] [each_unit] each" → term = N*M.
    GrammarRule(
        rule_id="galaxy_count_with_each",
        language="math",
        pattern="word_sequence:count_with_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 4,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["with", "having"]},
                {"category": "number", "capture": "each"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "2 pizzas with 16 each", "output": "term=2*16"}],
    ),
    # "[N] [items] has/have [M] [each_unit] each" → term = N*M.
    GrammarRule(
        rule_id="galaxy_count_has_each",
        language="math",
        pattern="word_sequence:count_has_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "each"},
                {"word_in": ["each", "apiece"]},
            ],
        },
        examples=[{"input": "2 boxes have 10 each", "output": "term=2*10"}],
    ),
    # "[N] [items] that/which each has/have [M] [each_unit]" → term = N*M.
    # Example: "3 classes that each have 30 students" (no trailing "each").
    GrammarRule(
        rule_id="galaxy_count_that_each_have",
        language="math",
        pattern="word_sequence:count_that_each_have",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 10,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["that", "which", "who"]},
                {"word": "each"},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "each"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "3 classes that each have 30 students", "output": "term=3*30"}],
    ),
    # "... with [N] [items]" → base quantity (common GSM8K phrasing: "a class with 50 students").
    GrammarRule(
        rule_id="galaxy_with_quantity_noun",
        language="math",
        pattern="word_sequence:with_quantity_noun",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 2,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "with"},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        examples=[{"input": "a class with 50 students", "output": "base=50"}],
    ),
    # "each of the first [N] [items] has [M]" → term = N*M.
    GrammarRule(
        rule_id="galaxy_first_n_has_each",
        language="math",
        pattern="word_sequence:first_n_has_each",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 12,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "first"},
                {"category": "number", "capture": "count"},
                {"category": "noun"},
                {"word_in": ["has", "have"]},
                {"category": "number", "capture": "each"},
            ],
        },
        examples=[{"input": "each of the first four houses has 3 gnomes", "output": "term=4*3"}],
    ),
    # ------------------------------------------------------------------ #
    # Multi-step operations with "each of N"
    # ------------------------------------------------------------------ #
    # "gave [X] ... to each of [N]" → subtract X*N (common GSM8K pattern).
    GrammarRule(
        rule_id="galaxy_gave_each_of",
        language="math",
        pattern="word_sequence:gave_each_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["gave", "gives", "give"]},
                {"category": "number", "capture": "per"},
                {"word": "to"},
                {"word": "each"},
                {"word": "of"},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "gave 3 apples to each of 4 friends", "output": "op=-12"}],
    ),
    # "got/received [X] ... from each of [N]" → add X*N.
    GrammarRule(
        rule_id="galaxy_received_each_of",
        language="math",
        pattern="word_sequence:received_each_of",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["received", "gets", "got", "gained", "earned", "found"]},
                {"category": "number", "capture": "per"},
                {"word": "from"},
                {"word": "each"},
                {"word": "of"},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "got 2 dollars from each of 5 friends", "output": "op=+10"}],
    ),
    # "paid/spent [X] ... each for [N]" → subtract X*N.
    GrammarRule(
        rule_id="galaxy_spent_each_for_count",
        language="math",
        pattern="word_sequence:spent_each_for_count",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["spent", "spends", "spend", "paid", "pays", "pay"]},
                {"category": "number", "capture": "price"},
                {"word_in": ["each", "apiece"]},
                {"word_in": ["for", "on"]},
                {"category": "number", "capture": "count"},
            ],
        },
        examples=[{"input": "paid 3 each for 4 items", "output": "op=-12"}],
    ),
    # "[N] shared among [M]" / "[N] split among [M]" → division expression.
    GrammarRule(
        rule_id="galaxy_shared_among",
        language="math",
        pattern="word_sequence:shared_among",
        rpn_program="",
        domain="math_arithmetic",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word_in": ["split", "shared", "divided"]},
                {"word_in": ["among", "between"]},
                {"category": "number", "capture": "b"},
            ],
        },
        examples=[{"input": "12 cookies shared among 3 friends", "output": "12 3 /"}],
    ),
    # "twice a week" → multiply by 2.
    GrammarRule(
        rule_id="galaxy_twice_a_week",
        language="math",
        pattern="word_sequence:twice_a_week",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "twice"},
                {"word_in": ["a", "per", "each"]},
                {"word_in": ["week", "weeks"]},
            ],
        },
        examples=[{"input": "twice a week", "output": "op=*2"}],
    ),
    # "... week ... year" → multiply by 52 (weeks per year).
    GrammarRule(
        rule_id="galaxy_week_to_year",
        language="math",
        pattern="word_sequence:week_to_year",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "proper_noun", "noun"],
            "word_pattern": [
                {"word_in": ["week", "weeks"]},
                {"word_in": ["year", "years", "annually"]},
            ],
        },
        examples=[{"input": "per week ... a year", "output": "op=*52"}],
    ),
    # ------------------------------------------------------------------ #
    # Multi-step reading (temporal / remaining)
    # ------------------------------------------------------------------ #
    GrammarRule(
        rule_id="galaxy_book_total_pages",
        language="math",
        pattern="word_sequence:book_total_pages",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word_in": ["reading", "read"]},
                {"category": "number", "capture": "total"},
                {"word_in": ["page", "pages"]},
                {"word_in": ["book"]},
            ],
        },
        examples=[{"input": "reading a 120-page book", "output": "total=120"}],
    ),
    GrammarRule(
        rule_id="galaxy_yesterday_read",
        language="math",
        pattern="word_sequence:yesterday_read",
        rpn_program="",
        domain="math_extraction",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 6,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "yesterday"},
                {"word_in": ["read", "reads"]},
                {"category": "number", "capture": "yesterday"},
                {"word_in": ["page", "pages"]},
            ],
        },
        examples=[{"input": "Yesterday she read 12 pages", "output": "yesterday=12"}],
    ),
    GrammarRule(
        rule_id="galaxy_today_twice_as_yesterday",
        language="math",
        pattern="word_sequence:today_twice_yesterday",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 8,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "today"},
                {"word_in": ["read", "reads"]},
                {"word": "twice"},
                {"word": "yesterday"},
            ],
        },
        examples=[{"input": "today she read twice ... yesterday", "output": "today=2*yesterday"}],
    ),
    GrammarRule(
        rule_id="galaxy_half_remaining",
        language="math",
        pattern="word_sequence:half_remaining",
        rpn_program="",
        domain="math_operation",
        semantics={
            "pattern_type": "word_sequence",
            "match_mode": "subsequence",
            "max_skip": 5,
            "skip_categories": ["stopword", "symbol", "noun", "proper_noun"],
            "word_pattern": [
                {"word": "half"},
                {"word": "remaining"},
            ],
        },
        examples=[{"input": "half of the remaining pages", "output": "answer=remaining/2"}],
    ),
]

# --------------------------------------------------------------------------- #
# Composition templates (stored as Grammar Galaxy entries for TRM selection)
# --------------------------------------------------------------------------- #
COMPOSITION_TEMPLATES = [
    GrammarRule(
        rule_id="template_simple_apply",
        language="math",
        pattern="composition_template:simple_apply",
        rpn_program="",
        domain="math_composition_template",
        semantics={
            "template_id": "simple_apply",
            "description": "Apply extracted base + sequential operations (no aggregation).",
        },
        examples=[{"input": "John has 15. He gave 3.", "output": "15 3 -"}],
    ),
    GrammarRule(
        rule_id="template_extract_operate_aggregate",
        language="math",
        pattern="composition_template:extract_operate_aggregate",
        rpn_program="",
        domain="math_composition_template",
        semantics={
            "template_id": "extract_operate_aggregate",
            "description": "Extract quantities, apply ops in order, aggregate if requested.",
        },
        examples=[{"input": "Natalia sold 48. Half as many in May. Altogether?", "output": "48 48 2 / +"}],
    ),
    GrammarRule(
        rule_id="template_rate_duration",
        language="math",
        pattern="composition_template:rate_duration",
        rpn_program="",
        domain="math_composition_template",
        semantics={
            "template_id": "rate_duration",
            "description": "Compose rate * duration (and then any extra multipliers).",
        },
        examples=[{"input": "Earns 5 per day for 7 days", "output": "5 7 *"}],
    ),
    GrammarRule(
        rule_id="template_distribute_and_sum",
        language="math",
        pattern="composition_template:distribute_and_sum",
        rpn_program="",
        domain="math_composition_template",
        semantics={
            "template_id": "distribute_and_sum",
            "description": "Compose multiple product terms and sum them (e.g., 2*16 + 2*8).",
        },
        examples=[{"input": "2 large(16) and 2 small(8)", "output": "2 16 * 2 8 * +"}],
    ),
    GrammarRule(
        rule_id="template_multi_step_store_recall",
        language="math",
        pattern="composition_template:multi_step_store_recall",
        rpn_program="",
        domain="math_composition_template",
        semantics={
            "template_id": "multi_step_store_recall",
            "description": "Use STORE/RECALL for multi-step derived quantities (temporal / remaining).",
        },
        examples=[
            {
                "input": "120 total, yesterday 12, today twice, tomorrow half remaining",
                "output": "120 STORE_A 12 STORE_B ... RECALL_D 2 /",
            }
        ],
    ),
]


# Calculus rules using ∑ and ∫
CALCULUS_RULES = [
    GrammarRule(
        rule_id="calc_riemann_sum",
        language="math",
        pattern="∑[i=a..b] f(i)",
        rpn_program="a RECALL b RECALL 1 - swap - 1 + { i STORE f RECALL i RECALL swap CALL } swap times +",
        domain="math_calculus",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "∑[i=1..5] i", "output": "15"}],
    ),
    GrammarRule(
        rule_id="calc_definite_integral",
        language="math",
        pattern="∫[a..b] f(x) dx",
        rpn_program="a RECALL b RECALL f RECALL TRAPEZOIDAL_INTEGRATE",
        domain="math_calculus",
        symbol_refs=[8747],  # ∫
        examples=[{"input": "∫[0..1] x² dx", "output": "0.333"}],
    ),
    GrammarRule(
        rule_id="calc_partial_derivative",
        language="math",
        pattern="∂f/∂x",
        rpn_program="f RECALL x RECALL PARTIAL_DIFF",
        domain="math_calculus",
        symbol_refs=[8706],  # ∂
        examples=[{"input": "∂(x²y)/∂x", "output": "2xy"}],
    ),
    GrammarRule(
        rule_id="calc_power_rule",
        language="math",
        pattern=r"d/dx\s*\(?\s*x\^(\d+)\s*\)?",
        rpn_program="{g0} x {g0} 1 - pow *",
        domain="math_calculus",
        symbol_refs=[],
        examples=[{"input": "d/dx x^3", "output": "3x^2"}],
    ),
    GrammarRule(
        rule_id="calc_power_integral",
        language="math",
        pattern=r"∫\s*x\^(\d+)\s*dx",
        rpn_program="x {g0} 1 + pow {g0} 1 + /",
        domain="math_calculus",
        symbol_refs=[],
        examples=[{"input": '∫ x^2 dx', "output": "x^3/3"}],
    ),
    GrammarRule(
        rule_id="calc_scaled_power_rule",
        language="math",
        pattern=r"d/dx\s*([0-9]+)x\^(\d+)",
        rpn_program="{g0} {g1} * x {g1} 1 - pow *",
        domain="math_calculus",
        symbol_refs=[],
        examples=[{"input": "d/dx 3x^4", "output": "12x^3"}],
    ),
    GrammarRule(
        rule_id="calc_scaled_power_integral",
        language="math",
        pattern=r"∫\s*([0-9]+)x\^(\d+)\s*dx",
        rpn_program="{g0} x {g1} 1 + pow {g1} 1 + / *",
        domain="math_calculus",
        symbol_refs=[],
        examples=[{"input": '∫ 2x^3 dx', "output": "x^4/2"}],
    ),
    GrammarRule(
        rule_id="calc_exp_integral",
        language="math",
        pattern=r"∫\s*e\^x\s*dx",
        rpn_program="x exp",
        domain="math_calculus",
        symbol_refs=[],
        examples=[{"input": '∫ e^x dx', "output": "e^x"}],
    ),
]


# Set theory rules using ∈, ∪, ∩
SET_THEORY_RULES = [
    GrammarRule(
        rule_id="set_membership",
        language="math",
        pattern="x ∈ S",
        rpn_program="x RECALL S RECALL CONTAINS",
        domain="math_set",
        symbol_refs=[8712],  # ∈
        examples=[{"input": "3 ∈ {1,2,3}", "output": "true"}],
    ),
    GrammarRule(
        rule_id="set_union",
        language="math",
        pattern="A ∪ B",
        rpn_program="A RECALL B RECALL UNION",
        domain="math_set",
        symbol_refs=[8746],  # ∪
        examples=[{"input": "{1,2} ∪ {2,3}", "output": "{1,2,3}"}],
    ),
    GrammarRule(
        rule_id="set_intersection",
        language="math",
        pattern="A ∩ B",
        rpn_program="A RECALL B RECALL INTERSECT",
        domain="math_set",
        symbol_refs=[8745],  # ∩
        examples=[{"input": "{1,2} ∩ {2,3}", "output": "{2}"}],
    ),
]


# Logic rules using ∀, ∃, ⇒
LOGIC_RULES = [
    GrammarRule(
        rule_id="logic_forall",
        language="math",
        pattern="∀x P(x)",
        rpn_program="DOMAIN RECALL { x STORE P RECALL x RECALL swap CALL } ALL",
        domain="math_logic",
        symbol_refs=[8704],  # ∀
        examples=[{"input": "∀x∈ℕ: x≥0", "output": "true"}],
    ),
    GrammarRule(
        rule_id="logic_exists",
        language="math",
        pattern="∃x P(x)",
        rpn_program="DOMAIN RECALL { x STORE P RECALL x RECALL swap CALL } ANY",
        domain="math_logic",
        symbol_refs=[8707],  # ∃
        examples=[{"input": "∃x∈ℕ: x>10", "output": "true"}],
    ),
    GrammarRule(
        rule_id="logic_implies",
        language="math",
        pattern="P ⇒ Q",
        rpn_program="P RECALL neg Q RECALL or",
        domain="math_logic",
        symbol_refs=[8658],  # ⇒
        examples=[{"input": "rain ⇒ wet", "output": "¬rain ∨ wet"}],
    ),
]


# Statistics rules using ∑ (cross-domain)
STATISTICS_RULES = [
    GrammarRule(
        rule_id="stat_expected_value",
        language="math",
        pattern="E[X] = ∑ xᵢP(xᵢ)",
        rpn_program="VALUES RECALL PROBS RECALL { * } zipwith +",
        domain="math_statistics",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "E[dice]", "output": "3.5"}],
    ),
    GrammarRule(
        rule_id="stat_variance",
        language="math",
        pattern="Var[X] = ∑ (xᵢ-μ)²P(xᵢ)",
        rpn_program="VALUES RECALL MU RECALL { swap - 2 pow } map PROBS RECALL { * } zipwith +",
        domain="math_statistics",
        symbol_refs=[8721, 956],  # ∑, μ
        examples=[{"input": "Var[dice]", "output": "2.917"}],
    ),
]

# Linear algebra rules
LINEAR_ALGEBRA_RULES = [
    GrammarRule(
        rule_id="la_determinant_2x2",
        language="math",
        pattern=r"det\(\[\[([a-z]),([a-z])\],\[(?:[a-z]),([a-z])\]\]\)",
        rpn_program="{g0} {g3} * {g1} {g2} * -",
        domain="math_linear_algebra",
        symbol_refs=[],
        examples=[{"input": "det([[a,b],[c,d]])", "output": "ad-bc"}],
    ),
    GrammarRule(
        rule_id="la_matrix_inverse_generic",
        language="math",
        pattern=r"inverse\s*of\s*(\d+)x(\d+)",
        rpn_program="matrix_inverse",
        domain="math_linear_algebra",
        symbol_refs=[],
        examples=[{"input": "inverse of 2x2", "output": "adj/det"}],
    ),
]


# Finance rules using ∑ (cross-domain)
FINANCE_RULES = [
    GrammarRule(
        rule_id="fin_npv",
        language="math",
        pattern="NPV = ∑ CFₜ/(1+r)ᵗ",
        rpn_program="CASHFLOWS RECALL RATE RECALL { t STORE 1 RATE + t pow / } mapi +",
        domain="math_finance",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "NPV([100,100,100], 0.1)", "output": "248.69"}],
    ),
    GrammarRule(
        rule_id="fin_compound_interest",
        language="english",
        pattern=r"compound\s*interest.*principal\s*(\d+).*rate\s*([0-9.]+)%.*years?\s*(\d+)",
        rpn_program="{g0} 1 {g1} 100 / + {g2} pow *",
        domain="math_finance",
        symbol_refs=[],
        examples=[{"input": "compound interest on principal 1000 at rate 5% for 2 years", "output": "1102.5"}],
    ),
    GrammarRule(
        rule_id="fin_simple_interest",
        language="english",
        pattern=r"simple\s*interest.*principal\s*(\d+).*rate\s*([0-9.]+)%.*years?\s*(\d+)",
        rpn_program="{g0} {g0} {g1} 100 / * {g2} * +",
        domain="math_finance",
        symbol_refs=[],
        examples=[{"input": "simple interest on principal 1000 at rate 5% for 2 years", "output": "1100"}],
    ),
    GrammarRule(
        rule_id="fin_present_value",
        language="english",
        pattern=r"present\s*value.*future\s*(\d+).*rate\s*([0-9.]+)%.*years?\s*(\d+)",
        rpn_program="{g0} 1 {g1} 100 / + {g2} pow /",
        domain="math_finance",
        symbol_refs=[],
        examples=[{"input": "present value of future 1000 at rate 5% for 2 years", "output": "907.03"}],
    ),
    GrammarRule(
        rule_id="fin_continuous_compound",
        language="english",
        pattern=r"continuous\s*compounding?.*principal\s*(\d+).*rate\s*([0-9.]+)%.*years?\s*(\d+)",
        rpn_program="{g0} {g1} {g2} * 100 / exp *",
        domain="math_finance",
        symbol_refs=[],
        examples=[{"input": "continuous compounding principal 1000 at 5% for 2 years", "output": "1105.17"}],
    ),
    GrammarRule(
        rule_id="fin_annuity_future_value",
        language="english",
        pattern=r"future\s*value\s*of\s*annuity\s*payment\s*(\d+).*rate\s*([0-9.]+)%.*periods?\s*(\d+)",
        rpn_program="{g0} 1 {g1} 100 / + {g2} pow 1 - {g1} 100 / / *",
        domain="math_finance",
        symbol_refs=[],
        examples=[{"input": "future value of annuity payment 500 at 6% for 10 periods", "output": "\"FV\""}],
    ),
]

# =============================================================================
# WORD PROBLEM ARITHMETIC RULES (for GSM8K, grade school math)
# =============================================================================
WORD_PROBLEM_RULES = [
    # Addition patterns
    GrammarRule(
        rule_id="wp_addition_more",
        language="english",
        pattern="X more than Y",
        rpn_program="Y X +",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[
            {"input": "3 more than 5", "output": "8"},
            {"input": "gave her 3 more", "output": "base 3 +"},
        ],
    ),
    GrammarRule(
        rule_id="wp_addition_total",
        language="english",
        pattern="X and Y total/altogether",
        rpn_program="X Y +",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[
            {"input": "5 and 3 altogether", "output": "8"},
            {"input": "sold in April and May total", "output": "april may +"},
        ],
    ),
    GrammarRule(
        rule_id="wp_addition_combined",
        language="english",
        pattern="X combined with Y",
        rpn_program="X Y +",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "savings combined with gift", "output": "savings gift +"}],
    ),
    # Subtraction patterns
    GrammarRule(
        rule_id="wp_subtraction_less",
        language="english",
        pattern="X less than Y",
        rpn_program="Y X -",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "3 less than 10", "output": "7"}],
    ),
    GrammarRule(
        rule_id="wp_subtraction_remaining",
        language="english",
        pattern="X remaining after Y",
        rpn_program="X Y -",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "100 remaining after spending 30", "output": "70"}],
    ),
    GrammarRule(
        rule_id="wp_subtraction_difference",
        language="english",
        pattern="difference between X and Y",
        rpn_program="X Y - abs",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "difference between 10 and 7", "output": "3"}],
    ),
    # Multiplication patterns
    GrammarRule(
        rule_id="wp_multiplication_times",
        language="english",
        pattern="X times Y",
        rpn_program="X Y *",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "3 times 4", "output": "12"}],
    ),
    GrammarRule(
        rule_id="wp_multiplication_twice",
        language="english",
        pattern="twice X",
        rpn_program="X 2 *",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "twice 15", "output": "30"}],
    ),
    GrammarRule(
        rule_id="wp_multiplication_double",
        language="english",
        pattern="double X",
        rpn_program="X 2 *",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "double the amount", "output": "amount 2 *"}],
    ),
    GrammarRule(
        rule_id="wp_multiplication_triple",
        language="english",
        pattern="triple X",
        rpn_program="X 3 *",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "triple the price", "output": "price 3 *"}],
    ),
    GrammarRule(
        rule_id="wp_multiplication_per",
        language="english",
        pattern="X per Y for Z",
        rpn_program="X Z * Y /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "$12 per hour for 50 minutes", "output": "12 50 * 60 /"}],
    ),
    # Division patterns
    GrammarRule(
        rule_id="wp_division_half",
        language="english",
        pattern="half of X",
        rpn_program="X 2 /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[
            {"input": "half of 48", "output": "24"},
            {"input": "half as many", "output": "X 2 /"},
        ],
    ),
    GrammarRule(
        rule_id="wp_division_third",
        language="english",
        pattern="a third of X",
        rpn_program="X 3 /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "a third of 90", "output": "30"}],
    ),
    GrammarRule(
        rule_id="wp_division_quarter",
        language="english",
        pattern="a quarter of X",
        rpn_program="X 4 /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "a quarter of 100", "output": "25"}],
    ),
    GrammarRule(
        rule_id="wp_division_split",
        language="english",
        pattern="X split/divided among Y",
        rpn_program="X Y /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "100 split among 4 people", "output": "25"}],
    ),
    GrammarRule(
        rule_id="wp_division_each",
        language="english",
        pattern="X each from Y total",
        rpn_program="Y X /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "$5 each from $100", "output": "20"}],
    ),
    # Percentage patterns
    GrammarRule(
        rule_id="wp_percentage_of",
        language="english",
        pattern="X% of Y",
        rpn_program="Y X * 100 /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[
            {"input": "20% of 50", "output": "10"},
            {"input": "50% of 100", "output": "50"},
        ],
    ),
    GrammarRule(
        rule_id="wp_percentage_discount",
        language="english",
        pattern="Y with X% discount",
        rpn_program="Y Y X * 100 / -",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "$100 with 20% discount", "output": "80"}],
    ),
    GrammarRule(
        rule_id="wp_percentage_increase",
        language="english",
        pattern="Y increased by X%",
        rpn_program="Y Y X * 100 / +",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "50 increased by 10%", "output": "55"}],
    ),
    # Ratio patterns
    GrammarRule(
        rule_id="wp_ratio",
        language="english",
        pattern="ratio of X to Y",
        rpn_program="X Y /",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "ratio of 10 to 5", "output": "2"}],
    ),
    # Multi-step chain patterns
    GrammarRule(
        rule_id="wp_chain_then",
        language="english",
        pattern="X then Y",
        rpn_program="X Y",
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "add 5 then multiply by 2", "output": "5 + 2 *"}],
    ),
]

# =============================================================================
# ALGEBRA RULES (multi-step via STORE/RECALL)
# =============================================================================


def _compose_quadratic(m: re.Match[str]) -> str:
    """
    Compose RPN for quadratic formula using STORE/RECALL.

    Pattern targets: x^2 ± bx ± c = 0 (a=1).
    Emits one root (x1); the other root can be obtained by swapping +/-
    at the sqrt step, but returning a single valid root is enough for baseline.
    """
    sign1 = m.group(1)  # + or -
    b = m.group(2)
    sign2 = m.group(3)
    c = m.group(4)

    b_val = f"-{b}" if sign1 == "-" else b
    c_val = f"-{c}" if sign2 == "-" else c

    rpn = f"1 STORE_A {b_val} STORE_B {c_val} STORE_C "
    # Tier-3 programmable surface currently doesn't expose `pow`; use `DUP *` for squaring.
    rpn += "RECALL_B DUP * RECALL_A RECALL_C * 4 * - STORE_D "
    rpn += "RECALL_B neg RECALL_D sqrt + RECALL_A 2 * /"
    return rpn.strip()


def _compose_linear(m: re.Match[str]) -> str:
    """Compose RPN for linear equation: ax ± b = c -> x = (c - b) / a."""
    a = m.group(1)
    sign = m.group(2)
    b = m.group(3)
    c = m.group(4)

    b_val = f"-{b}" if sign == "-" else b
    return f"{c} {b_val} - {a} /"


ALGEBRA_RULES = [
    GrammarRule(
        rule_id="quadratic_standard_form",
        language="math",
        pattern=r"x\^2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)\s*=\s*0",
        rpn_program=_compose_quadratic,
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "x^2 - 5x + 6 = 0", "output": "2, 3"}],
    ),
    GrammarRule(
        rule_id="linear_equation",
        language="math",
        pattern=r"(\d+)x\s*([+\-])\s*(\d+)\s*=\s*(\d+)",
        rpn_program=_compose_linear,
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "3x + 5 = 20", "output": "5"}],
    ),
]

# =============================================================================
# GSM8K-SPECIFIC TEMPLATES (regex -> executable RPN)
# =============================================================================

GSM8K_TEMPLATES = [
    GrammarRule(
        rule_id="gsm_times_total",
        language="english",
        pattern=r"(?:has|had|sold|made|owns|bought|collected)\\s+(?:\\w+\\s+)?(\\d+).*?(\\d+)\\s*times\\s*(?:as many|that).*?(?:total|altogether|all)",
        rpn_program="{g0} {g0} {g1} * +",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_cost_more_total",
        language="english",
        pattern=r"\\$(\d+).*?\\$(\d+)\s*more.*?(?:total|cost)",
        rpn_program="{g0} {g0} {g1} + +",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_divide_equally",
        language="english",
        pattern=r"(\\d+)\\s+(?:\\w+\\s+)?(?:divided|split|shared)\\s+(?:equally\\s+)?(?:among|between|into)\\s+(\\d+)",
        rpn_program="{g0} {g1} /",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_rate_multiplication",
        language="english",
        pattern=r"(\d+)\s*(?:per|each|every).*?(\d+)",
        rpn_program="{g0} {g1} *",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_start_gain_lose",
        language="english",
        pattern=r"(?:started|began|had)\s*(?:with)?\s*(\d+).*?(?:gained|got|received)\s*(\d+).*?(?:lost|spent|gave)\s*(\d+)",
        rpn_program="{g0} {g1} + {g2} -",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_buy_each_total",
        language="english",
        pattern=r"(?:bought|purchased)\s*(\d+).*?\\$(\d+)\s*(?:each|per)",
        rpn_program="{g0} {g1} *",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_distance_speed_time",
        language="english",
        pattern=r"(\d+)\s*(?:miles|km).*?(?:per hour|mph).*?(\d+)\s*(?:hours?)",
        rpn_program="{g0} {g1} *",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_average_of_two",
        language="english",
        pattern=r"average\s*of\s*(\d+)\s*and\s*(\d+)",
        rpn_program="{g0} {g1} + 2 /",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_sum_first_n",
        language="english",
        pattern=r"sum\s+of\s+first\s+(\d+)\s+(?:integers|numbers)",
        rpn_program="{g0} {g0} 1 + * 2 /",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
    GrammarRule(
        rule_id="gsm_half_altogether_v2",
        language="english",
        pattern=r"(?:sold|made|had)\\s+(?:\\w+\\s+)?(\\d+).*?half\\s+(?:as many|that many).*?(?:altogether|total|all)",
        rpn_program="{g0} {g0} 2 / +",
        domain="math_word_problem",
        symbol_refs=[],
        examples=[],
    ),
]

# =============================================================================
# SYMBOLIC MATH RULES (for competition-style problems)
# =============================================================================
SYMBOLIC_RULES = [
    GrammarRule(
        rule_id="sym_fraction",
        language="math",
        pattern="\\frac{a}{b}",
        rpn_program="a b /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\frac{3}{4}", "output": "0.75"}],
    ),
    GrammarRule(
        rule_id="sym_power",
        language="math",
        pattern="a^{b}",
        rpn_program="a b pow",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "2^{3}", "output": "8"}],
    ),
    GrammarRule(
        rule_id="sym_square",
        language="math",
        pattern="x^2",
        rpn_program="x 2 pow",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "5^2", "output": "25"}],
    ),
    GrammarRule(
        rule_id="sym_sqrt",
        language="math",
        pattern="\\sqrt{x}",
        rpn_program="x sqrt",
        domain="math_algebra",
        symbol_refs=[8730],  # √
        examples=[{"input": "\\sqrt{16}", "output": "4"}],
    ),
    GrammarRule(
        rule_id="sym_nth_root",
        language="math",
        pattern="\\sqrt[n]{x}",
        rpn_program="x 1 n / pow",
        domain="math_algebra",
        symbol_refs=[8730],
        examples=[{"input": "\\sqrt[3]{8}", "output": "2"}],
    ),
    GrammarRule(
        rule_id="sym_abs",
        language="math",
        pattern="|x|",
        rpn_program="x abs",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "|-5|", "output": "5"}],
    ),
    GrammarRule(
        rule_id="sym_mod",
        language="math",
        pattern="a mod b",
        rpn_program="a b mod",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "17 mod 5", "output": "2"}],
    ),
    GrammarRule(
        rule_id="sym_mod_equiv",
        language="math",
        pattern="a ≡ b (mod n)",
        rpn_program="a n mod b n mod eq",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "17 ≡ 2 (mod 5)", "output": "true"}],
    ),
    GrammarRule(
        rule_id="sym_factorial",
        language="math",
        pattern="n!",
        rpn_program="n factorial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "5!", "output": "120"}],
    ),
    GrammarRule(
        rule_id="sym_binomial",
        language="math",
        pattern="\\binom{n}{k}",
        rpn_program="n k binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "\\binom{5}{2}", "output": "10"}],
    ),
    GrammarRule(
        rule_id="sym_log",
        language="math",
        pattern="\\log_{b}(x)",
        rpn_program="x log b log /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\log_{2}(8)", "output": "3"}],
    ),
    GrammarRule(
        rule_id="sym_ln",
        language="math",
        pattern="\\ln(x)",
        rpn_program="x log",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\ln(e)", "output": "1"}],
    ),
    GrammarRule(
        rule_id="sym_sin",
        language="math",
        pattern="\\sin(x)",
        rpn_program="x sin",
        domain="math_trig",
        symbol_refs=[],
        examples=[{"input": "\\sin(0)", "output": "0"}],
    ),
    GrammarRule(
        rule_id="sym_cos",
        language="math",
        pattern="\\cos(x)",
        rpn_program="x cos",
        domain="math_trig",
        symbol_refs=[],
        examples=[{"input": "\\cos(0)", "output": "1"}],
    ),
    GrammarRule(
        rule_id="sym_quadratic",
        language="math",
        pattern="x = \\frac{-b ± \\sqrt{b^2-4ac}}{2a}",
        rpn_program="b neg b 2 pow 4 a * c * - sqrt + 2 a * /",
        domain="math_algebra",
        symbol_refs=[8730, 177],  # √, ±
        examples=[{"input": "x^2 + 5x + 6 = 0", "output": "x = -2, -3"}],
    ),
]

# =============================================================================
# COMPETITION MATH RULES (AMC, AIME, MATH dataset patterns)
# =============================================================================
COMPETITION_MATH_RULES = [
    GrammarRule(
        rule_id="comp_permutation_arrange",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|arrangements?) (?:to |can )?(arrange|order|line up) (\\d+)",
        rpn_program="{g1} factorial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to arrange 5 people?", "output": "120"},
            {"input": "Number of arrangements of 7 items", "output": "5040"},
        ],
    ),
    GrammarRule(
        rule_id="comp_permutation_pnk",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|permutations?) (?:to )?(?:choose|select|pick) (\\d+) from (\\d+) (?:where |when )?order matters",
        rpn_program="{g1} factorial {g1} {g0} - factorial /",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to choose 3 from 10 where order matters?", "output": "720"},
        ],
    ),
    GrammarRule(
        rule_id="comp_combination_choose",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|combinations?) (?:to )?(?:choose|select|pick) (\\d+) from (\\d+)",
        rpn_program="{g1} {g0} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to choose 3 from 10?", "output": "120"},
            {"input": "Number of combinations selecting 5 from 20", "output": "15504"},
        ],
    ),
    GrammarRule(
        rule_id="comp_combination_committee",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|committees?) (?:to )?(?:form|create|select) a (?:committee|group|team) of (\\d+) from (\\d+)",
        rpn_program="{g1} {g0} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to form a committee of 4 from 12?", "output": "495"},
        ],
    ),
    GrammarRule(
        rule_id="comp_combination_handshakes",
        language="english",
        pattern=r"(\\d+) people (?:shake hands|meet|greet)",
        rpn_program="{g0} 2 binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "10 people shake hands, how many handshakes?", "output": "45"},
        ],
    ),
    GrammarRule(
        rule_id="comp_binomial_coefficient",
        language="english",
        pattern=r"coefficient of x\\^(\\d+) in \\(1\\+x\\)\\^(\\d+)",
        rpn_program="{g1} {g0} binomial",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "coefficient of x^3 in (1+x)^10", "output": "120"}],
    ),
    GrammarRule(
        rule_id="comp_binomial_expansion_term",
        language="english",
        pattern=r"(\\d+)(?:th|st|nd|rd) term (?:in |of )?\\(.*\\)\\^(\\d+)",
        rpn_program="{g1} {g0} 1 - binomial",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "4th term in (a+b)^7", "output": "35"}],
    ),
    GrammarRule(
        rule_id="comp_probability_exactly_k",
        language="english",
        pattern=r"probability (?:of )?(?:exactly )?(\\d+) successes? in (\\d+) trials? with p=([0-9.]+)",
        rpn_program="{g1} {g0} binomial {g2} {g0} pow {g2} 1 swap - {g1} {g0} - pow * *",
        domain="math_probability",
        symbol_refs=[],
        examples=[
            {"input": "probability of exactly 3 successes in 10 trials with p=0.5", "output": "0.1172"},
        ],
    ),
    GrammarRule(
        rule_id="comp_divisors_count",
        language="english",
        pattern=r"(?:how many|number of) (?:positive )?divisors of (\\d+)",
        rpn_program="{g0} PRIME_FACTORIZE { 1 + } map *",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "How many divisors of 12?", "output": "6"}],
    ),
    GrammarRule(
        rule_id="comp_gcd",
        language="english",
        pattern=r"(?:gcd|greatest common divisor|hcf) (?:of )?(\\d+) and (\\d+)",
        rpn_program="{g0} {g1} GCD",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "GCD of 24 and 36", "output": "12"}],
    ),
    GrammarRule(
        rule_id="comp_lcm",
        language="english",
        pattern=r"(?:lcm|least common multiple) (?:of )?(\\d+) and (\\d+)",
        rpn_program="{g0} {g1} * {g0} {g1} GCD /",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "LCM of 4 and 6", "output": "12"}],
    ),
    GrammarRule(
        rule_id="comp_arithmetic_sum",
        language="english",
        pattern=r"sum (?:of )?(?:first )?(\\d+) (?:positive )?integers",
        rpn_program="{g0} {g0} 1 + * 2 /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "sum of first 100 integers", "output": "5050"}],
    ),
    GrammarRule(
        rule_id="comp_geometric_sum",
        language="english",
        pattern=r"sum (?:of )?geometric series (?:with )?a=([0-9.]+),? r=([0-9.]+),? n=(\\d+)",
        rpn_program="{g0} 1 {g1} {g2} pow - * 1 {g1} - /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "sum of geometric series with a=1, r=2, n=5", "output": "31"}],
    ),
    GrammarRule(
        rule_id="comp_latex_binom",
        language="math",
        pattern=r"\\binom\\{(\\d+)\\}\\{(\\d+)\\}",
        rpn_program="{g0} {g1} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "\\binom{10}{3}", "output": "120"}],
    ),
    GrammarRule(
        rule_id="comp_latex_frac",
        language="math",
        pattern=r"\\frac{([^}]+)}{([^}]+)}",
        rpn_program="{g0} {g1} /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\frac{6}{2}", "output": "3"}],
    ),
    GrammarRule(
        rule_id="comp_latex_sqrt",
        language="math",
        pattern=r"\\sqrt{([^}]+)}",
        rpn_program="{g0} sqrt",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\sqrt{16}", "output": "4"}],
    ),
    GrammarRule(
        rule_id="comp_latex_factorial",
        language="math",
        pattern=r"(\\d+)!",
        rpn_program="{g0} factorial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "5!", "output": "120"}],
    ),
    GrammarRule(
        rule_id="comp_mod_remainder",
        language="english",
        pattern=r"(?:what is )?(?:the )?remainder (?:when )?(\\d+) (?:is )?divided by (\\d+)",
        rpn_program="{g0} {g1} mod",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "remainder when 17 divided by 5", "output": "2"}],
    ),
    GrammarRule(
        rule_id="comp_mod_congruence",
        language="math",
        pattern=r"(\\d+) ≡ \\? \\(mod (\\d+)\\)",
        rpn_program="{g0} {g1} mod",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "17 ≡ ? (mod 5)", "output": "2"}],
    ),
    # ===== ADDITIONAL COMPETITION PATTERNS =====
    GrammarRule(
        rule_id="comp_floor_latex",
        language="math",
        pattern=r"\\lfloor\\s*([^\\\\]+)\\s*\\rfloor",
        rpn_program="{g0} floor",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\lfloor 3.7 \\rfloor", "output": "3"}],
    ),
    GrammarRule(
        rule_id="comp_ceil_latex",
        language="math",
        pattern=r"\\lceil\\s*([^\\\\]+)\\s*\\rceil",
        rpn_program="{g0} ceil",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "\\lceil 3.2 \\rceil", "output": "4"}],
    ),
    GrammarRule(
        rule_id="comp_totient",
        language="english",
        pattern=r"(?:euler'?s? )?(?:totient|phi) (?:function )?(?:of )?(\\d+)",
        rpn_program="{g0} euler_totient",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "totient of 12", "output": "4"}],
    ),
    GrammarRule(
        rule_id="comp_pmod_latex",
        language="math",
        pattern=r"(\\d+)\\s*\\\\equiv\\s*(\\d+)\\s*\\\\pmod\\{(\\d+)\\}",
        rpn_program="{g0} {g2} mod {g1} eq",
        domain="math_number_theory",
        symbol_refs=[8801],
        examples=[{"input": "17 \\equiv 2 \\pmod{5}", "output": "true"}],
    ),
    GrammarRule(
        rule_id="comp_prime_count",
        language="english",
        pattern=r"(?:how many|number of) primes? (?:less than|below|up to) (\\d+)",
        rpn_program="{g0} prime_count",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[{"input": "how many primes less than 10", "output": "4"}],
    ),
    GrammarRule(
        rule_id="comp_sum_squares",
        language="english",
        pattern=r"sum (?:of )?(?:first )?(\\d+) (?:perfect )?squares",
        rpn_program="{g0} {g0} 1 + * {g0} 2 * 1 + * 6 /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "sum of first 10 squares", "output": "385"}],
    ),
    GrammarRule(
        rule_id="comp_sum_cubes",
        language="english",
        pattern=r"sum (?:of )?(?:first )?(\\d+) cubes",
        rpn_program="{g0} {g0} 1 + * 2 / 2 pow",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "sum of first 10 cubes", "output": "3025"}],
    ),
    GrammarRule(
        rule_id="comp_triangular",
        language="english",
        pattern=r"(\\d+)(?:th|st|nd|rd) triangular number",
        rpn_program="{g0} {g0} 1 + * 2 /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[{"input": "10th triangular number", "output": "55"}],
    ),
    GrammarRule(
        rule_id="comp_fibonacci",
        language="english",
        pattern=r"(\\d+)(?:th|st|nd|rd) fibonacci number",
        rpn_program="{g0} fibonacci",
        domain="math_sequences",
        symbol_refs=[],
        examples=[{"input": "10th fibonacci number", "output": "55"}],
    ),
]


def get_all_math_rules() -> list:
    """Get all math grammar rules."""
    return (
        SOVEREIGN_MATH_RULES
        + ALGEBRA_RULES
        + GSM8K_TEMPLATES
        + CALCULUS_RULES
        + SET_THEORY_RULES
        + LOGIC_RULES
        + STATISTICS_RULES
        + LINEAR_ALGEBRA_RULES
        + FINANCE_RULES
        + WORD_PROBLEM_RULES
        + SYMBOLIC_RULES
        + COMPETITION_MATH_RULES
    )


def register_with_discovery_layer():
    """Register all math rules with the Discovery Layer."""
    from knowledge3d.cranium.discovery_layer import DiscoveryLayer

    discovery = DiscoveryLayer()

    for rule in get_all_math_rules():
        discovery.register_rule(
            rule_id=rule.rule_id, domain=rule.domain, symbol_refs=rule.symbol_refs
        )

    return discovery
