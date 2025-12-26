"""
Curated math templates with parametric capture groups.

Each template:
- Regex pattern with numeric captures
- RPN program with {i} placeholders
- Domain tag for routing
"""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


def get_gsm8k_templates() -> List[GrammarRule]:
    """Base GSM8K-focused templates (concise set)."""
    return [
        # === HALF / DOUBLE PATTERNS ===
        GrammarRule(
            rule_id="gsm_half_altogether",
            language="math",
            pattern=r"(\d+\.?\d*).*?half\s*(?:as many|that many|as much).*?(?:altogether|total|in all)",
            rpn_program="{0} DUP 2 / +",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_half_of",
            language="math",
            pattern=r"half\s*(?:of|as many as)?\s*(\d+\.?\d*)",
            rpn_program="{0} 2 /",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_twice_as_many",
            language="math",
            pattern=r"twice\s*(?:as many|as much|that)\s*(?:as)?\s*(\d+\.?\d*)",
            rpn_program="{0} 2 *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_double",
            language="math",
            pattern=r"double\s*(?:of|the)?\s*(\d+\.?\d*)",
            rpn_program="{0} 2 *",
            domain="math_arithmetic",
        ),
        # === RATE × TIME PATTERNS ===
        GrammarRule(
            rule_id="gsm_hourly_minutes",
            language="math",
            pattern=r"\$?(\d+\.?\d*)\s*(?:an|per)\s*hour.*?(\d+)\s*minutes?",
            rpn_program="{0} 60 / {1} *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_per_day_week",
            language="math",
            pattern=r"(\d+)\s*(?:per|a|each)\s*day.*?(\d+)\s*days?\s*(?:a|per)?\s*week",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_per_week_year",
            language="math",
            pattern=r"(\d+)\s*(?:per|a|each)\s*week.*?(?:year|annually)",
            rpn_program="{0} 52 *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_times_week",
            language="math",
            pattern=r"(\d+)\s*times?\s*(?:a|per|each)\s*week.*?(\d+\.?\d*)\s*(?:each|per|meters?|items?)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # === SUBTRACTION / REMAINDER PATTERNS ===
        GrammarRule(
            rule_id="gsm_total_minus_two",
            language="math",
            pattern=r"(\d+\.?\d*).*?(?:eats?|uses?|gives?|spends?)\s*(\d+\.?\d*).*?(?:and|also|plus)\s*(?:another)?\s*(\d+\.?\d*)",
            rpn_program="{0} {1} - {2} -",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_remainder_times_price",
            language="math",
            pattern=r"(\d+\.?\d*).*?(?:sells?|remaining|left).*?\$?(\d+\.?\d*)\s*(?:per|each)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_spent_remaining",
            language="math",
            # Avoid matching generic "gave" (often multi-step with multiple recipients).
            pattern=r"\$?(\d+\.?\d*).*?(?:spent|used)\s*\$?(\d+\.?\d*)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
        # === MULTIPLICATION CHAINS ===
        GrammarRule(
            rule_id="gsm_a_times_b",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:times|×|x)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_a_times_b_times_c",
            language="math",
            # Require explicit multiplication markers between all operands (avoid matching any 3 numbers).
            pattern=r"(\d+\.?\d*)\s*(?:times|×|x)\s*(\d+\.?\d*)\s*(?:times|×|x)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} * {2} *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_items_per_each",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:items?|pages?|cups?).*?(\d+\.?\d*)\s*(?:people|friends?|chickens?)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # === PERCENTAGE PATTERNS ===
        GrammarRule(
            rule_id="gsm_percent_of",
            language="math",
            pattern=r"(\d+\.?\d*)\s*%\s*(?:of)\s*(\d+\.?\d*)",
            rpn_program="{1} {0} 100 / *",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_increased_percent",
            language="math",
            pattern=r"(\d+\.?\d*).*?increased\s*(?:by)?\s*(\d+\.?\d*)\s*%",
            rpn_program="{0} {0} {1} 100 / * +",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_discount_percent",
            language="math",
            pattern=r"(\d+\.?\d*).*?(?:discount|off|reduced)\s*(?:of|by)?\s*(\d+\.?\d*)\s*%",
            rpn_program="{0} {0} {1} 100 / * -",
            domain="math_arithmetic",
        ),
        # === DIVISION PATTERNS ===
        GrammarRule(
            rule_id="gsm_divided_by",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:divided by|÷|/)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} /",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_split_equally",
            language="math",
            pattern=r"(\d+\.?\d*).*?(?:split|divided|shared)\s*(?:equally|evenly)?\s*(?:among|between)?\s*(\d+)",
            rpn_program="{0} {1} /",
            domain="math_arithmetic",
        ),
        # === ADDITION PATTERNS ===
        GrammarRule(
            rule_id="gsm_plus",
            language="math",
            # Avoid bare "and" (too ambiguous in word problems) – rely on total/altogether rules instead.
            pattern=r"(\d+\.?\d*)\s*(?:plus|\+|added to)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} +",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="gsm_total_of_two",
            language="math",
            # Prefer explicit joiners to reduce grabbing unrelated numbers.
            pattern=r"(\d+\.?\d*)\s*(?:and|\+)\s*(\d+\.?\d*).*?(?:total|altogether|combined|sum)",
            rpn_program="{0} {1} +",
            domain="math_arithmetic",
        ),
        # === MULTI-STEP / FINANCE ===
        GrammarRule(
            rule_id="gsm_buy_repair_percent",
            language="math",
            pattern=r"(?:buys?|bought)\s*.*?\$?(\d+,?\d*).*?(?:repairs?|renovations?)\s*\$?(\d+,?\d*).*?(\d+\.?\d*)\s*%",
            rpn_program="{0} 1 {2} 100 / + * {0} {1} + -",
            domain="math_finance",
        ),
        GrammarRule(
            rule_id="gsm_need_have_give",
            language="math",
            pattern=r"(?:costs?|needs?)\s*\$?(\d+\.?\d*).*?(?:has?|have)\s*\$?(\d+\.?\d*).*?(?:gave?|gives?)\s*\$?(\d+\.?\d*)",
            rpn_program="{0} {1} - {2} -",
            domain="math_arithmetic",
        ),
    ]


def get_expanded_templates() -> List[GrammarRule]:
    """Expanded templates with synonym-rich variants for GSM8K-style problems."""
    templates: List[GrammarRule] = []

    # HALF variants with flexible endings/middles
    half_endings = [
        r"(?:altogether|total|in all|combined|in total|all together)",
        r"(?:how many|what is|find)",
        r"(?:did (?:she|he|they) (?:sell|make|have|get))",
    ]
    half_middles = [
        r"half\s*(?:as many|as much|that many|that much|of that)",
        r"(?:sold|made|had|got)\s*half",
        r"half\s*(?:of|from)\s*(\d+\.?\d*)",
    ]
    for middle in half_middles:
        for ending in half_endings:
            templates.append(
                GrammarRule(
                    rule_id=f"gsm_half_variant_{len(templates)}",
                    language="math",
                    pattern=rf"(\d+\.?\d*).*?{middle}.*?{ending}",
                    rpn_program="{0} DUP 2 / +",
                    domain="math_arithmetic",
                )
            )

    # Purchase / cost patterns
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_cost_each_bought",
                language="math",
                pattern=r"(?:each|per|every)\s*.*?(?:costs?|\$)\s*(\d+\.?\d*).*?(?:bought|purchased|got|buys?)\s*(\d+)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_price_quantity",
                language="math",
                pattern=r"\$?(\d+\.?\d*)\s*(?:each|per|apiece).*?(\d+)\s*(?:items?|pieces?|units?|of them)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_bought_at_price",
                language="math",
                pattern=r"(?:bought|purchased|got)\s*(\d+).*?\$?(\d+\.?\d*)\s*(?:each|per|apiece)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
        ]
    )

    # Per-day/week/month patterns
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_per_day_for_days",
                language="math",
                pattern=r"(\d+\.?\d*)\s*(?:per|each|a|every)\s*day.*?(?:for)?\s*(\d+)\s*days?",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_per_week_for_weeks",
                language="math",
                pattern=r"(\d+\.?\d*)\s*(?:per|a|each|every)\s*week.*?(?:for)?\s*(\d+)\s*weeks?",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_hours_per_day",
                language="math",
                pattern=r"(\d+\.?\d*)\s*hours?\s*(?:a|per|each)\s*day.*?(\d+)\s*days?",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
        ]
    )

    # Gain / loss patterns
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_has_gets_more",
                language="math",
                pattern=r"(?:has|have|had|starts? with)\s*(\d+\.?\d*).*?(?:gets?|gains?|receives?|finds?|earns?)\s*(\d+\.?\d*)\s*(?:more)?",
                rpn_program="{0} {1} +",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_has_loses",
                language="math",
                pattern=r"(?:has|have|had)\s*(\d+\.?\d*).*?(?:loses?|lost|gives? away|spent)\s*(\d+\.?\d*)",
                rpn_program="{0} {1} -",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_quantity_removed",
                language="math",
                pattern=r"(?:there (?:are|were)|has|have)\s*(\d+).*?(\d+)\s*(?:fly away|flew away|leave|left|are removed|are taken|were taken)",
                rpn_program="{0} {1} -",
                domain="math_arithmetic",
            ),
        ]
    )

    # Multiplication with "times"
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_x_times_as_many_as_y",
                language="math",
                pattern=r"(\d+)\s*times\s*(?:as many|as much)\s*(?:as)?\s*.*?(\d+)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_n_times_that",
                language="math",
                pattern=r"(\d+).*?(\d+)\s*times\s*(?:that|as much|as many|more)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
        ]
    )

    # Distribution patterns
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_each_for_people",
                language="math",
                pattern=r"(\d+\.?\d*)\s*(?:each|apiece).*?(?:for|to|among)?\s*(\d+)\s*(?:people|children|students|friends?|members?|guests?)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_people_get_each",
                language="math",
                pattern=r"(\d+)\s*(?:people|children|students|friends?).*?(?:get|receive|have)\s*(\d+\.?\d*)\s*(?:each|apiece)",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
        ]
    )

    # Multi-step add/sub patterns
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_add_three",
                language="math",
                pattern=r"(?:had|has|starts? with)\s*(\d+\.?\d*).*?(?:got|gets|gains?|receives?)\s*(\d+\.?\d*).*?(?:then|and|also).*?(\d+\.?\d*)\s*more",
                rpn_program="{0} {1} + {2} +",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_sub_three",
                language="math",
                pattern=r"(\d+\.?\d*).*?(?:loses?|spent|gave)\s*(\d+\.?\d*).*?(?:then|and|also).*?(?:loses?|spent|gave)\s*(\d+\.?\d*)",
                rpn_program="{0} {1} - {2} -",
                domain="math_arithmetic",
            ),
        ]
    )

    # Age problems
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_years_older",
                language="math",
                pattern=r"(\d+)\s*years?\s*older.*?(?:who is|is)\s*(\d+)",
                rpn_program="{1} {0} +",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_years_younger",
                language="math",
                pattern=r"(\d+)\s*years?\s*younger.*?(?:who is|is)\s*(\d+)",
                rpn_program="{1} {0} -",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_in_years_will_be",
                language="math",
                pattern=r"[Ii]n\s*(\d+)\s*years?.*?(?:will be|be)\s*(\d+)",
                rpn_program="{1} {0} -",
                domain="math_arithmetic",
            ),
        ]
    )

    # Work / rate problems
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_pages_in_hours",
                language="math",
                pattern=r"(\d+)\s*pages?.*?(?:in|per|every)\s*(\d+)\s*hours?",
                rpn_program="{0} {1} /",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_hourly_rate_work",
                language="math",
                pattern=r"(?:earns?|makes?|gets?)\s*\$?(\d+\.?\d*)\s*(?:per|an?|each)\s*hour.*?(?:works?|worked)\s*(\d+\.?\d*)\s*hours?",
                rpn_program="{0} {1} *",
                domain="math_arithmetic",
            ),
        ]
    )

    # Remaining / left-over
    templates.extend(
        [
            GrammarRule(
                rule_id="gsm_uses_left",
                language="math",
                # Exclude "gives/gave": those are frequently multi-step (multiple recipients) and are handled by composite rules.
                pattern=r"(?:has|have|had)\s*(\d+\.?\d*).*?(?:uses?|used|eats?|ate)\s*(\d+\.?\d*).*?(?:left|remaining|remain)",
                rpn_program="{0} {1} -",
                domain="math_arithmetic",
            ),
            GrammarRule(
                rule_id="gsm_total_minus_some",
                language="math",
                pattern=r"(\d+)\s*(?:total|in all).*?(\d+)\s*(?:are|is|were).*?(?:how many|what)",
                rpn_program="{0} {1} -",
                domain="math_arithmetic",
            ),
        ]
    )

    return templates


def get_all_templates() -> List[GrammarRule]:
    """Get all curated templates sorted by specificity (longer, more captures first)."""
    templates: List[GrammarRule] = []
    templates.extend(get_gsm8k_templates())
    templates.extend(get_expanded_templates())
    templates.sort(key=lambda r: (len(r.pattern), r.pattern.count("("), r.rule_id), reverse=True)
    return templates


__all__ = [
    "get_all_templates",
    "get_gsm8k_templates",
    "get_expanded_templates",
]
