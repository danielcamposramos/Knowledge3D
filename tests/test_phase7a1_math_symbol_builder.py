from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id
from knowledge3d.ingestion.math_symbol_builder import (
    build_math_symbol_galaxy,
    canonical_math_symbol_entry,
    iter_math_symbol_codepoints,
    math_symbol_star_id,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_math_symbol_plus_sign_links_to_character_star():
    char_id = canonical_char_star_id("+")
    char_star = MeaningCentricStar(star_id=char_id, meaning_rpn="PLUS FORM", domain="Character")
    build = build_math_symbol_galaxy(
        existing_char_stars={char_id: char_star},
        codepoints=[ord("+")],
        math_classes={ord("+"): "B"},
    )

    star = build.stars["math_symbol_plus_sign"]
    assert star["char_refs"] == ["char_u002b"]
    assert star["latex_commands"] == ["+"]
    assert star["selection_role"] == "operator"
    assert star["has_executable_program"] is True
    assert star["program_ref"] == "rpn_program_addition"
    assert "ADD" in star["meaning_rpn"]
    assert build.target_updates["char_u002b"]["taxonomy_refs"] == ["math_symbol_plus_sign"]


def test_large_operator_summation_gets_latex_and_followup_when_template_deferred():
    char_id = canonical_char_star_id("∑")
    char_star = MeaningCentricStar(star_id=char_id, meaning_rpn="SUM FORM", domain="Character")
    build = build_math_symbol_galaxy(
        existing_char_stars={char_id: char_star},
        codepoints=[ord("∑")],
        math_classes={ord("∑"): "O"},
    )

    star = build.stars["math_symbol_n_ary_summation"]
    assert star["char_refs"] == ["char_u2211"]
    assert "\\sum" in star["latex_commands"]
    assert star["selection_role"] == "operator"
    assert star["has_executable_program"] is False
    assert build.followups == [
        {
            "star_id": "math_symbol_n_ary_summation",
            "codepoint": "U+2211",
            "reason": "large_operator_template_deferred",
        }
    ]


def test_pi_symbol_is_value_bearing_operand_not_delimiter():
    char_id = canonical_char_star_id("π")
    char_star = MeaningCentricStar(star_id=char_id, meaning_rpn="PI FORM", domain="Character")
    build = build_math_symbol_galaxy(
        existing_char_stars={char_id: char_star},
        codepoints=[ord("π")],
    )

    star = build.stars["math_symbol_greek_small_letter_pi"]
    assert star["selection_role"] == "operand"
    assert star["meaning_rpn"] == "PI CONSTANT STORE"
    assert star["char_refs"] == ["char_u03c0"]


def test_missing_character_target_is_reported_not_created():
    build = build_math_symbol_galaxy(codepoints=[ord("+")], math_classes={ord("+"): "B"})

    assert build.stars["math_symbol_plus_sign"]["char_refs"] == []
    assert build.skipped_links == [{"source": "math_symbol_plus_sign", "target": "char_u002b", "reason": "target_missing"}]
    assert build.target_updates == {}


def test_canonical_math_symbol_entry_contract():
    build = build_math_symbol_galaxy(codepoints=[ord("+")], math_classes={ord("+"): "B"})
    entry = canonical_math_symbol_entry(build.stars["math_symbol_plus_sign"])

    assert entry == {
        "kind": "math_symbol",
        "key": "U+002B",
        "star_id": "math_symbol_plus_sign",
        "metadata": {
            "latex_commands": ["+"],
            "math_class": "B",
            "has_executable_program": True,
        },
    }


def test_math_inventory_contains_seed_symbols():
    inventory = set(iter_math_symbol_codepoints())

    assert ord("+") in inventory
    assert ord("∑") in inventory
    assert ord("π") in inventory
    assert math_symbol_star_id("+") == "math_symbol_plus_sign"
