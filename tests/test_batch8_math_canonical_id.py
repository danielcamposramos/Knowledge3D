from __future__ import annotations

import pytest

from knowledge3d.ingestion.math_canonical_id import MathCanonicalIdError, normalise_canonical_id


def test_canonical_id_normalises_all_batch8_dialects() -> None:
    assert normalise_canonical_id("rule_order_of_operations_pemdas") == ("rule", "rule_order_of_operations_pemdas")
    assert normalise_canonical_id("formula::triangle_area_base_height") == ("formula", "formula_triangle_area_base_height")
    assert normalise_canonical_id("formula_population_variance") == ("formula", "formula_population_variance")


def test_canonical_id_round_trip_is_idempotent() -> None:
    _, canonical_key = normalise_canonical_id("formula::triangle_area_base_height")
    assert normalise_canonical_id(canonical_key) == ("formula", canonical_key)


def test_canonical_id_rejects_unknown_category() -> None:
    with pytest.raises(MathCanonicalIdError):
        normalise_canonical_id("unknown_example")
