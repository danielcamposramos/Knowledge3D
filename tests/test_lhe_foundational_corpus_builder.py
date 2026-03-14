from __future__ import annotations

import json

from scripts.build_lhe_foundational_corpus import (
    _FORBIDDEN_SMOKE_IDS,
    _FORBIDDEN_SMOKE_STRINGS,
    BOOTSTRAP_TAG,
    build_lhe_foundational_payload,
)


def test_lhe_foundational_payload_is_deterministic_and_counted() -> None:
    rows_a, manifest_a = build_lhe_foundational_payload()
    rows_b, manifest_b = build_lhe_foundational_payload()

    assert rows_a == rows_b
    assert manifest_a == manifest_b
    assert manifest_a["bootstrap"] == BOOTSTRAP_TAG
    assert manifest_a["concept_family_count"] == 2048
    assert manifest_a["domain_concept_counts"] == {
        "mathematics": 448,
        "physics": 288,
        "cs_ai": 288,
        "biology_medicine": 256,
        "chemistry": 192,
        "humanities_social_science": 256,
        "engineering": 160,
        "other": 160,
    }
    assert manifest_a["row_count"] == len(rows_a)
    assert manifest_a["galaxy_row_counts"]["Reality"] == 2048
    assert manifest_a["galaxy_row_counts"]["Word"] == 2048


def test_lhe_foundational_payload_has_stable_ids_and_valid_refs() -> None:
    rows, manifest = build_lhe_foundational_payload()
    reality_ids = {
        row["entry"]["id"] for row in rows if row["galaxy"] == "Reality"
    }
    word_ids = {
        row["entry"]["id"] for row in rows if row["galaxy"] == "Word"
    }
    math_rows = [row for row in rows if row["galaxy"] == "Math"]
    grammar_rows = [row for row in rows if row["galaxy"] == "Grammar"]

    assert "concept_math_homology_group" in reality_ids
    assert "concept_physics_gamma_matrices" in reality_ids
    assert "concept_humanities_non_sadism_principle" in reality_ids
    assert "word_homology_group" in word_ids
    assert "word_gamma_matrices" in word_ids
    assert "word_non_sadism_principle" in word_ids
    assert any(row["entry"]["id"] == "math_gamma_matrix_clifford_relation" for row in math_rows)
    assert any(row["entry"]["id"] == "grammar_humanities_non_sadism_principle_reasoning" for row in grammar_rows)
    assert manifest["representative_ids"]["reality"][0] == "concept_math_homology_group"

    for row in rows:
        entry = row["entry"]
        metadata = entry.get("metadata", {})
        if row["galaxy"] == "Word":
            assert metadata["meaning_ref"] in reality_ids
        if row["galaxy"] == "Math":
            assert metadata["formalizes_ref"] in reality_ids
        if row["galaxy"] == "Grammar":
            assert metadata["reasons_about_ref"] in reality_ids


def test_lhe_foundational_payload_is_clean_and_covers_required_subfields() -> None:
    rows, manifest = build_lhe_foundational_payload()
    blob = json.dumps(rows, ensure_ascii=True, sort_keys=True)

    for forbidden in _FORBIDDEN_SMOKE_IDS:
        assert forbidden not in blob
    for forbidden in _FORBIDDEN_SMOKE_STRINGS:
        assert forbidden not in blob

    required_subfields = {
        "algebraic_topology",
        "lie_theory",
        "moduli_and_elliptic",
        "functional_analysis",
        "number_theory",
        "relativistic_quantum",
        "qft_gauge",
        "deep_learning",
        "cryptography",
        "molecular_biology",
        "immunology",
        "organic_chemistry",
        "ethics_population",
        "rhetoric_language_figures",
        "circuits_and_signal",
        "chess",
    }
    for subfield in required_subfields:
        assert manifest["subfield_counts"].get(subfield, 0) > 0

    assert manifest["galaxy_row_counts"]["Grammar"] > 0
    assert manifest["galaxy_row_counts"]["Math"] > 0


def test_lhe_foundational_payload_domain_filter_and_scaling() -> None:
    rows, manifest = build_lhe_foundational_payload(
        target_concepts=64,
        domains=["mathematics", "other"],
    )

    assert manifest["concept_family_count"] == 64
    assert manifest["domains"] == ["mathematics", "other"]
    assert set(manifest["domain_concept_counts"]) == {"mathematics", "other"}
    assert manifest["galaxy_row_counts"]["Reality"] == 64
    assert all(
        row["entry"]["metadata"]["bootstrap"] == BOOTSTRAP_TAG
        for row in rows
        if isinstance(row.get("entry", {}).get("metadata"), dict)
    )
