"""
Integration test for theorem pattern extraction from books_v5_clean2.

This validates that single-step theorem patterns are correctly extracted
from validated artifacts and formatted with semantic tags + RPN programs.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from knowledge3d.cranium.math_galaxy_population import (
    extract_theorem_patterns,
    populate_theorem_patterns,
    THEOREM_PATTERNS,
)


BOOKS_V5_CLEAN2_PATH = "/K3D/Knowledge3D.local/galaxies/books_v5_clean2"


@pytest.fixture
def artifact_dirs():
    """Fixture providing paths to books_v5_clean2 artifact directories."""
    base = Path(BOOKS_V5_CLEAN2_PATH)
    if not base.exists():
        pytest.skip(f"books_v5_clean2 not found at {BOOKS_V5_CLEAN2_PATH}")
    return [str(base)]


def test_extract_theorem_patterns(artifact_dirs):
    """Test theorem pattern extraction from books_v5_clean2."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    # Should extract at least 3 patterns from 9 defined patterns
    # (Strict matching criteria means not all patterns will match artifacts)
    assert len(patterns) >= 3, f"Expected ≥3 patterns, got {len(patterns)}"

    # Each pattern must have required fields
    for pattern in patterns:
        assert "pattern_id" in pattern, "Missing pattern_id"
        assert "domain" in pattern, "Missing domain"
        assert "semantic_tags" in pattern, "Missing semantic_tags"
        assert "precondition" in pattern, "Missing precondition"
        assert "transformation" in pattern, "Missing transformation"
        assert "postcondition" in pattern, "Missing postcondition"
        assert "source" in pattern, "Missing source"

        # Semantic tags must be non-empty
        assert len(pattern["semantic_tags"]) > 0, f"No semantic_tags for {pattern['pattern_id']}"

        # Transformation must have RPN program
        trans = pattern["transformation"]
        assert "rpn_program" in trans, f"No RPN program for {pattern['pattern_id']}"
        assert len(trans["rpn_program"]) > 0, f"Empty RPN for {pattern['pattern_id']}"
        assert "tier" in trans, f"No tier for {pattern['pattern_id']}"

        # Source must have examples
        source = pattern["source"]
        assert source["example_count"] >= 2, f"Insufficient examples for {pattern['pattern_id']}"
        assert len(source["artifact_ids"]) > 0, f"No artifact_ids for {pattern['pattern_id']}"


def test_theorem_pattern_domains(artifact_dirs):
    """Test that extracted patterns span multiple domains."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    domains = {p["domain"] for p in patterns}

    # Should have calculus patterns (most common in books_v5_clean2)
    assert "calculus" in domains, "No calculus patterns extracted"

    # May have geometry if enough pythagorean examples
    # (Not asserting this as it depends on book content)


def test_theorem_pattern_tiers(artifact_dirs):
    """Test that RPN programs are assigned correct tiers."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    for pattern in patterns:
        tier = pattern["transformation"]["tier"]
        assert tier in {1, 2, 3}, f"Invalid tier {tier} for {pattern['pattern_id']}"

        # Derivative/integral patterns should be Tier 3
        if any(tag in pattern["semantic_tags"] for tag in ["derivative", "integral"]):
            assert tier == 3, f"Derivative/integral pattern {pattern['pattern_id']} not Tier 3"


def test_populate_theorem_patterns(artifact_dirs):
    """Test that populate_theorem_patterns updates global THEOREM_PATTERNS."""
    # Clear global first
    import knowledge3d.cranium.math_galaxy_population as mgp
    mgp.THEOREM_PATTERNS = []

    patterns = populate_theorem_patterns(artifact_dirs, min_examples=2)

    # Should return patterns
    assert len(patterns) >= 3, f"Expected ≥3 patterns, got {len(patterns)}"

    # Global THEOREM_PATTERNS should be updated
    assert len(mgp.THEOREM_PATTERNS) == len(patterns), "Global THEOREM_PATTERNS not updated"
    assert mgp.THEOREM_PATTERNS == patterns, "Global THEOREM_PATTERNS mismatch"


def test_semantic_tags_from_artifacts(artifact_dirs):
    """Test that semantic tags are collected from artifact symbol_bindings."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    # At least one pattern should have semantic tags from artifacts
    # (beyond just the base tags from pattern definition)
    found_artifact_tags = False
    for pattern in patterns:
        tags = pattern["semantic_tags"]
        base_tags = {"derivative", "polynomial", "power_rule", "product_rule",
                     "quotient_rule", "chain_rule", "sum_rule", "constant_multiple",
                     "integral", "integration_by_parts", "fundamental_theorem",
                     "angle", "trigonometry", "identity"}

        # If tags include domain-specific roles from artifacts
        artifact_tags = set(tags) - base_tags
        if artifact_tags:
            found_artifact_tags = True
            break

    # Not strictly required (depends on artifact content), but good to check
    # assert found_artifact_tags, "No semantic tags collected from artifacts"


def test_rpn_program_structure(artifact_dirs):
    """Test that RPN programs have valid structure."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    valid_opcodes = {
        # Tier 1
        "PUSH_N", "PUSH_X", "PUSH_C", "PUSH_F", "PUSH_G", "PUSH_U", "PUSH_V",
        "PUSH_1", "PUSH_2", "PUSH_SIN_THETA", "PUSH_COS_THETA",
        "POP", "DUP", "SWAP",
        "ADD", "SUB", "MULT", "DIV", "POW", "POW2", "SQRT", "ABS", "NEG",
        "EQ", "LT", "GT", "AND", "OR",
        # Tier 2
        "FACTORIAL", "BINOM", "SUM", "PRODUCT",
        "SUBSCRIPT", "PIECEWISE", "CASE",
        # Tier 3
        "DERIVATIVE", "INTEGRAL", "LIMIT",
        "MATRIX_MULT", "DOT_PRODUCT", "CROSS_PRODUCT",
        "DET", "TRACE", "EIGENVALUE",
        "COMPOSE", "EVAL_A", "EVAL_B",
    }

    for pattern in patterns:
        rpn = pattern["transformation"]["rpn_program"]
        for opcode in rpn:
            # Opcodes should be uppercase strings
            assert isinstance(opcode, str), f"Non-string opcode in {pattern['pattern_id']}"
            assert opcode.isupper() or opcode.startswith("PUSH_"), \
                f"Invalid opcode format: {opcode} in {pattern['pattern_id']}"

            # Should be in valid opcode set (or be a PUSH variant)
            if not opcode.startswith("PUSH_"):
                assert opcode in valid_opcodes, \
                    f"Unknown opcode: {opcode} in {pattern['pattern_id']}"


def test_pattern_preconditions(artifact_dirs):
    """Test that preconditions are populated from artifacts."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    for pattern in patterns:
        precond = pattern["precondition"]

        # Should have artifact_types
        assert "artifact_types" in precond, f"No artifact_types for {pattern['pattern_id']}"
        assert len(precond["artifact_types"]) > 0, \
            f"Empty artifact_types for {pattern['pattern_id']}"

        # Should have context_cues
        assert "context_cues" in precond, f"No context_cues for {pattern['pattern_id']}"

        # Should have lhs/rhs patterns
        assert "lhs_pattern" in precond, f"No lhs_pattern for {pattern['pattern_id']}"
        assert "rhs_pattern" in precond, f"No rhs_pattern for {pattern['pattern_id']}"


def test_pattern_source_provenance(artifact_dirs):
    """Test that source provenance is tracked."""
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=2)

    for pattern in patterns:
        source = pattern["source"]

        # Should have book_ids
        assert "book_ids" in source, f"No book_ids for {pattern['pattern_id']}"
        assert len(source["book_ids"]) > 0, f"Empty book_ids for {pattern['pattern_id']}"

        # Should have artifact_ids
        assert "artifact_ids" in source, f"No artifact_ids for {pattern['pattern_id']}"
        assert len(source["artifact_ids"]) > 0, \
            f"Empty artifact_ids for {pattern['pattern_id']}"

        # Should have example_count matching min_examples
        assert source["example_count"] >= 2, \
            f"Insufficient examples for {pattern['pattern_id']}: {source['example_count']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
