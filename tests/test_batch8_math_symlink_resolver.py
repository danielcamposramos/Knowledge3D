from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id
from knowledge3d.ingestion.math_symbol_builder import math_symbol_star_id
from knowledge3d.ingestion.math_symlink_resolver import MathSymlinkResolveError, MathSymlinkResolver
from tests._batch8_helpers import FakeCanonicalLookup


def test_resolver_handles_letter_symbol_constant_and_concept_paths(tmp_path: Path) -> None:
    lookup = FakeCanonicalLookup()
    lookup.register(kind="meaning_star", key="concept_arithmetic_precedence", star_id="math_concept_arithmetic_precedence", metadata={})
    resolver = MathSymlinkResolver(lookup)

    assert resolver.resolve("letter::a") == canonical_char_star_id("a")
    assert resolver.resolve("symbol::plus") == math_symbol_star_id("+")
    assert resolver.resolve("constant::pi") == math_symbol_star_id("\u03c0")
    assert resolver.resolve("concept::arithmetic_precedence") == "math_concept_arithmetic_precedence"


def test_resolver_supports_allowlist_literal_match(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("symbol::unresolved_alias\n", encoding="utf-8")
    resolver = MathSymlinkResolver(FakeCanonicalLookup(), allowlist_path=allowlist)
    assert resolver.resolve("symbol::unresolved_alias") is None


def test_resolver_raises_on_unknown_alias() -> None:
    resolver = MathSymlinkResolver(FakeCanonicalLookup())
    with pytest.raises(MathSymlinkResolveError):
        resolver.resolve("symbol::missing_alias")
