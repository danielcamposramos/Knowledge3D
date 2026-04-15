from __future__ import annotations

import os

import pytest

from scripts.ingest_phase7a1_seed_audit import run_audit
from scripts.seed_batch8_canonical_math_aliases import (
    _dispatch_kind,
    iter_alias_rows,
    iter_concept_alias_rows,
    iter_concept_seed_rows,
    seed,
)


class FakeCanonicalLookup:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._records: dict[tuple[str, str], dict[str, object]] = {}
        self._star_ids: set[str] = set()

    def ensure_collection(self) -> None:
        return None

    def register(self, *, kind: str, key: str, star_id: str, metadata=None) -> str:
        payload = {
            "kind": kind,
            "key": key,
            "star_id": str(star_id),
            "metadata": dict(metadata or {}),
        }
        self.calls.append(payload)
        self._records[(kind, key)] = payload
        self._star_ids.add(str(star_id))
        return str(star_id)

    def star_id_exists(self, star_id: str) -> bool:
        return str(star_id) in self._star_ids


def test_seed_issues_exactly_87_register_calls() -> None:
    lookup = FakeCanonicalLookup()
    counts = seed(lookup)
    assert sum(counts.values()) == 88
    assert counts["meaning_star"] == 14
    assert len(lookup.calls) == 88


def test_kind_dispatch_matches_prefix_families() -> None:
    assert _dispatch_kind("math_symbol_plus_sign") == "math_symbol"
    assert _dispatch_kind("char_u005e") == "char"
    assert _dispatch_kind("concept_reciprocal") == "concept"


def test_all_alias_rows_are_registered_with_expected_triplets() -> None:
    lookup = FakeCanonicalLookup()
    seed(lookup)
    expected = [(_dispatch_kind(star_id), alias_name, star_id) for alias_name, star_id in iter_alias_rows()]
    expected.extend(("concept", key, star_id) for key, star_id in iter_concept_alias_rows())
    expected.extend(("meaning_star", key, star_id) for key, star_id in iter_concept_seed_rows())
    observed = [(call["kind"], call["key"], call["star_id"]) for call in lookup.calls]
    assert observed == expected


def test_seed_iteration_is_stable_across_two_calls() -> None:
    lookup = FakeCanonicalLookup()
    seed(lookup)
    first_calls = list(lookup.calls)
    seed(lookup)
    assert len(lookup.calls) == 176
    assert lookup.calls[:88] == first_calls
    assert lookup.calls[88:] == first_calls


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("K3D_QDRANT_INTEGRATION") != "1", reason="requires K3D_QDRANT_INTEGRATION=1")
def test_real_seed_makes_audit_pass() -> None:
    from knowledge3d.ingestion.canonical_lookup import CanonicalLookup

    lookup = CanonicalLookup()
    seed(lookup)
    payload = run_audit(lookup)
    assert payload["checked"] == 88
    assert payload["present"] == 88
    assert payload["missing"] == []
