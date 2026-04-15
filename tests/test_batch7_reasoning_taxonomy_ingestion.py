from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingest_reasoning_taxonomy import (
    ALLOWLIST_PATH,
    enforce_integrity,
    ingest_reasoning_taxonomy,
    load_allowlist,
    parse_all_catalogues,
)


class FakeCanonicalLookup:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], dict[str, object]] = {}
        self.star_ids: set[str] = set()

    def register(self, *, kind: str, key: str, star_id: str, metadata=None) -> str:
        self.records[(kind, key)] = {
            "kind": kind,
            "key": key,
            "star_id": star_id,
            "metadata": dict(metadata or {}),
        }
        self.star_ids.add(str(star_id))
        return str(star_id)

    def star_id_exists(self, star_id: str) -> bool:
        return str(star_id) in self.star_ids


def test_integrity_gate_fails_without_allowlist_for_repo_inputs() -> None:
    payloads = parse_all_catalogues()
    with pytest.raises(ValueError, match="reasoning_taxonomy_dangling_refs"):
        enforce_integrity(payloads, FakeCanonicalLookup(), frozenset())


def test_integrity_gate_passes_with_batch7_allowlist() -> None:
    payloads = parse_all_catalogues()
    enforce_integrity(payloads, FakeCanonicalLookup(), load_allowlist(ALLOWLIST_PATH))


def test_ingestion_is_idempotent_under_deterministic_kind_key_pairs() -> None:
    lookup = FakeCanonicalLookup()
    summary_one = ingest_reasoning_taxonomy(lookup=lookup)
    count_after_first = len(lookup.records)
    summary_two = ingest_reasoning_taxonomy(lookup=lookup)
    count_after_second = len(lookup.records)

    assert summary_one["stars"] >= 20
    assert count_after_first == count_after_second
    assert summary_two["templates"] >= 1


def test_symlink_and_template_passes_register_expected_records() -> None:
    lookup = FakeCanonicalLookup()
    ingest_reasoning_taxonomy(lookup=lookup)

    symlink_records = [payload for (kind, _), payload in lookup.records.items() if kind == "reasoning_taxonomy_symlink"]
    template_records = [payload for (kind, _), payload in lookup.records.items() if kind == "grammar_template"]
    meaning_records = [payload for (kind, _), payload in lookup.records.items() if kind == "meaning_star"]

    assert symlink_records
    assert template_records
    assert meaning_records
    assert all(
        payload["metadata"].get("language") in {"en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru"}
        for payload in template_records
    )
    assert all(payload["metadata"].get("context_id") == 0 for payload in meaning_records)
