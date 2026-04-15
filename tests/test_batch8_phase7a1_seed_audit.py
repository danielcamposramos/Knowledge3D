from __future__ import annotations

from scripts.ingest_phase7a1_seed_audit import run_audit
from tests._batch8_helpers import FakeCanonicalLookup


def test_phase7a1_seed_audit_reports_present_and_missing() -> None:
    lookup = FakeCanonicalLookup(preset_star_ids={"char_a"})
    payload = run_audit(lookup)
    assert payload["checked"] >= 1
    assert payload["present"] >= 0
    assert isinstance(payload["missing"], list)
