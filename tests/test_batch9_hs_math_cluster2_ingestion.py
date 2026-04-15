from __future__ import annotations

import os

import pytest
from qdrant_client import models

from knowledge3d.ingestion.canonical_lookup import CANONICAL_COLLECTION, CanonicalLookup
from scripts.ingest_hs_math_cluster2 import run_cluster2_ingestion
from scripts.seed_batch8_canonical_math_aliases import seed
from tests._batch8_helpers import FakeCanonicalLookup


def test_cluster2_dry_run_has_no_hard_misses() -> None:
    lookup = FakeCanonicalLookup()
    seed(lookup)
    summary = run_cluster2_ingestion(lookup, write=False)
    assert summary["rows"] == 56
    assert summary["skipped_blocks"] == []
    assert summary["forward_refs"] == []


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("K3D_QDRANT_INTEGRATION") != "1", reason="requires K3D_QDRANT_INTEGRATION=1")
def test_batch9_cluster2_qdrant_write_is_idempotent() -> None:
    lookup = CanonicalLookup()
    seed(lookup)
    run_cluster2_ingestion(lookup, write=True)
    count_after_first = lookup.client.get_collection(CANONICAL_COLLECTION).points_count
    run_cluster2_ingestion(lookup, write=True)
    count_after_second = lookup.client.get_collection(CANONICAL_COLLECTION).points_count
    assert count_after_first == count_after_second

    points, _ = lookup.client.scroll(
        collection_name=CANONICAL_COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=1,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="kind", match=models.MatchValue(value="meaning_star")),
                models.FieldCondition(key="metadata.subkind", match=models.MatchValue(value="math_hs_cluster2")),
            ]
        ),
    )
    assert points
