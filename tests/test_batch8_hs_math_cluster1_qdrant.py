from __future__ import annotations

import os

import pytest
from qdrant_client import models

from knowledge3d.ingestion.canonical_lookup import CANONICAL_COLLECTION, CanonicalLookup
from scripts.ingest_hs_math_cluster1 import run_cluster1_ingestion


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("K3D_QDRANT_INTEGRATION") != "1", reason="requires K3D_QDRANT_INTEGRATION=1")
def test_batch8_cluster1_qdrant_write_is_idempotent() -> None:
    lookup = CanonicalLookup()
    run_cluster1_ingestion(lookup, write=True)
    count_after_first = lookup.client.get_collection(CANONICAL_COLLECTION).points_count
    run_cluster1_ingestion(lookup, write=True)
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
                models.FieldCondition(key="metadata.subkind", match=models.MatchValue(value="math_hs_cluster1")),
            ]
        ),
    )
    assert points
