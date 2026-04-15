from __future__ import annotations

import os

import pytest
from qdrant_client import models

from knowledge3d.ingestion.canonical_lookup import CANONICAL_COLLECTION, CanonicalLookup
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from scripts.ingest_reasoning_taxonomy import ingest_reasoning_taxonomy


@pytest.mark.skipif(
    os.environ.get("K3D_QDRANT_INTEGRATION") != "1",
    reason="requires K3D_QDRANT_INTEGRATION=1",
)
def test_reasoning_taxonomy_ingestion_is_idempotent_and_materializes_meaning_star() -> None:
    lookup = CanonicalLookup()
    summary_one = ingest_reasoning_taxonomy(lookup=lookup)
    count_after_first = lookup.client.get_collection(CANONICAL_COLLECTION).points_count
    summary_two = ingest_reasoning_taxonomy(lookup=lookup)
    count_after_second = lookup.client.get_collection(CANONICAL_COLLECTION).points_count

    assert summary_one["stars"] >= 20
    assert summary_two["stars"] >= 20
    assert count_after_first == count_after_second
    assert lookup.find_star_id(kind="meaning_star", key="concept_automated_reasoning") == "concept_automated_reasoning"

    points, _ = lookup.client.scroll(
        collection_name=CANONICAL_COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=1,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="kind", match=models.MatchValue(value="meaning_star")),
                models.FieldCondition(key="key", match=models.MatchValue(value="concept_automated_reasoning")),
            ]
        ),
    )
    assert len(points) == 1
    payload = dict(points[0].payload or {})
    meaning_star = MeaningCentricStar.from_dict(payload["metadata"]["meaning_star"])
    assert meaning_star.star_id == "concept_automated_reasoning"
    assert meaning_star.context_id == 0
