from __future__ import annotations

import math

import pytest

from knowledge3d.cranium.sovereign_matryoshka_embedder import (
    SovereignMatryoshkaTextEmbedder,
    get_sovereign_matryoshka_text_embedder,
)
from knowledge3d.ingestion.star_crafter import StarCrafter
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 1.0e-8:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width <= 0:
        return 0.0
    return sum(float(left[index]) * float(right[index]) for index in range(width))


@pytest.mark.gpu
def test_phase6b3_tier_prefix_contract_and_determinism() -> None:
    embedder = get_sovereign_matryoshka_text_embedder()
    stack = embedder.embed_stack("plus addition sum arithmetic operator")
    again = embedder.embed_stack("plus addition sum arithmetic operator")

    assert tuple(stack) == SovereignMatryoshkaTextEmbedder.TIER_DIMS
    for tier in SovereignMatryoshkaTextEmbedder.TIER_DIMS:
        assert len(stack[tier]) == tier
        assert stack[tier] == pytest.approx(again[tier], abs=1.0e-7)
        assert stack[tier] == pytest.approx(_normalize(stack[2048][:tier]), abs=1.0e-6)


@pytest.mark.gpu
def test_phase6b3_crafter_and_knowledgeverse_share_singleton() -> None:
    singleton = get_sovereign_matryoshka_text_embedder()
    crafter = StarCrafter()
    kv = Knowledgeverse.__new__(Knowledgeverse)
    kv._sovereign_text_embedder = None

    assert crafter.embedder is singleton
    assert kv.get_sovereign_text_embedder() is singleton


@pytest.mark.gpu
def test_phase6b3_crafted_math_rows_route_with_same_tier() -> None:
    embedder = get_sovereign_matryoshka_text_embedder()
    rows = StarCrafter(embedder=embedder).craft_all()
    by_id = {str(row.get("id") or ""): row for row in rows}

    def rank(query: str, target_id: str) -> int:
        query_vec = embedder.embed_tier(query, 64)
        scored = sorted(
            (
                (_cosine(query_vec, list(row.get("embedding_tier_64") or row.get("embedding") or [])), row_id)
                for row_id, row in by_id.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [row_id for _score, row_id in scored].index(target_id) + 1

    assert rank("plus", "math_operator_addition") <= 10
    assert rank("addition", "math_operator_addition") <= 10
    assert rank("two", "concept_digit_two") <= 20
    assert rank("3", "concept_digit_three") <= 20
