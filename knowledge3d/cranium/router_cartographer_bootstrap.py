"""Bootstrap Meta-Navigation routing stars into the Grammar galaxy."""

from __future__ import annotations

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


_ROUTING_META_REFS = [
    "meta_navigation",
    "routing_signal",
    "bootstrap:router_cartographer_v1",
]


ROUTING_STARS = [
    MeaningCentricStar(
        star_id="routing:task_type:math",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",
        meaning_rpn="MATH_QUERY_PATTERN",
        behavior_rpn="1.0",
        taxonomy_refs=["routing", "task_type", "math"],
        grammar_refs=["meta_navigation", "task_type", "math"],
        meta_refs=_ROUTING_META_REFS,
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="routing:task_type:question",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",
        meaning_rpn="QUESTION_QUERY_PATTERN",
        behavior_rpn="2.0",
        taxonomy_refs=["routing", "task_type", "question"],
        grammar_refs=["meta_navigation", "task_type", "question"],
        meta_refs=_ROUTING_META_REFS,
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="routing:task_type:spatial",
        meaning_class="routing_signal",
        domain="meta_navigation",
        galaxy_ref="Grammar",
        meaning_rpn="SPATIAL_QUERY_PATTERN",
        behavior_rpn="3.0",
        taxonomy_refs=["routing", "task_type", "spatial"],
        grammar_refs=["meta_navigation", "task_type", "spatial"],
        meta_refs=_ROUTING_META_REFS,
        confidence=1,
        polarity=1,
    ),
]


def build_router_cartographer_stars() -> list[MeaningCentricStar]:
    """Return the foundational routing stars for boot-time seeding."""
    return [MeaningCentricStar.from_dict(star.to_dict()) for star in ROUTING_STARS]


__all__ = ["ROUTING_STARS", "build_router_cartographer_stars"]
