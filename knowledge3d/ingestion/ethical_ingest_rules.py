"""Seed ethical trit rules for ingestion-time star metadata.

This module is ingestion-only. Runtime gating consumes the native
`ethical_trit` field from Galaxy star records on GPU.
"""

from __future__ import annotations

FORBIDDEN_STARS: frozenset[str] = frozenset(
    {
        "harm_intent",
        "deception_malicious",
        "coercion_unjustified",
        "exploit_vulnerability",
    }
)

DEFEASIBLE_STARS: frozenset[str] = frozenset(
    {
        "self_defense",
        "triage_tradeoff",
        "consent_sensitive",
        "safety_exception",
    }
)


def ethical_trit_for_star_id(star_id: str) -> int:
    key = str(star_id or "").strip().lower()
    if key in FORBIDDEN_STARS:
        return -1
    if key in DEFEASIBLE_STARS:
        return 1
    return 0


__all__ = ["DEFEASIBLE_STARS", "FORBIDDEN_STARS", "ethical_trit_for_star_id"]
