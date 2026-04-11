from __future__ import annotations

import pytest

from knowledge3d.ingestion.star_crafter import build_foundational_star_crafter_outputs
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.sovereign_hot_path import SovereignHotPath


def _runtime_stub() -> SovereignHotPath:
    return SovereignHotPath.__new__(SovereignHotPath)


class _FakeMatryoshkaEmbedder:
    def embed_stack(self, text: str) -> dict[int, list[float]]:
        seed = sum(ord(char) for char in str(text or ""))
        return {
            tier: [float(((seed + index) % 17) + 1) for index in range(tier)]
            for tier in (64, 128, 512, 2048)
        }


def test_phase6b1_legacy_route_validator_unchanged() -> None:
    runtime = _runtime_stub()
    with pytest.raises(ValueError, match="sovereign_build_route_invalid"):
        runtime._validate_route_link_coverage(
            [
                {
                    "id": "legacy_router",
                    "selection_role": "router",
                    "sovereign_route_exempt": False,
                    "executor_refs": [],
                    "validator_refs": [],
                }
            ]
        )


def test_phase6b1_exempt_router_accepted_with_symlink_closure() -> None:
    runtime = _runtime_stub()
    stars = [
        {
            "id": "grammar_binary_op_infix",
            "selection_role": "router",
            "sovereign_route_exempt": True,
            "grammar_refs": ["concept_digit_two"],
            "metadata": {
                "meaning_star": {
                    "grammar_refs": ["concept_digit_two"],
                    "surface_forms": {},
                }
            },
        },
        {
            "id": "concept_digit_two",
            "selection_role": "answer",
            "sovereign_route_exempt": True,
            "taxonomy_refs": ["grammar_binary_op_infix"],
            "metadata": {
                "meaning_star": {
                    "taxonomy_refs": ["grammar_binary_op_infix"],
                    "surface_forms": {},
                }
            },
        },
    ]
    runtime._validate_route_link_coverage(stars)
    runtime._validate_symlink_closure(stars)


def test_phase6b1_symlink_closure_enforced() -> None:
    runtime = _runtime_stub()
    with pytest.raises(ValueError, match="sovereign_build_symlink_closure_invalid"):
        runtime._validate_symlink_closure(
            [
                {
                    "id": "broken_exempt_router",
                    "selection_role": "router",
                    "sovereign_route_exempt": True,
                    "grammar_refs": ["missing_peer"],
                    "metadata": {
                        "meaning_star": {
                            "grammar_refs": ["missing_peer"],
                            "surface_forms": {},
                        }
                    },
                }
            ]
        )


def test_phase6b1_crafter_rows_all_marked_exempt() -> None:
    rows = build_foundational_star_crafter_outputs(embedder=_FakeMatryoshkaEmbedder())
    assert rows
    for row in rows:
        assert row["sovereign_route_exempt"] is True
        assert dict(row.get("metadata") or {}).get("sovereign_route_exempt") is True


@pytest.mark.gpu
def test_phase6b1_full_boot_integration() -> None:
    kv = Knowledgeverse()
    runtime = kv._get_sovereign_hot_path()
    assert runtime.star_table.star_count > 0
    assert runtime.program_table.size_bytes > 0
    crafted = {str(star.get("id") or ""): dict(star) for star in getattr(runtime, "_host_stars", [])}
    assert "math_operator_addition" in crafted
    assert crafted["math_operator_addition"]["sovereign_route_exempt"] is True
