from __future__ import annotations

from pathlib import Path


def cluster1_fixture_text() -> str:
    return "\n".join(
        [
            "#### concept_precedence",
            "- **canonical_id**: `concept_arithmetic_precedence`",
            "- **is_a**: `concept_operation_order`",
            "- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.parenthesis][TPACK 1]`",
            "- **symlinks**: `star.symbol.parenthesis`",
            "- **surface_forms**:",
            '  - en: "arithmetic precedence"',
            '  - pt: "precedência aritmética"',
            '  - es: "precedencia aritmética"',
            '  - fr: "priorité arithmétique"',
            '  - de: "arithmetische priorität"',
            '  - it: "precedenza aritmetica"',
            '  - ja: "算術の優先順位"',
            '  - zh: "算术优先级"',
            '  - ru: "арифметический приоритет"',
            "- **saudades**: `false`",
            "",
            "#### rule_pemdas",
            "- **canonical_id**: `rule_order_of_operations_pemdas`",
            "- **is_a**: `concept_arithmetic_precedence`",
            "- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.parenthesis][GALAXY_LOOKUP star.symbol.plus][TPACK 2]`",
            "- **symlinks**: `star.symbol.parenthesis, star.symbol.plus, concept::arithmetic_precedence`",
            "- **surface_forms**:",
            '  - en: "PEMDAS"',
            '  - pt: "PEMDAS"',
            '  - es: "jerarquía de operaciones"',
            '  - fr: "ordre des opérations"',
            '  - de: "Punkt vor Strich"',
            '  - it: "ordine delle operazioni"',
            '  - ja: "計算の順序"',
            '  - zh: "运算顺序"',
            '  - ru: "порядок действий"',
            "- **saudades**: `true`",
        ]
    )


class FakeCanonicalLookup:
    def __init__(self, *, preset_star_ids: set[str] | None = None, drop_edges: set[str] | None = None) -> None:
        self.records: dict[tuple[str, str], dict[str, object]] = {}
        self.star_ids = set(preset_star_ids or set())
        self.drop_edges = set(drop_edges or set())

    def ensure_collection(self) -> None:
        return None

    def exists(self, *, kind: str, key: str) -> bool:
        return (kind, key) in self.records

    def find_star_id(self, *, kind: str, key: str) -> str:
        return str(self.records[(kind, key)]["star_id"])

    def star_id_exists(self, star_id: str) -> bool:
        return str(star_id) in self.star_ids

    def register(self, *, kind: str, key: str, star_id: str, metadata=None) -> str:
        if kind == "math_symlink" and key in self.drop_edges:
            return str(star_id)
        self.records[(kind, key)] = {
            "kind": kind,
            "key": key,
            "star_id": str(star_id),
            "metadata": dict(metadata or {}),
        }
        self.star_ids.add(str(star_id))
        return str(star_id)

    def _scroll_exact(self, *, kind: str, key: str) -> dict[str, object] | None:
        return dict(self.records.get((kind, key), {})) or None


def write_fixture(tmp_path: Path, text: str | None = None, name: str = "fixture_cluster1.md") -> Path:
    path = tmp_path / name
    path.write_text(text or cluster1_fixture_text(), encoding="utf-8")
    return path
