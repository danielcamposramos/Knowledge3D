from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingest_hs_math_cluster1 import Batch8DriverError, run_cluster1_ingestion
from tests._batch8_helpers import FakeCanonicalLookup, write_fixture


def test_three_pass_driver_distinguishes_forward_ref_from_hard_miss(tmp_path: Path) -> None:
    source = write_fixture(
        tmp_path,
        text="\n".join(
            [
                "#### concept_precedence",
                "- **canonical_id**: `concept_arithmetic_precedence`",
                "- **is_a**: `concept_operation_order`",
                "- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.parenthesis]`",
                "- **symlinks**: `star.symbol.parenthesis`",
                "- **surface_forms**:",
                '  - en: "precedence"',
                '  - pt: "precedência"',
                '  - es: "precedencia"',
                '  - fr: "priorité"',
                '  - de: "Priorität"',
                '  - it: "precedenza"',
                '  - ja: "優先順位"',
                '  - zh: "优先级"',
                '  - ru: "приоритет"',
                "",
                "#### rule_bad",
                "- **canonical_id**: `rule_bad`",
                "- **is_a**: `concept_arithmetic_precedence`",
                "- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.parenthesis]`",
                "- **symlinks**: `concept::arithmetic_precedence, symbol::missing_alias`",
                "- **surface_forms**:",
                '  - en: "bad"',
                '  - pt: "bad"',
                '  - es: "bad"',
                '  - fr: "bad"',
                '  - de: "bad"',
                '  - it: "bad"',
                '  - ja: "bad"',
                '  - zh: "bad"',
                '  - ru: "bad"',
            ]
        ),
    )
    with pytest.raises(Batch8DriverError) as exc:
        run_cluster1_ingestion(FakeCanonicalLookup(), source=source, write=False)
    assert exc.value.code == 1


def test_pass3_confirmation_failure_uses_exit_code_3(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    lookup = FakeCanonicalLookup(drop_edges={"math_rule_order_of_operations_pemdas::math_symbol_plus_sign"})
    with pytest.raises(Batch8DriverError) as exc:
        run_cluster1_ingestion(lookup, source=source, write=True)
    assert exc.value.code == 3
