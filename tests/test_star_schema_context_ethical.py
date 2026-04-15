from __future__ import annotations

import re
import uuid
from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import canonical_context_id
from knowledge3d.knowledgeverse.galaxy_vram_table import (
    STAR_CONTEXT_ID_OFFSET,
    STAR_ETHICAL_TRIT_OFFSET,
    STAR_RECORD_BYTES,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from knowledge3d.knowledgeverse.star_materializer_bridge import (
    FINALIZED_CATALOG_INPUT_ENTRY_BYTES,
    FINALIZED_CATALOG_INPUT_ENTRY_FORMAT,
    RAW_CATALOG_INPUT_ENTRY_BYTES,
    RAW_CATALOG_INPUT_ENTRY_FORMAT,
)


def test_canonical_context_id_is_stable_nonzero_for_scoped_contexts() -> None:
    left = canonical_context_id("Brazil", "Imperial Era")
    right = canonical_context_id("brazil", "imperial-era")
    assert left == right
    assert left != 0
    assert left == (uuid.uuid5(uuid.NAMESPACE_URL, "context::brazil::imperial_era").int & 0xFFFFFFFF)
    assert canonical_context_id("", "") == 0


def test_meaning_star_context_and_ethical_trit_round_trip() -> None:
    star = MeaningCentricStar(
        star_id="ethical_context_star",
        meaning_rpn="CONTEXT TEST",
        domain="Test",
        context_id=canonical_context_id("EU", "2026"),
        ethical_trit=7,
    )
    payload = star.to_dict()
    assert payload["context_id"] == canonical_context_id("EU", "2026")
    assert payload["ethical_trit"] == 1

    restored = MeaningCentricStar.from_dict(payload)
    assert restored.context_id == payload["context_id"]
    assert restored.ethical_trit == 1


def test_star_vram_context_fields_are_tail_aligned() -> None:
    assert STAR_RECORD_BYTES == 408
    assert STAR_CONTEXT_ID_OFFSET == 400
    assert STAR_ETHICAL_TRIT_OFFSET == 404
    assert RAW_CATALOG_INPUT_ENTRY_BYTES == 384
    assert FINALIZED_CATALOG_INPUT_ENTRY_BYTES == 360
    assert RAW_CATALOG_INPUT_ENTRY_FORMAT.endswith("Ib3x12x")
    assert FINALIZED_CATALOG_INPUT_ENTRY_FORMAT.endswith("Ib3x12x")


def test_cuda_star_record_stride_has_single_source_of_truth() -> None:
    cuda_dir = Path("knowledge3d/cranium/cuda")
    pattern = re.compile(r"\b(STAR_RECORD_BYTES|K3D_STAR_RECORD_BYTES)\b")
    offenders: list[str] = []
    for path in sorted(cuda_dir.glob("*.cu")) + sorted(cuda_dir.glob("*.cuh")):
        if path.name == "device_functions.cuh":
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path))
    assert offenders == []


def test_fused_trm_ptx_matches_live_galaxy_star_stride() -> None:
    source = Path("knowledge3d/cranium/ptx/trm_step_fused.cu").read_text(encoding="utf-8")
    match = re.search(r"#define\s+K3D_GALAXY_STAR_RECORD_BYTES\s+(\d+)u", source)
    assert match is not None
    assert int(match.group(1)) == STAR_RECORD_BYTES
