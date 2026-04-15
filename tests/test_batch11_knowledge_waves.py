from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion import canonical_curriculum_loader
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm
from knowledge3d.ingestion.hs_curriculum_parser import parse_curriculum_file
from scripts.ingest_hs_curriculum_remaining import ingest_remaining_curriculum
from scripts.run_headless_tablet_benchmarks import _attach_kernel_coverage_audit, _run_batch11_warmup_probes


def test_batch11_parser_extracts_priority_wave_rows_from_repo_sources() -> None:
    natural = parse_curriculum_file(Path("TEMP/KIMI_HS_NATURAL_SCIENCES_PHYS_CHEM_BIO_2026-04-13.md"))
    history = parse_curriculum_file(Path("TEMP/KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_2026-04-13.md"))
    applied = parse_curriculum_file(Path("TEMP/KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_2026-04-13.md"))
    arc = parse_curriculum_file(Path("TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md"))

    assert len(natural.rows) >= 1
    assert len(history.rows) >= 10
    assert len(applied.rows) >= 10
    assert len(arc.rows) >= 30
    assert arc.rows[0].domain == "arc"


def test_batch11_dry_run_summary_includes_arc_parallel_track() -> None:
    summary = ingest_remaining_curriculum(object(), write=False)

    assert summary["phase"] == "dry_run"
    files = dict(summary["files"])
    assert "KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md" in files
    assert int(files["KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md"]["rows"]) >= 30


def test_batch11_kernel_coverage_audit_attaches_inventory() -> None:
    coverage = _attach_kernel_coverage_audit(
        {
            "program_id_coverage": {"gpu_task_dispatch_sovereign": 3, "trm_step_fused": 1},
            "opcode_coverage": {"nine_chain_swarm_kernel": 1},
        }
    )

    audit = dict(coverage["kernel_coverage_audit"])
    assert audit["inventory_total"] >= 50
    assert "trm_step_fused" in audit["fired_kernels"]
    assert "nine_chain_swarm_kernel" in audit["fired_kernels"]
    assert "gpu_task_dispatch_sovereign" in audit["unmatched_trace_tokens"]


def test_batch11_warmup_probes_build_valid_session_tapes(tmp_path: Path) -> None:
    class _FakeTablet:
        def __init__(self) -> None:
            self.surface_kinds: list[str] = []

        def run_tape_session(self, tape, *, enforce_preflight: bool = True):
            self.surface_kinds.append(str(tape.surface_kind))
            frame = tape.frames[0]
            return {
                "results": [
                    {
                        "emitted": {
                            "trace_star_ids": ["router_demo", "fact_demo"],
                            "answer_materialized": True,
                            "route_family": str(tape.surface_kind),
                            "route": {"specialist": str(frame.envelope.specialist)},
                            "task_result": {
                                "winner_role": str(frame.envelope.specialist),
                            },
                        }
                    }
                ]
            }

    tablet = _FakeTablet()
    summary = _run_batch11_warmup_probes(tablet=tablet, log_dir=tmp_path)

    assert sorted(tablet.surface_kinds) == ["GAME_2D", "MATH", "QUESTION", "QUESTION"]
    assert sorted(summary) == ["GAME_2D", "LHE", "MATH", "MMLU"]


def test_batch11_loader_preserves_curriculum_metadata(monkeypatch) -> None:
    star = MeaningCentricStar(
        star_id="method_transform_reflect_horizontal",
        meaning_class="method",
        meaning_rpn="[RECALL grid][TPACK FLIP_H][STORE reflected]",
        domain="arc",
        surface_forms={"en": SurfaceForm(word_ref="horizontal reflection")},
    )
    payloads = [
        {
            "metadata": {
                "meaning_star": star.to_dict(),
                "subkind": "arc_reasoning_primitives",
                "source_file": "TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md",
                "source_line": 406,
                "rpn_sketch": "[RECALL grid][TPACK FLIP_H][STORE reflected]",
                "symlink_refs": ["star.concept.grid"],
                "surface_forms_raw": {"en": "horizontal reflection"},
                "is_a": ["concept_grid_transform"],
                "domain": "arc",
            }
        }
    ]

    monkeypatch.setattr(
        canonical_curriculum_loader,
        "_iter_meaning_star_payloads",
        lambda _lookup: payloads,
    )

    class _FakeManager:
        def __init__(self) -> None:
            self.rows: list[tuple[str, dict[str, object]]] = []

        def upsert_entry(self, galaxy_name: str, entry: dict[str, object]) -> str:
            self.rows.append((galaxy_name, dict(entry)))
            return "inserted"

    class _FakeKnowledgeverse:
        def __init__(self) -> None:
            self.galaxy_manager = _FakeManager()

        def ensure_default_galaxies_loaded(self) -> dict[str, int]:
            return {}

    fake = _FakeKnowledgeverse()
    summary = canonical_curriculum_loader.load_canonical_curriculum_into_knowledgeverse(fake)

    assert summary["inserted"] == 1
    galaxy_name, entry = fake.galaxy_manager.rows[0]
    assert galaxy_name == "Tool"
    metadata = dict(entry["metadata"])
    assert metadata["rpn_sketch"] == "[RECALL grid][TPACK FLIP_H][STORE reflected]"
    assert metadata["curriculum_metadata"]["rpn_sketch"] == "[RECALL grid][TPACK FLIP_H][STORE reflected]"
    assert metadata["curriculum_metadata"]["source_line"] == 406
