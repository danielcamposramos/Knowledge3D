#!/usr/bin/env python3
"""Generate a persistent multimodal execution journal batch and promotion report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from knowledge3d.cranium.bridges.procedural_material_bridge import SurfaceMaterialCandidate
from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.objects_3d_galaxy import bootstrap_3d_objects_galaxy
from knowledge3d.knowledgeverse.reality_galaxy import bootstrap_reality_galaxy

from scripts.build_tool_promotion_report import _load_json_object, _load_rows, build_report


def _material_candidates() -> tuple[SurfaceMaterialCandidate, SurfaceMaterialCandidate]:
    cool = SurfaceMaterialCandidate(
        material_id="cool",
        name="Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    return cool, warm


def _sample_grid() -> list[list[int]]:
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1
    return grid


def _sample_audio_signal(length: int = 1024) -> TernaryVector:
    return TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(length)])


def _house_room_events() -> list[dict[str, Any]]:
    return [
        {
            "tool_id": "tool_house_library_scene_v1",
            "query_context": "what I know",
            "quality_signal": 0.91,
            "ternary_quality": 1,
            "outcome": 1,
            "timestamp_us": 10,
            "chain_tool_ids": ["tool_geom_profile_lathe_mesh_v1", "tool_fusion_surface_material_projection_v1"],
        },
        {
            "tool_id": "tool_house_garden_scene_v1",
            "query_context": "what I am learning",
            "quality_signal": 0.56,
            "ternary_quality": 0,
            "outcome": 0,
            "timestamp_us": 20,
            "chain_tool_ids": ["tool_signal_audio_spectrogram_v1", "tool_signal_spectrogram_surface_v1"],
        },
        {
            "tool_id": "tool_house_museum_scene_v1",
            "query_context": "my history",
            "quality_signal": 0.18,
            "ternary_quality": -1,
            "outcome": -1,
            "timestamp_us": 30,
            "chain_tool_ids": ["tool_fusion_surface_material_projection_v1"],
        },
    ]


def _replay_entries() -> list[dict[str, Any]]:
    return [
        {"timestamp": 1, "action_type": "NAV_MOVE", "quality_signal": 0.82, "ternary_quality": 1, "final_confidence": 0.82},
        {"timestamp": 2, "action_type": "DIALOGUE", "quality_signal": 0.74, "ternary_quality": 1, "final_confidence": 0.74},
        {"timestamp": 3, "action_type": "WRITE_MEM", "quality_signal": 0.88, "ternary_quality": 1, "final_confidence": 0.88},
    ]


def _compose_program(kv: Knowledgeverse, *, query_text: str, specialist: str) -> dict[str, Any]:
    rows = kv.galaxy_manager.query(
        query_text=query_text,
        specialist=specialist,
        top_k=8,
        galaxies=["Tool"],
    )
    return kv.trm_navigator.compose(
        query=query_text,
        patterns=rows,
        specialist=specialist,
    )


def _ensure_storage_root(storage_root: Path, *, fresh: bool) -> None:
    if fresh and storage_root.exists():
        shutil.rmtree(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)


def run_batch(
    *,
    storage_root: Path,
    iterations: int = 8,
    fresh: bool = False,
    build_promotion_report: bool = True,
) -> dict[str, Any]:
    _ensure_storage_root(storage_root, fresh=fresh)
    bootstrap_reality_galaxy(storage_root=storage_root)
    bootstrap_3d_objects_galaxy(storage_root=storage_root)
    kv = Knowledgeverse(storage_root=storage_root)

    cool, warm = _material_candidates()
    grid = _sample_grid()
    room_events = _house_room_events()
    replay_entries = _replay_entries()

    signal_program = _compose_program(kv, query_text="audio signal surface material fusion", specialist="audio")
    contour_program = _compose_program(kv, query_text="contour textured mesh surface material projection", specialist="visual")
    replay_program = _compose_program(kv, query_text="house replay journal scene playback", specialist="visual")
    library_program = _compose_program(kv, query_text="knowledge library settled scene playback", specialist="visual")
    garden_program = _compose_program(kv, query_text="learning growing garden scene playback", specialist="visual")
    museum_program = _compose_program(kv, query_text="history archive failures lessons museum scene playback", specialist="visual")
    tour_program = _compose_program(kv, query_text="house tour overview all scene playback", specialist="visual")

    generation_queries = [
        ("simulate projectile from origin with initial velocity 10 m/s at 45 degrees", "Reality", "Grammar"),
        ("generate L-system plant with axiom F and rule F->F[+F]F[-F]F for 3 iterations", "Reality", "3DObjects"),
        ("generate UV sphere mesh with radius 5 and 16 subdivisions", "3DObjects", "3DObjects"),
        ("visualize electric field for point charge at origin with magnitude 1.0", "Reality", "Drawing"),
        ("generate textured mesh from contour with surface material", "3DObjects", "3DObjects"),
    ]

    for idx in range(max(1, int(iterations))):
        clip_id = f"signal_batch_{idx}"
        kv.trm_navigator.execute(
            signal_program,
            input_data={
                "clip_id": clip_id,
                "audio_signal": _sample_audio_signal(),
                "material_candidates": (cool, warm),
                "negative_materials": (warm,),
                "frame_size": 256,
                "threshold": 0.15,
                "displacement_gain": 0.4,
                "preview_size": 32,
            },
        )
        kv.trm_navigator.execute(
            contour_program,
            input_data={
                "drawing_contour": grid,
                "surface_material": cool,
                "material_candidates": (cool, warm),
                "color": 1,
                "preview_size": 32,
            },
        )
        kv.trm_navigator.execute(
            replay_program,
            input_data={
                "replay_entries": replay_entries,
                "frame_count": 4,
                "scene_layout": "golden_orbit",
            },
        )
        kv.trm_navigator.execute(
            library_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            garden_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            museum_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            tour_program,
            input_data={"execution_events": room_events, "max_events_per_room": 3},
        )
        for query_text, source_galaxy, target_galaxy in generation_queries:
            kv.trm_navigator.generate_from_procedural(
                query=query_text,
                source_galaxy=source_galaxy,
                target_galaxy=target_galaxy,
                store_result=True,
            )

    logs_dir = storage_root / "logs"
    checkpoints_dir = storage_root / "checkpoints"
    results_dir = storage_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pressure_path = logs_dir / "tool_promotion_pressure.jsonl"
    events_path = logs_dir / "execution_events.jsonl"
    grammar_path = logs_dir / "execution_grammar_patterns.jsonl"
    quality_path = checkpoints_dir / "execution_quality_tracker.json"

    pressure_rows = _load_rows(pressure_path)
    event_rows = _load_rows(events_path)
    grammar_rows = _load_rows(grammar_path)
    quality_state = _load_json_object(quality_path)

    report_path = results_dir / "tool_promotion_report.json"
    report: dict[str, Any] | None = None
    if build_promotion_report:
        report = build_report(
            pressure_rows,
            source_path=pressure_path,
            event_rows=event_rows,
            event_source_path=events_path,
            grammar_rows=grammar_rows,
            grammar_source_path=grammar_path,
            quality_state=quality_state,
            quality_state_path=quality_path,
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    top_candidate = (report or {}).get("candidate_summary", {}).get("top_candidate")
    summary = {
        "storage_root": str(storage_root),
        "iterations": int(iterations),
        "pressure_rows": int(len(pressure_rows)),
        "event_rows": int(len(event_rows)),
        "grammar_rows": int(len(grammar_rows)),
        "quality_state_exists": bool(quality_path.exists()),
        "report_path": str(report_path),
        "top_candidate": top_candidate,
    }
    summary_path = results_dir / "execution_journal_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local/runtime_execution_journal_batch",
        help="Persistent storage root for journal generation.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of batch iterations to run.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the target storage root before generating the batch.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Generate journals only, without building the promotion report.",
    )
    args = parser.parse_args()

    summary = run_batch(
        storage_root=Path(args.storage_root),
        iterations=max(1, int(args.iterations)),
        fresh=bool(args.fresh),
        build_promotion_report=not bool(args.skip_report),
    )
    print(
        "[execution-journal-batch] "
        f"pressure_rows={summary['pressure_rows']} "
        f"event_rows={summary['event_rows']} "
        f"grammar_rows={summary['grammar_rows']} "
        f"storage_root={summary['storage_root']}"
    )
    top_candidate = summary.get("top_candidate")
    if isinstance(top_candidate, dict):
        print(
            "[execution-journal-batch] "
            f"top_candidate={top_candidate.get('name')} "
            f"priority={top_candidate.get('promotion_priority_score')} "
            f"readiness={top_candidate.get('promotion_readiness_score')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
