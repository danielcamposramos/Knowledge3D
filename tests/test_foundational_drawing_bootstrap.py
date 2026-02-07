from __future__ import annotations

import json
import os

from knowledge3d.knowledgeverse import Knowledgeverse


def test_drawing_foundational_bootstrap_count(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing = kv.galaxy_manager.get_galaxy("Drawing")
    foundational = [
        e for e in drawing.entries if str(e.get("type")) == "foundational_primitive"
    ]
    assert len(foundational) >= 100


def test_drawing_foundational_cross_modal_symlinks(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing = kv.galaxy_manager.get_galaxy("Drawing")
    by_id = {str(e.get("id")): e for e in drawing.entries}

    cubic = by_id["cubic_bezier_eval"]
    wave = by_id["sine_wave_as_curve"]
    mat = by_id["mat4_mul_vec4"]

    assert cubic["metadata"]["symlink"] == "character_galaxy"
    assert wave["metadata"]["symlink"] == "audio_galaxy"
    assert mat["metadata"]["symlink"] == "math_galaxy"


def test_drawing_foundational_bootstrap_logs_single_event(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    _ = kv.galaxy_manager.get_galaxy("Drawing")
    events = [
        ev for ev in kv.shadow_copy.event_buffer if ev.get("type") == "drawing_foundational_bootstrap"
    ]
    assert len(events) == 1
    assert int(events[0]["data"]["loaded_entries"]) >= 100


def test_drawing_bootstrap_loads_optional_ollama_enrichment(tmp_path):
    enrich_path = tmp_path / "ollama_enrichment.jsonl"
    row = {
        "id": "ollama_extra_curve",
        "name": "Ollama Extra Curve",
        "type": "foundational_primitive",
        "domain": "drawing",
        "category": "llm_enrichment",
        "rpn_program": "P0 P1 P2 0.5 QUADRATIC_BEZIER_EVAL",
        "metadata": {"source": "ollama_enrichment", "symlink": "character_galaxy"},
    }
    enrich_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    old = os.environ.get("K3D_DRAWING_ENRICHED_JSONL")
    os.environ["K3D_DRAWING_ENRICHED_JSONL"] = str(enrich_path)
    try:
        kv = Knowledgeverse(storage_root=tmp_path / "kv")
        drawing = kv.galaxy_manager.get_galaxy("Drawing")
    finally:
        if old is None:
            os.environ.pop("K3D_DRAWING_ENRICHED_JSONL", None)
        else:
            os.environ["K3D_DRAWING_ENRICHED_JSONL"] = old

    ids = {str(e.get("id")) for e in drawing.entries}
    assert "ollama_extra_curve" in ids


def test_drawing_bootstrap_supports_multiple_enrichment_files(tmp_path):
    p1 = tmp_path / "e1.jsonl"
    p2 = tmp_path / "e2.jsonl"
    p1.write_text(
        json.dumps(
            {
                "id": "extra_a",
                "rpn_program": "A B ADD",
                "metadata": {"symlink": "math_galaxy"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(
            {
                "id": "extra_b",
                "rpn_program": "X Y MUL",
                "metadata": {"symlink": "character_galaxy"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    old = os.environ.get("K3D_DRAWING_ENRICHED_JSONL")
    os.environ["K3D_DRAWING_ENRICHED_JSONL"] = f"{p1},{p2}"
    try:
        kv = Knowledgeverse(storage_root=tmp_path / "kv_multi")
        drawing = kv.galaxy_manager.get_galaxy("Drawing")
    finally:
        if old is None:
            os.environ.pop("K3D_DRAWING_ENRICHED_JSONL", None)
        else:
            os.environ["K3D_DRAWING_ENRICHED_JSONL"] = old

    ids = {str(e.get("id")) for e in drawing.entries}
    assert "extra_a" in ids
    assert "extra_b" in ids
