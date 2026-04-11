from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
from knowledge3d.knowledgeverse.gpu_task_dispatch import _navigate_galaxy_ref, cpu_reference_dispatch
from knowledge3d.knowledgeverse.vram_task_buffer import EMBEDDING_DIMS


def test_galaxy_top8_finds_action_and_translation_stars():
    galaxy_stars = build_foundational_galaxy_table()
    encoder = ARC3FrameEncoder()

    frame = [[0] * 8 for _ in range(8)]
    frame[1][4] = 2
    frame_emb = encoder.encode(frame)
    _, top_indices, _ = _navigate_galaxy_ref(frame_emb, galaxy_stars, route_family="GAME_2D")
    top_ids = {str(galaxy_stars[index]["id"]) for index in top_indices}

    assert top_indices, top_indices
    assert all(str(galaxy_stars[index].get("route_family") or "") == "GAME_2D" for index in top_indices)
    assert {
        "game2d_router",
        "game2d_surface_bridge",
        "game2d_action_move_up",
        "game2d_transform_inference_executor",
    } & top_ids, top_ids


def test_cpu_reference_dispatch_reports_top_galaxy_diagnostics():
    galaxy_stars = build_foundational_galaxy_table()
    encoder = ARC3FrameEncoder()

    frame = [[0] * 8 for _ in range(8)]
    frame[1][4] = 2
    frame_emb = encoder.encode(frame)

    row = cpu_reference_dispatch(
        [
            {
                "type": "ARC3_TASK",
                "query_embedding": frame_emb,
                "option_embeddings": [[0.0] * EMBEDDING_DIMS for _ in range(7)],
                "subject": "arc3",
                "domain_hint": "arc3_interactive",
            }
        ],
        galaxy_stars=galaxy_stars,
    )[0]

    assert row["top_galaxy_star_indices"]
    assert len(row["top_galaxy_star_indices"]) == len(row["top_galaxy_star_scores"])


def test_no_sentence_transformers_in_frame_encoder():
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "knowledge3d/cranium/cuda/arc3_frame_encoder.cu").read_text(encoding="utf-8")
    assert "sentence_transform" not in content.lower()
    assert "import torch" not in content
    assert "import numpy" not in content
    assert "hash_token_into_embedding_arc3" not in content
