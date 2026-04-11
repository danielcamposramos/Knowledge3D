from __future__ import annotations

from knowledge3d.knowledgeverse.galaxy_vram_table import (
    EMBEDDING_DIMS,
    STAR_FLAG_LEARNABLE,
    GalaxyVRAMTable,
)
from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch
from knowledge3d.knowledgeverse.persistent_brain import PersistentBrainState
from knowledge3d.knowledgeverse.sleep_time_micro import SleepTimeMicro
from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer

from tests.foundational_test_utils import build_resolved_foundational_stars


def test_galaxy_vram_table_round_trip():
    stars = build_resolved_foundational_stars()
    table = GalaxyVRAMTable(max_stars=128)
    try:
        count = table.load_stars(stars)
        roundtrip = table.read_stars()
    finally:
        table.close()

    assert count >= 80
    assert len(roundtrip) == count
    assert all(len(row["embedding"]) == EMBEDDING_DIMS for row in roundtrip[:7])
    assert any(abs(float(value)) > 1.0e-8 for value in roundtrip[0]["embedding"][32:])
    assert any(row["component_refs"] for row in roundtrip[:7])
    assert (roundtrip[0]["flags"] & STAR_FLAG_LEARNABLE) != 0
    assert roundtrip[7]["star_type"] == 1
    assert any(row["star_type"] == 5 for row in roundtrip)
    assert any(row["star_type"] == 6 for row in roundtrip)


def test_foundational_galaxy_action_links_to_math_and_spatial():
    stars = build_resolved_foundational_stars()
    id_to_index = {str(star["id"]): index for index, star in enumerate(stars)}
    move_up_refs = [stars[index]["id"] for index in stars[id_to_index["atom:action:move_up"]]["component_refs"]]
    transform_detect_refs = [stars[index]["id"] for index in stars[id_to_index["transform_detect"]]["component_refs"]]

    assert "translate_2d" in move_up_refs
    assert "translation_concept" in move_up_refs
    assert "vec2_add" in move_up_refs
    assert "delta_op" in transform_detect_refs
    assert "symmetry_x" in transform_detect_refs


def test_sleep_time_micro_updates_learnable_galaxy_star():
    stars = build_resolved_foundational_stars()
    table = GalaxyVRAMTable(max_stars=128)
    brain = PersistentBrainState()
    task_buffer = VRAMTaskBuffer(max_tasks=1)
    sleep = SleepTimeMicro()
    try:
        table.load_stars(stars)
        table._host_stars = []
        before = table.read_stars(1)[0]["embedding"]
        task = {
            "type": "GAME_2D",
            "query_embedding": [1.0] + ([0.0] * (EMBEDDING_DIMS - 1)),
            "option_embeddings": [[1.0 if i == j else 0.0 for i in range(EMBEDDING_DIMS)] for j in range(7)],
            "subject": "arc3_subject",
            "domain_hint": "arc3_domain",
        }
        task_buffer.bulk_load([task])
        GPUTaskDispatch().launch(
            task_buffer,
            1,
            brain_ptr=brain.gpu_ptr,
            star_table=table,
        )
        sleep.consolidate(
            brain.gpu_ptr,
            1,
            galaxy_ptr=table.gpu_ptr,
            chosen_star_index=0,
        )
        table._host_stars = []
        after = table.read_stars(1)[0]["embedding"]
    finally:
        task_buffer.close()
        brain.close()
        table.close()

    assert before != after
