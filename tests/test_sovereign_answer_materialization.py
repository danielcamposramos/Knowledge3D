from __future__ import annotations

import ctypes

import numpy as np
import pytest

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.cranium.bridges.trm_step_fused_bridge import GALAXY_EMBEDDING_DIMS, TRMStepFusedBridge
from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.galaxy_vram_table import GalaxyVRAMTable


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _device_vector(values: list[float]):
    host_type = ctypes.c_float * len(values)
    host = host_type(*[float(value) for value in values])
    ptr = loader.gpu_malloc(ctypes.sizeof(host))
    loader.memcpy_htod(ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))
    return ptr


class _FailingDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("daemon fallback should not run on the sovereign materialization path")


@pytest.mark.gpu
def test_cosine_decode_y_new_selects_answer_eligible_star() -> None:
    _ensure_cuda()
    table = GalaxyVRAMTable(max_stars=4)
    bridge = TRMStepFusedBridge()
    d_y_new = None
    try:
        x_axis = [1.0] + [0.0] * (GALAXY_EMBEDDING_DIMS - 1)
        y_axis = [0.0, 1.0] + [0.0] * (GALAXY_EMBEDDING_DIMS - 2)
        table.load_stars(
            [
                {
                    "id": "ineligible_exact_match",
                    "embedding": y_axis,
                    "selection_role": "answer",
                    "answer_eligible": False,
                    "metadata": {"answer_text": "do not select"},
                },
                {
                    "id": "answer_star_paris",
                    "embedding": y_axis,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "galaxy_id": 77,
                    "metadata": {"answer_text": "Paris"},
                },
                {
                    "id": "weaker_answer",
                    "embedding": x_axis,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "metadata": {"answer_text": "London"},
                },
            ]
        )
        d_y_new = _device_vector(y_axis + [0.0] * (512 - GALAXY_EMBEDDING_DIMS))
        bridge.bind_galaxy_table(
            table.gpu_ptr,
            table.star_count,
            embedding_dims=GALAXY_EMBEDDING_DIMS,
            host_stars=table.read_stars(),
        )

        result = bridge.decode_y_new_top_star(d_y_new)

        assert result["answer_materialized"] is True
        assert result["top_star_idx"] == 1
        assert result["top_star_galaxy_id"] == 77
        assert result["top_star_role"] == 4
        assert result["top_star"]["metadata"]["answer_text"] == "Paris"
        assert result["top_star_score"] == pytest.approx(1.0, abs=1e-6)
    finally:
        if d_y_new is not None:
            loader.gpu_free(d_y_new)
        bridge.cleanup()
        table.close()


@pytest.mark.gpu
def test_full_tablet_to_answer_uses_bridge_star_materialization(tmp_path) -> None:
    _ensure_cuda()
    table = GalaxyVRAMTable(max_stars=4)
    bridge = TRMStepFusedBridge()
    allocations = []
    try:
        x_axis = [1.0] + [0.0] * (GALAXY_EMBEDDING_DIMS - 1)
        y_axis = [0.0, 1.0] + [0.0] * (GALAXY_EMBEDDING_DIMS - 2)
        table.load_stars(
            [
                {
                    "id": "wrong_answer",
                    "embedding": x_axis,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "metadata": {"answer_text": "3"},
                },
                {
                    "id": "correct_answer",
                    "embedding": y_axis,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "metadata": {"answer_text": "4"},
                },
            ]
        )
        bridge.bind_galaxy_table(
            table.gpu_ptr,
            table.star_count,
            embedding_dims=GALAXY_EMBEDDING_DIMS,
            host_stars=table.read_stars(),
        )

        q = np.zeros(512, dtype=np.float32)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)
        z_new = np.zeros(512, dtype=np.float32)
        y_new = np.zeros(512, dtype=np.float32)
        workspace = np.zeros(4096, dtype=np.float32)
        W1 = np.zeros((1024, 512), dtype=np.float32)
        W2 = np.zeros((512, 1024), dtype=np.float32)
        W3 = np.zeros((1024, 512), dtype=np.float32)
        W4 = np.zeros((512, 1024), dtype=np.float32)
        W1[1, 1] = 10.0
        W2[1, 1] = 0.1
        W3[1, 1] = 10.0
        W4[1, 1] = 0.1

        for array in (q, y, z, W1.ravel(), W2.ravel(), W3.ravel(), W4.ravel(), z_new, y_new, workspace):
            allocations.append(_device_vector([float(value) for value in array]))
        bridge.bind_query_runtime_buffers(
            q_ptr=allocations[0],
            y_ptr=allocations[1],
            z_ptr=allocations[2],
            W1_ptr=allocations[3],
            W2_ptr=allocations[4],
            W3_ptr=allocations[5],
            W4_ptr=allocations[6],
            z_new_ptr=allocations[7],
            y_new_ptr=allocations[8],
            workspace_ptr=allocations[9],
        )

        tablet = HeadlessTabletMPC(
            command_handler=_FailingDaemon(),
            bridge=bridge,
            storage_root=tmp_path,
        )
        envelope = TabletIngest.math_problem(
            task_id="full_tablet_to_answer",
            question="What is 2 + 2?",
            expected_answer="4",
        )
        envelope.metadata["query_embedding_512"] = [0.0, 1.0] + [0.0] * 510

        result = tablet.submit(envelope)
    finally:
        for ptr in allocations:
            loader.gpu_free(ptr)
        bridge.cleanup()
        table.close()

    assert result["tablet_contract"]["sovereign_path"] == "tablet_bridge_ring"
    assert result["response"]["task_result"]["top_star_idx"] == 1
    assert result["response"]["task_result"]["answer_materialized"] is True
    assert result["emitted"]["answer_text"] == "4"
    assert result["emitted"]["numeric_answer"] == 4.0
    assert result["emitted"]["correct"] is True
