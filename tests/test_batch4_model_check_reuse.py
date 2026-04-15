from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.bridges.model_check_reuse_bridge import ModelCheckReuseBridge
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_model_check_reuse_kernel_compiles() -> None:
    ptx = compile_cuda_file(
        "knowledge3d/cranium/cuda/model_check_reuse.cu",
        arch="sm_86",
        use_fast_math=False,
    )
    assert ".entry k3d_model_check_reuse" in ptx


def test_model_check_reuse_returns_success_on_tiny_valid_graph() -> None:
    bridge = ModelCheckReuseBridge()
    result = bridge.run(
        state_props=[0b0001, 0b0010, 0b0100],
        adjacency=[
            1, 0xFFFFFFFF,
            2, 0xFFFFFFFF,
            0xFFFFFFFF, 0xFFFFFFFF,
        ],
        num_states=3,
        max_degree=2,
        root_state=0,
        target_mask=0b0100,
        forbidden_mask=0b1000,
    )
    assert result.status == bridge.STATUS_PASS
    assert result.witness_state == 2
    assert result.visited_count >= 1


def test_model_check_reuse_returns_fail_on_violating_graph() -> None:
    bridge = ModelCheckReuseBridge()
    result = bridge.run(
        state_props=[0b0001, 0b1000, 0b0100],
        adjacency=[
            1, 0xFFFFFFFF,
            2, 0xFFFFFFFF,
            0xFFFFFFFF, 0xFFFFFFFF,
        ],
        num_states=3,
        max_degree=2,
        root_state=0,
        target_mask=0b0100,
        forbidden_mask=0b1000,
    )
    assert result.status == bridge.STATUS_FAIL
    assert result.witness_state == 1


def test_model_check_reuse_returns_unknown_on_capacity_breach() -> None:
    bridge = ModelCheckReuseBridge()
    state_props = [0b0001] * 65
    adjacency = [0xFFFFFFFF] * (65 * 1)
    result = bridge.run(
        state_props=state_props,
        adjacency=adjacency,
        num_states=65,
        max_degree=1,
        root_state=0,
        target_mask=0b0100,
        forbidden_mask=0b1000,
    )
    assert result.status == bridge.STATUS_UNKNOWN
