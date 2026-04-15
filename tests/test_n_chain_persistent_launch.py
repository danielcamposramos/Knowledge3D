from __future__ import annotations

import os

import pytest

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file


pytestmark = pytest.mark.skipif(
    os.environ.get("K3D_PYTEST_PROBE_CUDA") != "1",
    reason="real CUDA probe disabled",
)


def test_n_chain_persistent_kernel_compiles() -> None:
    ptx = compile_cuda_file(
        "knowledge3d/cranium/cuda/k3d_swarm_persistent.cu",
        arch="sm_86",
        use_fast_math=False,
    )
    assert ".entry k3d_swarm_sovereign" in ptx
    assert "k3d_swarm_sovereign" in ptx


def test_n_chain_bridge_launch_tick_shutdown_cuda() -> None:
    from knowledge3d.cranium.bridges.n_chain_swarm_bridge import NChainSwarmBridge

    bridge = NChainSwarmBridge()
    try:
        bridge.launch()
        assert bridge.n_active == bridge.N_DEFAULT
        for _ in range(16):
            result = bridge.tick(
                {"n_cand_frustum": 17, "h_belief_q10": 384, "t_remaining_us": 20_000},
                timeout_s=5.0,
            )
            assert bridge.N_FLOOR <= result["n_active"] <= 17
            assert result["halting_flag"] == bridge.FLAG_COMPLETE
            assert result["halting_counter"] >= result["n_active"]
    finally:
        bridge.cleanup()
