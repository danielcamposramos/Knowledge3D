"""
Tests for Unified FSM - Full 5-State Cognitive Pipeline
"""

import numpy as np
import pytest
from pathlib import Path


def test_fsm_kernels_load():
    """Test that FSM PTX kernels load successfully."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")

    base = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"
    fsm_path = base / "fused_head_fsm_full.ptx"

    assert fsm_path.exists(), f"FSM kernel not found: {fsm_path}"

    module = cupy.RawModule(path=str(fsm_path))

    # Verify all entry points are accessible
    dispatch_kernel = module.get_function("fused_head_fsm_dispatch")
    attention_kernel = module.get_function("ptxfuse_attention")
    rpn_kernel = module.get_function("rpn_reason_dispatch")

    assert dispatch_kernel is not None
    assert attention_kernel is not None
    assert rpn_kernel is not None


def test_unified_attention_kernel():
    """Test unified attention kernel directly."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")

    base = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"
    module = cupy.RawModule(path=str(base / "fused_head_fsm_full.ptx"))
    attention_kernel = module.get_function("ptxfuse_attention")

    # Create test data
    n_nodes = 10
    emb_dim = 512

    # Unified buffer (simplified - just fused_emb portion)
    unified_buf = cupy.random.randn(n_nodes, 1024, dtype=cupy.float32)  # 4KB per node
    query_emb = cupy.ones(emb_dim, dtype=cupy.float32)
    attention_out = cupy.zeros(n_nodes, dtype=cupy.float32)

    # Launch kernel
    block = 32
    grid = (1,)

    attention_kernel(
        grid,
        (block,),
        (
            unified_buf,
            query_emb,
            np.int32(n_nodes),
            attention_out
        )
    )

    cupy.cuda.runtime.deviceSynchronize()

    # Verify output shape and that scores were computed
    scores = attention_out.get()
    assert scores.shape == (n_nodes,)
    assert not np.all(scores == 0), "Attention scores should be non-zero"


def test_rpn_dispatch_kernel():
    """Test RPN dispatch kernel directly."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")

    base = Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "ptx"
    module = cupy.RawModule(path=str(base / "fused_head_fsm_full.ptx"))
    rpn_kernel = module.get_function("rpn_reason_dispatch")

    # Create RPN stack with add operation
    rpn_stack = cupy.zeros(256, dtype=cupy.uint32)
    rpn_stack_cpu = rpn_stack.get()
    rpn_stack_cpu[0] = 1  # Op code: add
    rpn_stack_cpu.view(np.float32)[1] = 2.0  # arg1
    rpn_stack_cpu.view(np.float32)[2] = 3.0  # arg2
    rpn_stack = cupy.asarray(rpn_stack_cpu)

    # Dummy unified buffer and output
    unified_buf = cupy.zeros((10, 1024), dtype=cupy.float32)
    output_action = cupy.zeros(512, dtype=cupy.float32)

    # Launch kernel
    block = 1
    grid = (1,)

    rpn_kernel(
        grid,
        (block,),
        (
            rpn_stack,
            unified_buf,
            output_action,
            np.int32(10)
        )
    )

    cupy.cuda.runtime.deviceSynchronize()

    # Verify result
    result_stack = rpn_stack.get()
    result = result_stack.view(np.float32)[1]

    assert np.isclose(result, 5.0, atol=1e-5), f"Expected 2+3=5, got {result}"
    assert result_stack[0] == 4, "RPN flag should be set to 4 (output state)"


def test_unified_fsm_context():
    """Test UnifiedFSMContext class."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")

    from knowledge3d.cranium.unified_fsm import UnifiedFSMContext

    fsm = UnifiedFSMContext()

    # Test buffer creation
    n_nodes = 5
    buf = fsm.create_unified_buffer(n_nodes)
    assert buf.shape[0] == n_nodes

    # Test attention kernel
    query_emb = np.random.randn(512).astype(np.float32)
    scores = fsm.launch_unified_attention(buf, query_emb)
    assert scores.shape == (n_nodes,)
