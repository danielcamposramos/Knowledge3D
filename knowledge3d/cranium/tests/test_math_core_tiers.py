import numpy as np

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


def _run(engine: TieredRPNEngine, expr: str) -> float:
    """Helper to run a simple scalar RPN program via TieredRPNEngine."""
    # We mirror ModularRPNEngine.compile_tokens/tokenize logic in a minimal way here:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

    mod = ModularRPNEngine()
    tokens = mod.tokenize_rpn(expr)
    op_codes, scalars, vectors = mod.compile_tokens(tokens)
    return engine.execute_single(
        instance_id=0,
        op_codes=op_codes,
        scalars=scalars,
        vectors=vectors,
    )


def test_tier1_simple_arithmetic():
    """Programs with only Tier‑1 ops should execute via Tier‑1 engine and give correct results."""
    engine = TieredRPNEngine()

    # Pure scalar arithmetic uses + and * only.
    result = _run(engine, "2 3 + 5 *")  # (2+3)*5 = 25
    assert np.isclose(result, 25.0)

    # Ensure tier cache recognizes the same op‑set as Tier‑1.
    # We don't assert tier counts directly (internal), but we at least
    # exercise multiple calls with identical opcode sets.
    result2 = _run(engine, "10 4 - 2 /")  # (10-4)/2 = 3
    assert np.isclose(result2, 3.0)


def test_tier2_vector_dot():
    """Programs that include vector ops should route to Tier‑2 or above and still be correct."""
    engine = TieredRPNEngine()

    # Compute dot product [1,0,0]·[0,1,0] = 0
    result = _run(engine, "[1,0,0] [0,1,0] dot")
    assert np.isclose(result, 0.0)

    # Compute magnitude of cross product → |[1,0,0]×[0,1,0]| = 1
    result2 = _run(engine, "[1,0,0] [0,1,0] cross mag")
    assert np.isclose(result2, 1.0)


def test_tier3_is_exercised_by_existing_suite():
    """
    Sanity check: Tier‑3 is covered by dedicated TRM tests.

    Here we only assert that the bridge can be constructed; detailed behavior
    and matrix semantics are validated in `tests/test_rpn_tier3.py` and
    `tests/test_trm_rpn_gpu.py`. This keeps this test focused on Tier‑1/2
    integration and avoids duplicating complex Tier‑3 programs here.
    """
    engine = TieredRPNEngine()
    assert engine is not None
