from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


class TestTieredRPNEngine:
    def test_tier1_dispatch(self) -> None:
        try:
            engine = TieredRPNEngine()
        except RuntimeError as exc:
            if "invalid device context" in str(exc):
                pytest.skip("CUDA context unavailable for tiered RPN tests")
            raise
        op_codes = [0, 0, 0x0A]  # 2 + 3
        scalars = [2.0, 3.0, 0.0]
        vectors = np.zeros((len(op_codes), 3), dtype=np.float32)

        assert engine._determine_tier(op_codes) == 1
        result = engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)
        assert abs(result - 5.0) < 1e-5

    def test_tier2_dispatch_dot(self) -> None:
        try:
            engine = TieredRPNEngine()
        except RuntimeError as exc:
            if "invalid device context" in str(exc):
                pytest.skip("CUDA context unavailable for tiered RPN tests")
            raise
        op_codes = [1, 1, 60]  # push vec, push vec, dot
        scalars = [0.0, 0.0, 0.0]
        vectors = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        assert engine._determine_tier(op_codes) == 2
        result = engine.execute_scalar(op_codes, scalars=scalars, vectors=vectors)
        assert abs(result - 32.0) < 1e-5

    def test_tier3_dispatch_matrix(self) -> None:
        try:
            engine = TieredRPNEngine()
        except RuntimeError as exc:
            if "invalid device context" in str(exc):
                pytest.skip("CUDA context unavailable for tiered RPN tests")
            raise
        op_codes = [0x02, 0x02, 0x5A]
        scalars = [2.0, 2.0, 2.0, 2.0]
        matrices = np.array(
            [
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            dtype=np.float32,
        )

        assert engine._determine_tier(op_codes) == 3
        result = engine.execute_matrix(
            op_codes,
            matrix_shape=(2, 2),
            scalars=scalars,
            matrices=matrices,
        )
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)
