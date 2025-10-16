from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine


@pytest.fixture(scope="module")
def tier3_engine() -> AdvancedRPNEngine:
    try:
        return AdvancedRPNEngine()
    except RuntimeError as exc:
        if "invalid device context" in str(exc):
            pytest.skip("CUDA context unavailable for Tier-3 RPN tests")
        raise


class TestTier3RPN:
    def test_matrix_matmul(self, tier3_engine: AdvancedRPNEngine) -> None:
        op_codes = np.array([0x02, 0x02, 0x5A], dtype=np.uint16)  # A, B, MATMUL
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        scalars = np.array([A.shape[0], A.shape[1], B.shape[0], B.shape[1]], dtype=np.float32)
        literals = np.concatenate([A.flatten(), B.flatten()]).astype(np.float32)

        result = tier3_engine.execute_matrix(
            0,
            op_codes,
            output_shape=A.shape,
            scalars=scalars,
            matrices=literals,
        )
        expected = A @ B
        assert np.allclose(result, expected, atol=1e-5)

    def test_matrix_matmul_3x3(self, tier3_engine: AdvancedRPNEngine) -> None:
        op_codes = np.array([0x02, 0x02, 0x5A], dtype=np.uint16)
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float32)
        B = np.array([[9.0, 8.0, 7.0],
                      [6.0, 5.0, 4.0],
                      [3.0, 2.0, 1.0]], dtype=np.float32)
        scalars = np.array([A.shape[0], A.shape[1], B.shape[0], B.shape[1]], dtype=np.float32)
        literals = np.concatenate([A.flatten(), B.flatten()]).astype(np.float32)

        result = tier3_engine.execute_matrix(
            0,
            op_codes,
            output_shape=A.shape,
            scalars=scalars,
            matrices=literals,
        )
        expected = A @ B
        assert np.allclose(result, expected, atol=1e-5)

    def test_inverse_vs_numpy(self, tier3_engine: AdvancedRPNEngine) -> None:
        op_codes = np.array([0x02, 0x5D], dtype=np.uint16)  # Matrix literal, inverse
        matrix = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32)
        scalars = np.array([matrix.shape[0], matrix.shape[1]], dtype=np.float32)
        matrices = matrix.flatten().astype(np.float32)

        result = tier3_engine.execute_matrix(
            1,
            op_codes,
            output_shape=matrix.shape,
            scalars=scalars,
            matrices=matrices,
        )
        expected = np.linalg.inv(matrix)
        assert np.allclose(result, expected, atol=1e-5)

    def test_matrix_trace(self, tier3_engine: AdvancedRPNEngine) -> None:
        op_codes = np.array([0x02, 0x5E], dtype=np.uint16)  # Matrix literal, trace
        matrix = np.array(
            [
                [6.0, 1.0, 1.0],
                [4.0, -2.0, 5.0],
                [2.0, 8.0, 7.0],
            ],
            dtype=np.float32,
        )
        scalars = np.array([matrix.shape[0], matrix.shape[1]], dtype=np.float32)
        matrices = matrix.flatten().astype(np.float32)

        scalar = tier3_engine.execute_scalar(
            2,
            op_codes,
            scalars=scalars,
            matrices=matrices,
        )
        expected = float(np.trace(matrix))
        assert pytest.approx(expected, abs=1e-4) == scalar

    def test_determinant_vs_numpy(self, tier3_engine: AdvancedRPNEngine) -> None:
        op_codes = np.array([0x02, 0x5C], dtype=np.uint16)  # Matrix literal, determinant
        matrix = np.array(
            [
                [6.0, 1.0, 1.0],
                [4.0, -2.0, 5.0],
                [2.0, 8.0, 7.0],
            ],
            dtype=np.float32,
        )
        scalars = np.array([matrix.shape[0], matrix.shape[1]], dtype=np.float32)
        matrices = matrix.flatten().astype(np.float32)

        det = tier3_engine.execute_scalar(
            3,
            op_codes,
            scalars=scalars,
            matrices=matrices,
        )
        expected = float(np.linalg.det(matrix))
        assert pytest.approx(expected, abs=1e-4) == det
