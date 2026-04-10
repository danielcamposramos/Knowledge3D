from __future__ import annotations

import numpy as np

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class _CaptureGate:
    def __init__(self) -> None:
        self.worker_weights: list[float] = []
        self.minimum_threshold: float = 0.0
        self.gap_threshold: float = 0.0
        self.agreement_threshold: float = 0.0

    def analyze_scores(
        self,
        scores,
        candidate_hashes,
        *,
        worker_weights=None,
        minimum_threshold=0.0,
        gap_threshold=0.0,
        agreement_threshold=0.0,
    ):
        self.worker_weights = [float(value) for value in list(worker_weights or [])]
        self.minimum_threshold = float(minimum_threshold)
        self.gap_threshold = float(gap_threshold)
        self.agreement_threshold = float(agreement_threshold)
        return (
            np.asarray([1, 1, 1, 1], dtype=np.int32),
            np.asarray([0.92, 0.20, 2.0, 0.0], dtype=np.float32),
        )


def _assert_halting_weights(
    tmp_path,
    *,
    task_type: str,
    worker_slots: list[int],
    expected: list[float],
) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / f"kv_halting_{task_type.lower()}")
    gate = _CaptureGate()
    kv._halting_gate = gate
    steps: list[str] = []

    converged = kv._halting_gate_converged(
        task_type=task_type,
        task=None,
        path_scores=[0.91, 0.61, 0.22],
        candidate_ids=["cand_a", "cand_b", "cand_c"],
        worker_slots=worker_slots,
        selection_steps=steps,
    )

    assert converged is True
    assert gate.worker_weights == expected
    assert any(
        "weights=" + ",".join(f"{value:.2f}" for value in expected) in step
        for step in steps
    )


def test_arc_halting_gate_uses_game2d_weight_vector(tmp_path) -> None:
    _assert_halting_weights(
        tmp_path,
        task_type="ARC_TASK",
        worker_slots=[0, 3, 4],
        expected=[1.0, 2.0, 2.0],
    )


def test_math_halting_gate_uses_math_weight_vector(tmp_path) -> None:
    _assert_halting_weights(
        tmp_path,
        task_type="MATH_TASK",
        worker_slots=[0, 2, 6],
        expected=[2.0, 1.5, 2.0],
    )


def test_question_halting_gate_uses_question_weight_vector(tmp_path) -> None:
    _assert_halting_weights(
        tmp_path,
        task_type="MMLU_TASK",
        worker_slots=[0, 1, 8],
        expected=[1.5, 2.0, 1.5],
    )
