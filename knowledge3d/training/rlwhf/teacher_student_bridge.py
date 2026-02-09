"""Teacher-student RLWHF bridge with ternary + multi-level pooling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ternary_bucket(value: float, *, low: float = 0.33, high: float = 0.66) -> int:
    """
    Map a continuous value to balanced ternary {-1, 0, +1}.

    This keeps RLWHF feedback non-binary while still compact and stable.
    """
    x = _clamp01(value)
    if x < low:
        return -1
    if x > high:
        return 1
    return 0


def ternary_to_index(value: int) -> int:
    """Convert {-1,0,+1} to base-3 index {0,1,2}."""
    if value <= -1:
        return 0
    if value >= 1:
        return 2
    return 1


def build_pool_id(
    *,
    correctness_t: int,
    honesty_t: int,
    transfer_t: int,
    novelty_t: int,
) -> str:
    """
    Build a multi-level ternary pool identifier.

    Four ternary axes produce 3^4 = 81 pools, preserving non-binary signal
    while staying compact for Galaxy metadata.
    """
    i0 = ternary_to_index(correctness_t)
    i1 = ternary_to_index(honesty_t)
    i2 = ternary_to_index(transfer_t)
    i3 = ternary_to_index(novelty_t)
    base3 = f"{i0}{i1}{i2}{i3}"
    ordinal = i0 * 27 + i1 * 9 + i2 * 3 + i3
    return f"ternary_pool_{base3}_{ordinal:02d}"


def pool_hamming_drift(previous_pool_id: str | None, current_pool_id: str | None) -> float:
    """
    Compute normalized Hamming drift between two ternary pool IDs.

    Returns value in [0,1], where:
    - 0.0: no drift (same pool)
    - 1.0: full drift (all ternary axes changed)
    """
    if not previous_pool_id or not current_pool_id:
        return 0.0

    def _base3(pool_id: str) -> str:
        marker = "ternary_pool_"
        if marker not in pool_id:
            return ""
        tail = pool_id.split(marker, 1)[1]
        return tail.split("_", 1)[0]

    prev = _base3(str(previous_pool_id))
    curr = _base3(str(current_pool_id))
    if not prev or not curr or len(prev) != len(curr):
        return 0.0
    diffs = sum(1 for a, b in zip(prev, curr) if a != b)
    return diffs / len(prev)


@dataclass(frozen=True)
class TeacherStudentFeedback:
    """Compact RLWHF feedback record for one curriculum iteration."""

    stage: str
    iteration: int
    train_accuracy: float
    transfer_accuracy: float
    oracle_at_all: float
    generated_pattern_total: int
    correctness_t: int
    honesty_t: int
    transfer_t: int
    novelty_t: int
    pool_id: str
    teacher_rating: int
    teacher_label: str
    quality_score: float
    timestamp: str

    def to_event_data(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "iteration": self.iteration,
            "train_accuracy": self.train_accuracy,
            "transfer_accuracy": self.transfer_accuracy,
            "oracle_at_all": self.oracle_at_all,
            "generated_pattern_total": self.generated_pattern_total,
            "correctness_t": self.correctness_t,
            "honesty_t": self.honesty_t,
            "transfer_t": self.transfer_t,
            "novelty_t": self.novelty_t,
            "pool_id": self.pool_id,
            "teacher_rating": self.teacher_rating,
            "teacher_label": self.teacher_label,
            "quality_score": self.quality_score,
            "specialist": "grammar",
            "galaxy": "Grammar",
            "query": f"rlwhf curriculum iteration {self.iteration} stage {self.stage}",
            "confidence": self.quality_score,
            "verification": "rlwhf_teacher_student_bridge",
        }

    def to_galaxy_entry(self) -> dict[str, Any]:
        key = self.timestamp.replace("-", "").replace(":", "").replace(".", "")
        return {
            "id": f"rlwhf_feedback_{self.stage.lower()}_{self.iteration:03d}_{key}",
            "name": f"RLWHF Feedback Stage {self.stage} Iter {self.iteration}",
            "domain": "grammar",
            "category": "rlwhf_feedback",
            "rpn_program": "CORR HON TRANS NOV POOL_ENCODE",
            "metadata": self.to_event_data(),
        }


class RLWHFTeacherStudentBridge:
    """
    Teacher/student mapper for curriculum-level RLWHF signals.

    The bridge is intentionally lightweight and deterministic:
    - Teacher signal: mapped to 5-level rating {-2..+2}
    - Student signal: transfer + generation health
    - Core memory: balanced ternary axes and 81-pool ID
    """

    _LABELS = {
        -2: "dishonest_or_failed",
        -1: "weak_or_incomplete",
        0: "uncertain",
        1: "good",
        2: "excellent",
    }

    def evaluate_iteration(
        self,
        *,
        stage: str,
        iteration: int,
        train_accuracy: float,
        transfer_accuracy: float,
        oracle_at_all: float,
        generated_pattern_total: int,
        expected_generation_floor: int = 50,
    ) -> TeacherStudentFeedback:
        """
        Produce teacher/student aligned feedback for one iteration.

        Honesty here measures calibration: confidence (train score) should match
        transfer/generation reality.
        """
        train = _clamp01(train_accuracy)
        transfer = _clamp01(transfer_accuracy)
        oracle = _clamp01(oracle_at_all)
        novelty = _clamp01(
            float(generated_pattern_total) / max(1.0, float(expected_generation_floor))
        )

        # Honesty as calibration between internal success and external transfer.
        calibration_gap = abs(train - max(transfer, oracle))
        honesty = _clamp01(1.0 - calibration_gap)

        correctness_t = ternary_bucket(train)
        honesty_t = ternary_bucket(honesty)
        transfer_t = ternary_bucket(max(transfer, oracle))
        novelty_t = ternary_bucket(novelty)
        pool_id = build_pool_id(
            correctness_t=correctness_t,
            honesty_t=honesty_t,
            transfer_t=transfer_t,
            novelty_t=novelty_t,
        )

        quality_score = _clamp01((0.40 * train) + (0.25 * honesty) + (0.25 * transfer) + (0.10 * novelty))
        teacher_rating = int(round((quality_score - 0.5) * 4.0))
        teacher_rating = max(-2, min(2, teacher_rating))

        return TeacherStudentFeedback(
            stage=str(stage),
            iteration=int(iteration),
            train_accuracy=train,
            transfer_accuracy=transfer,
            oracle_at_all=oracle,
            generated_pattern_total=int(generated_pattern_total),
            correctness_t=correctness_t,
            honesty_t=honesty_t,
            transfer_t=transfer_t,
            novelty_t=novelty_t,
            pool_id=pool_id,
            teacher_rating=teacher_rating,
            teacher_label=self._LABELS.get(teacher_rating, "uncertain"),
            quality_score=quality_score,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def evaluate_iteration_contrastive(
        self,
        *,
        forward_feedback: TeacherStudentFeedback,
    ) -> dict[str, Any]:
        """
        Build contrastive signal from a forward feedback record.

        This encodes learning from negative examples:
        - Strong failures increase anti-pattern pressure
        - Uncertainty keeps exploration active
        """
        transfer = _clamp01(forward_feedback.transfer_accuracy)
        oracle = _clamp01(forward_feedback.oracle_at_all)
        novelty = _clamp01(float(forward_feedback.generated_pattern_total) / 100.0)
        failure_pressure = _clamp01(1.0 - max(transfer, oracle))
        anti_pattern_pressure = _clamp01((0.65 * failure_pressure) + (0.35 * (1.0 - novelty)))
        explore_weight = _clamp01(1.0 - abs(forward_feedback.quality_score - 0.5) * 2.0)

        return {
            "stage": forward_feedback.stage,
            "iteration": forward_feedback.iteration,
            "anti_pattern_pressure": anti_pattern_pressure,
            "failure_pressure": failure_pressure,
            "explore_weight": explore_weight,
            "contrastive_recommendation": (
                "generate_anti_patterns" if anti_pattern_pressure >= 0.5 else "stabilize_current_patterns"
            ),
            "pool_id": forward_feedback.pool_id,
            "teacher_rating": forward_feedback.teacher_rating,
            "teacher_label": forward_feedback.teacher_label,
            "verification": "rlwhf_teacher_student_contrastive",
        }

    def persist_feedback(self, knowledgeverse: Any, feedback: TeacherStudentFeedback) -> None:
        """Persist RLWHF feedback as both event and Grammar memory entry."""
        if knowledgeverse is None:
            return
        try:
            knowledgeverse.log_event(
                event_type="rlwhf_teacher_feedback",
                event_data=feedback.to_event_data(),
            )
        except Exception:
            pass
        try:
            knowledgeverse.galaxy_manager.add_entry("Grammar", feedback.to_galaxy_entry())
        except Exception:
            pass

    def persist_contrastive_feedback(self, knowledgeverse: Any, payload: dict[str, Any]) -> None:
        """Persist contrastive teacher/student signal."""
        if knowledgeverse is None:
            return
        try:
            knowledgeverse.log_event(
                event_type="rlwhf_teacher_feedback_contrastive",
                event_data=dict(payload),
            )
        except Exception:
            pass
