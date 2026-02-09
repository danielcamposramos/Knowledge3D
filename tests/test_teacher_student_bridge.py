from __future__ import annotations

from knowledge3d.training.rlwhf.teacher_student_bridge import (
    RLWHFTeacherStudentBridge,
    build_pool_id,
    pool_hamming_drift,
    ternary_bucket,
)


def test_ternary_bucket_balanced_mapping():
    assert ternary_bucket(0.10) == -1
    assert ternary_bucket(0.50) == 0
    assert ternary_bucket(0.90) == 1


def test_multi_axis_pool_id_stable():
    pool = build_pool_id(
        correctness_t=1,
        honesty_t=0,
        transfer_t=-1,
        novelty_t=1,
    )
    assert pool.startswith("ternary_pool_")
    assert pool.endswith("_65")


def test_bridge_feedback_contains_teacher_student_and_pool_data():
    bridge = RLWHFTeacherStudentBridge()
    feedback = bridge.evaluate_iteration(
        stage="B",
        iteration=4,
        train_accuracy=0.86,
        transfer_accuracy=0.41,
        oracle_at_all=0.52,
        generated_pattern_total=42,
        expected_generation_floor=50,
    )

    assert feedback.stage == "B"
    assert feedback.iteration == 4
    assert feedback.pool_id.startswith("ternary_pool_")
    assert feedback.teacher_rating in {-2, -1, 0, 1, 2}
    assert 0.0 <= feedback.quality_score <= 1.0

    event = feedback.to_event_data()
    assert event["verification"] == "rlwhf_teacher_student_bridge"
    assert event["specialist"] == "grammar"
    assert event["galaxy"] == "Grammar"

    entry = feedback.to_galaxy_entry()
    assert entry["domain"] == "grammar"
    assert entry["category"] == "rlwhf_feedback"
    assert entry["metadata"]["pool_id"] == feedback.pool_id


def test_pool_hamming_drift_and_contrastive_feedback():
    assert pool_hamming_drift("ternary_pool_2000_54", "ternary_pool_2000_54") == 0.0
    assert pool_hamming_drift("ternary_pool_2000_54", "ternary_pool_2100_63") > 0.0

    bridge = RLWHFTeacherStudentBridge()
    fb = bridge.evaluate_iteration(
        stage="C",
        iteration=7,
        train_accuracy=0.9,
        transfer_accuracy=0.2,
        oracle_at_all=0.0,
        generated_pattern_total=0,
        expected_generation_floor=100,
    )
    contrastive = bridge.evaluate_iteration_contrastive(forward_feedback=fb)
    assert contrastive["verification"] == "rlwhf_teacher_student_contrastive"
    assert 0.0 <= contrastive["anti_pattern_pressure"] <= 1.0
    assert contrastive["contrastive_recommendation"] in {
        "generate_anti_patterns",
        "stabilize_current_patterns",
    }
