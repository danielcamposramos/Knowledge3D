from __future__ import annotations

from benchmarks.arc_submission_formatter import format_arc_submission, validate_arc_submission


def test_format_arc_submission_builds_multi_sample_two_attempt_shape() -> None:
    submission = format_arc_submission(
        [
            {
                "task_id": "abc123",
                "sample_index": 1,
                "predicted": [[1, 2], [3, 4]],
                "secondary_prediction": [[4, 3], [2, 1]],
                "input_grid": [[9]],
            },
            {
                "task_id": "abc123",
                "sample_index": 0,
                "predicted": [[0]],
                "input_grid": [[7]],
            },
            {
                "task_id": "xyz789",
                "sample_index": 0,
                "predicted": [[5]],
                "secondary_prediction": [[6]],
                "input_grid": [[8]],
            },
        ],
    )
    assert submission == {
        "abc123": [
            {"attempt_1": [[0]], "attempt_2": [[7]]},
            {"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[4, 3], [2, 1]]},
        ],
        "xyz789": [{"attempt_1": [[5]], "attempt_2": [[6]]}],
    }


def test_validate_arc_submission_rejects_non_grid_payloads() -> None:
    errors = validate_arc_submission({"bad": [{"attempt_1": "not-a-grid"}]})  # type: ignore[arg-type]
    assert errors


def test_format_arc_submission_uses_identity_grid_when_primary_missing() -> None:
    submission = format_arc_submission(
        [
            {
                "task_id": "identity_case",
                "sample_index": 0,
                "predicted": None,
                "input_grid": [[1, 0], [0, 1]],
            }
        ]
    )
    assert submission == {
        "identity_case": [{"attempt_1": [[1, 0], [0, 1]], "attempt_2": [[1, 0], [0, 1]]}]
    }
