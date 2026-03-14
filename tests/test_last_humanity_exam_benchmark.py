from __future__ import annotations

import pytest

pytest.importorskip("cupy")

from benchmarks.last_humanity_exam import LastHumanityExamBenchmark


def test_open_ended_match_requires_exact_normalized_text() -> None:
    bench = LastHumanityExamBenchmark(dataset_path="/tmp/does_not_exist", max_questions=1)

    assert bench._open_ended_match("Weak Non-Sadism", "Weak Non-Sadism")
    assert not bench._open_ended_match("Weak", "Weak Non-Sadism")
    assert not bench._open_ended_match("2", "\\(-((d - 2k)^2) + d\\)")


def test_open_ended_match_uses_numeric_equivalence_only_for_numeric_answers() -> None:
    bench = LastHumanityExamBenchmark(dataset_path="/tmp/does_not_exist", max_questions=1)

    assert bench._open_ended_match("3.0", "3")
    assert not bench._open_ended_match("1.25663706212e-6 N/A^2", "3")
