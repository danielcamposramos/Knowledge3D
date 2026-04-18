from __future__ import annotations

from knowledge3d.tablet.wine.math_wine import build_math_session_tape


def test_math_session_tape_strips_benchmark_labels_from_wire_payload() -> None:
    tape = build_math_session_tape(
        session_id="math_wire_contract",
        suite_name="math",
        rows=[
            {
                "id": "amc_1",
                "problem_text": "If 2x + 3 = 11, what is x?",
                "answer": "4",
                "competition": "AMC",
                "source": "math_sender",
                "dataset": "competition_set",
                "suite": "math",
            }
        ],
        use_enriched=True,
    )

    frame = tape.frames[0]
    payload = tape.to_payload()["frames"][0]

    assert "competition" not in frame.envelope.task
    assert "source" not in frame.envelope.task
    assert "dataset" not in frame.envelope.task
    assert "competition" not in frame.source_meta
    assert "source" not in frame.source_meta
    assert "dataset" not in frame.source_meta
    assert "competition" not in payload["source_meta"]
    assert "source" not in payload["source_meta"]
    assert "dataset" not in payload["source_meta"]
    assert payload["source_meta"]["suite"] == "math"
