from __future__ import annotations

from knowledge3d.tablet.wine.question_wine import build_question_session_tape, lhe_question_envelope


def test_lhe_translator_uses_question_surface() -> None:
    envelope = lhe_question_envelope(task_id="lhe_1", question="If all A are B and all B are C, what follows?")
    assert envelope.surface_kind == "QUESTION"
    assert str(envelope.task.get("surface_kind") or "") == "QUESTION"
    assert "type" not in envelope.task


def test_question_session_tape_emits_natural_question_surface() -> None:
    tape = build_question_session_tape(
        session_id="sess_1",
        suite_name="lhe",
        rows=[{"id": "q1", "question": "Explain why the evidence supports X", "options": ["A", "B"]}],
        use_enriched=True,
        surface_kind="QUESTION",
    )
    frame = tape.frames[0]
    assert frame.envelope.surface_kind == "QUESTION"
    assert "type" not in frame.envelope.task
