from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import TabletEnvelope, TabletIngest


QUESTION_ROUTE_GALAXIES: tuple[str, ...] = ()


def build_question_route(
    *,
    specialist: str = "auto",
    domain_hint: str | None = "general",
    galaxies: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "specialist": str(specialist or "auto"),
        "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
        "galaxy_names": [
            str(name)
            for name in (galaxies or QUESTION_ROUTE_GALAXIES)
            if str(name).strip()
        ],
    }


def build_question_task(
    *,
    task_id: str,
    question: str,
    options: Sequence[str] | None = None,
    domain: str = "general",
    expected_answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = TabletIngest.question_task(
        task_id=task_id,
        question=question,
        options=options,
        domain=domain,
        expected_answer=expected_answer,
        metadata=metadata,
    )
    return dict(envelope.task), build_question_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
    )


def mmlu_question_envelope(
    *,
    task_id: str,
    question: str,
    options: Sequence[str],
    subject: str,
    expected_answer: str | None = None,
) -> TabletEnvelope:
    return TabletIngest.question_task(
        task_id=task_id,
        question=question,
        options=options,
        domain=subject,
        expected_answer=expected_answer,
    )


def lhe_question_envelope(
    *,
    task_id: str,
    question: str,
    options: Sequence[str] | None = None,
    domain: str = "general",
    expected_answer: str | None = None,
) -> TabletEnvelope:
    return TabletIngest.question_task(
        task_id=task_id,
        question=question,
        options=options,
        domain=domain,
        expected_answer=expected_answer,
    )
