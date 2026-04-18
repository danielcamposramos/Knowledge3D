from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    TabletEnvelope,
    TabletIngest,
    TabletSessionFrame,
    TabletSessionTape,
)


QUESTION_ROUTE_GALAXIES: tuple[str, ...] = (
    "Reality",
    "Grammar",
    "Word",
    "Character",
    "Language",
)


def build_question_route(
    *,
    specialist: str = "auto",
    domain_hint: str | None = "general",
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    return {
        "specialist": str(specialist or "auto"),
        "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
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
        route_policy=envelope.route_policy,
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


def build_question_session_tape(
    *,
    session_id: str,
    suite_name: str,
    rows: Sequence[Mapping[str, Any]],
    use_enriched: bool = True,
    surface_kind: str = "QUESTION",
) -> TabletSessionTape:
    frames: list[TabletSessionFrame] = []
    for index, row in enumerate(rows):
        normalized_surface_kind = "QUESTION"
        envelope = TabletIngest.question_task(
            task_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
            question=str(row.get("question_text") or row.get("question") or ""),
            options=list(row.get("options") or []),
            domain=str(row.get("subject") or row.get("domain") or "general"),
            expected_answer=str(
                row.get("correct_answer")
                or row.get("expected_answer")
                or ""
            ).strip()
            or None,
            surface_kind=normalized_surface_kind,
        )
        frames.append(
            TabletSessionFrame(
                frame_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
                envelope=envelope,
                expected=row.get("correct_answer") if "correct_answer" in row else row.get("expected_answer"),
                source_meta=dict(row),
            )
        )
    return TabletSessionTape(
        session_id=str(session_id),
        suite_name=str(suite_name),
        surface_kind=str(surface_kind or "QUESTION"),
        frames=tuple(frames),
        use_enriched=bool(use_enriched),
    )
