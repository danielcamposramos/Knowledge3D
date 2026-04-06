from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import TabletEnvelope, TabletIngest


MATH_ROUTE_GALAXIES: tuple[str, ...] = ()


def build_math_route(
    *,
    specialist: str = "math",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
) -> dict[str, Any]:
    route: dict[str, Any] = {"specialist": str(specialist or "math")}
    if domain_hint is not None and str(domain_hint).strip():
        route["domain_hint"] = str(domain_hint).strip()
    galaxy_names = [str(name) for name in (galaxies or MATH_ROUTE_GALAXIES) if str(name).strip()]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_math_task(
    *,
    task_id: str,
    question: str,
    competition: str | None = None,
    expected_answer: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = TabletIngest.math_task(
        task_id=task_id,
        question=question,
        expected_answer=expected_answer,
        competition=competition,
        metadata=metadata,
    )
    return dict(envelope.task), build_math_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
    )


def math_dataset_envelope(
    *,
    task_id: str,
    question: str,
    competition: str | None = None,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return TabletIngest.math_task(
        task_id=task_id,
        question=question,
        expected_answer=expected_answer,
        competition=competition,
    )


def gsm8k_math_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return math_dataset_envelope(
        task_id=task_id,
        question=question,
        competition="GSM8K",
        expected_answer=expected_answer,
    )


def imo_math_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return math_dataset_envelope(
        task_id=task_id,
        question=question,
        competition="IMO",
        expected_answer=expected_answer,
    )


def amc_aime_math_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
    competition: str | None = None,
) -> TabletEnvelope:
    return math_dataset_envelope(
        task_id=task_id,
        question=question,
        competition=competition or "AMC-AIME",
        expected_answer=expected_answer,
    )


def omni_math_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return math_dataset_envelope(
        task_id=task_id,
        question=question,
        competition="Omni-MATH",
        expected_answer=expected_answer,
    )
