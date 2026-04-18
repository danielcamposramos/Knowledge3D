from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    TabletEnvelope,
    TabletIngest,
    TabletSessionFrame,
    TabletSessionTape,
)


_MATH_WIRE_META_DROP_KEYS = frozenset({"competition", "dataset", "source"})


MATH_ROUTE_GALAXIES: tuple[str, ...] = (
    "Math",
    "Grammar",
    "Number",
    "Reality",
    "Word",
    "Character",
)


def build_math_route(
    *,
    specialist: str = "math",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    route: dict[str, Any] = {
        "specialist": str(specialist or "math"),
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
    }
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
    expected_answer: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = TabletIngest.math_task(
        task_id=task_id,
        question=question,
        expected_answer=expected_answer,
        metadata=metadata,
    )
    return dict(envelope.task), build_math_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
        route_policy=envelope.route_policy,
    )


def math_dataset_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return TabletIngest.math_task(
        task_id=task_id,
        question=question,
        expected_answer=expected_answer,
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
        expected_answer=expected_answer,
    )


def amc_aime_math_envelope(
    *,
    task_id: str,
    question: str,
    expected_answer: Any | None = None,
) -> TabletEnvelope:
    return math_dataset_envelope(
        task_id=task_id,
        question=question,
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
        expected_answer=expected_answer,
    )


def _wire_safe_source_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(row).items()
        if str(key) not in _MATH_WIRE_META_DROP_KEYS
    }


def build_math_session_tape(
    *,
    session_id: str,
    suite_name: str,
    rows: Sequence[Mapping[str, Any]],
    use_enriched: bool = True,
) -> TabletSessionTape:
    frames: list[TabletSessionFrame] = []
    for index, row in enumerate(rows):
        envelope = math_dataset_envelope(
            task_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
            question=str(row.get("problem_text") or row.get("question") or ""),
            expected_answer=row.get("answer") if "answer" in row else row.get("expected_answer"),
        )
        frames.append(
            TabletSessionFrame(
                frame_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
                envelope=envelope,
                expected=row.get("answer") if "answer" in row else row.get("expected_answer"),
                source_meta=_wire_safe_source_meta(row),
            )
        )
    return TabletSessionTape(
        session_id=str(session_id),
        suite_name=str(suite_name),
        surface_kind="MATH",
        frames=tuple(frames),
        use_enriched=bool(use_enriched),
    )
