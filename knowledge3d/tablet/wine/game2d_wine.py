from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import TabletEnvelope, TabletIngest


GAME_2D_ROUTE_GALAXIES: tuple[str, ...] = ()


def build_game2d_route(
    *,
    specialist: str = "visual",
    domain_hint: str | None = "game_2d",
    galaxies: Sequence[str] | None = None,
) -> dict[str, Any]:
    route = {
        "specialist": str(specialist or "visual"),
        "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
    }
    galaxy_names = [
        str(name)
        for name in (galaxies or GAME_2D_ROUTE_GALAXIES)
        if str(name).strip()
    ]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_game2d_task(
    *,
    task_id: str,
    query: str,
    input_grid: Any | None = None,
    goal_grid: Any | None = None,
    training_examples: Sequence[dict[str, Any]] | None = None,
    available_actions: Sequence[int] | None = None,
    action_options: Sequence[str] | None = None,
    expected_output: Any | None = None,
    expected_game_action: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = TabletIngest.game2d_task(
        task_id=task_id,
        query=query,
        input_grid=input_grid,
        goal_grid=goal_grid,
        training_examples=training_examples,
        available_actions=available_actions,
        action_options=action_options,
        expected_output=expected_output,
        expected_game_action=expected_game_action,
        metadata=metadata,
    )
    return dict(envelope.task), build_game2d_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
    )


def arc2_game_envelope(
    *,
    task_id: str,
    training_examples: Sequence[dict[str, Any]],
    input_grid: Any,
    expected_output: Any | None = None,
) -> TabletEnvelope:
    return TabletIngest.game2d_task(
        task_id=task_id,
        query="2d game transformation",
        input_grid=input_grid,
        goal_grid=expected_output,
        training_examples=training_examples,
        expected_output=expected_output,
    )


def arc3_game_envelope(
    *,
    task_id: str,
    frame: Any,
    goal_frame: Any | None = None,
    available_actions: Sequence[int] | None = None,
    action_options: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    return TabletIngest.game2d_task(
        task_id=task_id,
        query="2d game state",
        input_grid=frame,
        goal_grid=goal_frame,
        available_actions=available_actions,
        action_options=action_options,
        metadata=metadata,
    )
