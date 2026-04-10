from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    TabletEnvelope,
    TabletIngest,
)


GAME_2D_ROUTE_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Reality",
    "Grammar",
    "Math",
    "Tool",
    "Number",
    "Word",
    "Character",
    "Audio",
    "3DObjects",
    "Language",
)


def build_game2d_route(
    *,
    specialist: str = "visual",
    domain_hint: str | None = "game_2d",
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    route = {
        "specialist": str(specialist or "visual"),
        "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
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
    result_kind: str | None = None,
    task_context: Mapping[str, Any] | None = None,
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
        result_kind=result_kind,
        task_context=task_context,
        metadata=metadata,
    )
    return dict(envelope.task), build_game2d_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
        route_policy=envelope.route_policy,
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
        result_kind="grid",
    )


def arc3_game_envelope(
    *,
    task_id: str,
    frame: Any,
    goal_frame: Any | None = None,
    available_actions: Sequence[int] | None = None,
    action_options: Sequence[str] | None = None,
    training_examples: Sequence[dict[str, Any]] | None = None,
    query: str = "arc3 interactive game frame",
    step_count: int = 0,
    game_id: str = "",
    levels_completed: int = 0,
    world_model: Mapping[str, Any] | None = None,
    episode_context: Mapping[str, Any] | None = None,
    task_context_extras: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    ctx = dict(episode_context or {})
    task_context = {
        "step_count": int(step_count),
        "game_id": str(game_id or ""),
        "levels_completed": int(levels_completed),
        "world_model": dict(world_model or {}),
        "known_objects": dict(ctx.get("objects") or {}),
    }
    for key, value in dict(task_context_extras or {}).items():
        task_context[str(key)] = value
    return TabletIngest.game2d_task(
        task_id=task_id,
        query=str(query),
        input_grid=frame,
        goal_grid=goal_frame,
        training_examples=training_examples,
        available_actions=available_actions,
        action_options=action_options,
        expected_output=goal_frame,
        result_kind="control",
        task_context=task_context,
        metadata=metadata,
    )
