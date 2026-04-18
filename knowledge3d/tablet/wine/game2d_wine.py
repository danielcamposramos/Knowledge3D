from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    TabletEnvelope,
    TabletIngest,
    TabletSessionFrame,
    TabletSessionTape,
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
        query="Transform this grid using the examples.",
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
    query: str = "Continue this game frame using the available actions.",
    query_embedding: Sequence[float] | None = None,
    goal_embedding: Sequence[float] | None = None,
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
    if query_embedding is not None:
        task_context["query_embedding"] = [float(value) for value in list(query_embedding)]
    if goal_embedding is not None:
        task_context["goal_embedding"] = [float(value) for value in list(goal_embedding)]
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


def build_game2d_session_tape(
    *,
    session_id: str,
    suite_name: str,
    rows: Sequence[Mapping[str, Any]],
    use_enriched: bool = True,
) -> TabletSessionTape:
    frames: list[TabletSessionFrame] = []
    for index, row in enumerate(rows):
        envelope = arc2_game_envelope(
            task_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
            training_examples=list(row.get("train") or row.get("training_examples") or []),
            input_grid=(list(row.get("test") or [{}])[0].get("input") if isinstance(row.get("test"), list) and row.get("test") else row.get("input_grid")),
            expected_output=(list(row.get("test") or [{}])[0].get("output") if isinstance(row.get("test"), list) and row.get("test") else row.get("expected_output")),
        )
        expected = None
        if isinstance(row.get("test"), list) and row.get("test"):
            expected = row["test"][0].get("output")
        else:
            expected = row.get("expected_output")
        frames.append(
            TabletSessionFrame(
                frame_id=str(row.get("id") or row.get("task_id") or f"{suite_name}_{index}"),
                envelope=envelope,
                expected=expected,
                source_meta=dict(row),
            )
        )
    return TabletSessionTape(
        session_id=str(session_id),
        suite_name=str(suite_name),
        surface_kind="GAME_2D",
        frames=tuple(frames),
        use_enriched=bool(use_enriched),
    )
