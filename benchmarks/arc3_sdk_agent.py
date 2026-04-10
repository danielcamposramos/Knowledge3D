"""Truthful ARC-AGI-3 SDK adapter for the sovereign ARC R0 lane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import typing
from typing import Any

from benchmarks.arc_agi_3 import ACTION_NAMES, K3DARC3Agent, _focus_centroid, _lives_remaining, _movement_budget_snapshot
from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


DEFAULT_API_URL = "https://three.arcprize.org"
ACTIVE_STATES = {"IN_PROGRESS", "NOT_FINISHED"}


def _scorecard_url(api_url: str, card_id: str) -> str:
    base = str(api_url).rstrip("/")
    return f"{base}/scorecards/{card_id}"


def _patch_typing_self() -> None:
    if getattr(typing, "Self", None) is None:
        try:
            from typing_extensions import Self as _Self
        except Exception:
            typing.Self = typing.Any  # type: ignore[attr-defined]
        else:
            typing.Self = _Self  # type: ignore[attr-defined]


def _load_arc_sdk() -> tuple[Any | None, Any | None, str | None]:
    try:
        _patch_typing_self()
        import arc_agi  # type: ignore
    except Exception as exc:
        return None, None, f"arc_agi import failed: {exc}"

    try:
        from arcengine import GameAction  # type: ignore
    except Exception as exc:
        GameAction = None
        action_error = f"arcengine import failed: {exc}"
    else:
        action_error = None

    arcade_cls = getattr(arc_agi, "Arcade", None)
    make_fn = getattr(arc_agi, "make", None)
    if arcade_cls is None and not callable(make_fn):
        return arc_agi, GameAction, "arc_agi installed without Arcade/make runtime"
    if GameAction is None:
        return arc_agi, None, action_error or "arcengine missing"
    return arc_agi, GameAction, None


def sdk_status() -> dict[str, Any]:
    arc_agi, game_action_type, error = _load_arc_sdk()
    return {
        "package": "arc-agi",
        "package_available": bool(arc_agi is not None),
        "game_surface_available": bool(arc_agi is not None and (hasattr(arc_agi, "Arcade") or hasattr(arc_agi, "make"))),
        "game_action_available": bool(game_action_type is not None),
        "import_error": str(error or ""),
    }


def _normalize_grid(frame: Any) -> list[list[int]]:
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool))

    if isinstance(frame, list) and frame and all(isinstance(row, list) for row in frame):
        if all(all(_is_scalar(cell) for cell in row) for row in frame):
            return [[int(cell) for cell in row] for row in frame]
        for item in frame:
            resolved = _normalize_grid(item)
            if resolved and resolved != [[]]:
                return resolved
    if isinstance(frame, dict):
        for key in ("grid", "frame", "observation", "board", "image"):
            if key in frame:
                resolved = _normalize_grid(frame[key])
                if resolved and resolved != [[]]:
                    return resolved
        for value in frame.values():
            resolved = _normalize_grid(value)
            if resolved and resolved != [[]]:
                return resolved
    return [[]]


def _goal_from_payload(payload: Any) -> list[list[int]]:
    if isinstance(payload, dict):
        goal = _normalize_grid(payload.get("goal"))
        if goal and goal != [[]]:
            return goal
        task_data = payload.get("task_data") or {}
        for split_name in ("train", "test"):
            for example in list(task_data.get(split_name) or []):
                if not isinstance(example, dict):
                    continue
                output = _normalize_grid(example.get("output"))
                if output and output != [[]]:
                    return output
    return [[]]


def _resolve_api_key() -> str:
    api_key = os.environ.get("ARC_API_KEY", "").strip()
    if api_key:
        return api_key
    secrets_path = Path("/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt")
    if secrets_path.exists():
        return secrets_path.read_text(encoding="utf-8").strip()
    return ""


def _to_sdk_action(game_action_type: Any, action_name: str, payload: dict[str, Any]) -> Any:
    if game_action_type is None:
        return SimpleNamespace(
            name=str(action_name),
            value=str(action_name),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
        )
    if hasattr(game_action_type, action_name):
        enum_value = getattr(game_action_type, action_name)
        if action_name == "ACTION6" and {"x", "y"} <= set(payload):
            try:
                return enum_value(int(payload["x"]), int(payload["y"]))
            except Exception:
                pass
        return enum_value
    return SimpleNamespace(
        name=str(action_name),
        value=str(action_name),
        x=int(payload.get("x", 0)),
        y=int(payload.get("y", 0)),
    )


def _action_name(action: Any) -> str:
    return str(getattr(action, "name", getattr(action, "value", action)))


def _normalize_available_actions(values: list[Any] | None) -> list[int]:
    normalized: list[int] = []
    for value in list(values or []):
        if isinstance(value, int):
            normalized.append(int(value))
            continue
        name = _action_name(value).strip().upper()
        if name in ACTION_NAMES:
            normalized.append(ACTION_NAMES.index(name))
    return normalized


class _RemoteArcCompatEnv:
    """HTTP compatibility wrapper exposing reset/step/action_space."""

    def __init__(self, game_id: str, *, api_url: str = DEFAULT_API_URL, rendering: bool = False) -> None:
        self.game_id = str(game_id)
        self.requested_game_id = str(game_id)
        self.api_url = str(api_url).rstrip("/")
        self.rendering = bool(rendering)
        self.card_id = ""
        self.guid = ""
        self.state = "IN_PROGRESS"
        self.levels_completed = 0
        self.task_data: dict[str, Any] = {}
        self._available_actions: list[Any] = []
        self._api_key = _resolve_api_key()
        self._session = None
        self.game_catalog: list[dict[str, Any]] = []
        self.game_tags: list[str] = []
        self.game_title = ""
        self.game_id = self._resolve_full_game_id()

    @property
    def action_space(self) -> list[Any]:
        return list(self._available_actions)

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        import requests

        self._session = requests.Session()
        headers = {
            "Accept": "application/json",
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        self._session.headers.update(headers)
        return self._session

    def _resolve_full_game_id(self) -> str:
        session = self._ensure_session()
        requested = str(self.requested_game_id).strip()
        try:
            response = session.get(f"{self.api_url}/api/games", timeout=10)
            response.raise_for_status()
            games = list(response.json() or [])
        except Exception as exc:
            print(f"[ARC3] Failed to resolve game_id {requested!r}: {exc}")
            return requested

        self.game_catalog = [dict(row) for row in games if isinstance(row, dict)]
        exact_match: dict[str, Any] | None = None
        prefix_match: dict[str, Any] | None = None
        requested_lower = requested.lower()
        for row in self.game_catalog:
            game_id = str(row.get("game_id", "")).strip()
            if not game_id:
                continue
            if game_id == requested:
                exact_match = row
                break
            if game_id.lower().startswith(requested_lower):
                prefix_match = row
        chosen = exact_match or prefix_match
        if chosen is None:
            print(f"[ARC3] No full game_id match for {requested!r}; using requested value")
            return requested

        resolved = str(chosen.get("game_id", requested)).strip() or requested
        self.game_tags = [str(tag).strip() for tag in list(chosen.get("tags") or []) if str(tag).strip()]
        self.game_title = str(chosen.get("title", "")).strip()
        if resolved != requested:
            print(f"[ARC3] Resolved game_id: {requested!r} -> {resolved!r}")
        return resolved

    def _resolve_actions(self, payload: dict[str, Any]) -> list[Any]:
        values = list(payload.get("available_actions") or [])
        if not values:
            return [SimpleNamespace(name=name, value=name) for name in ACTION_NAMES]
        resolved: list[Any] = []
        for value in values:
            if isinstance(value, str) and value.strip().upper() in ACTION_NAMES:
                name = value.strip().upper()
            elif isinstance(value, int) and 1 <= int(value) <= len(ACTION_NAMES):
                name = ACTION_NAMES[int(value) - 1]
            elif isinstance(value, int) and 0 <= int(value) < len(ACTION_NAMES):
                name = ACTION_NAMES[int(value)]
            else:
                continue
            resolved.append(SimpleNamespace(name=name, value=name))
        return resolved or [SimpleNamespace(name=name, value=name) for name in ACTION_NAMES]

    def reset(self):
        session = self._ensure_session()
        scorecard = session.post(
            f"{self.api_url}/api/scorecard/open",
            json={"tags": ["k3d-sovereign-r0"]},
            timeout=30,
        ).json()
        self.card_id = str(scorecard.get("card_id", ""))
        print(f"[ARC3-DIAG] scorecard response keys: {list(scorecard.keys())}")
        payload = session.post(
            f"{self.api_url}/api/cmd/RESET",
            json={"card_id": self.card_id, "game_id": self.game_id},
            timeout=30,
        ).json()
        print(f"[ARC3-DIAG] RESET response keys: {list(payload.keys())}")
        print(f"[ARC3-DIAG] available_actions raw: {payload.get('available_actions')!r}")
        print(f"[ARC3-DIAG] state: {payload.get('state')!r}")
        print(f"[ARC3-DIAG] frame type: {type(payload.get('frame')).__name__}")
        if isinstance(payload.get("frame"), list):
            frame = payload["frame"]
            print(f"[ARC3-DIAG] frame shape: {len(frame)} rows × {len(frame[0]) if frame else 0} cols")
        self.guid = str(payload.get("guid", ""))
        self.state = str(payload.get("state", "IN_PROGRESS"))
        self.levels_completed = int(payload.get("levels_completed", 0) or 0)
        self.task_data = dict(payload.get("task_data") or {})
        self._available_actions = self._resolve_actions(payload)
        return {
            "grid": _normalize_grid(payload.get("frame")),
            "goal": _goal_from_payload(payload),
            "available_actions": list(self._available_actions),
            "state": self.state,
            "levels_completed": self.levels_completed,
            "task_data": dict(self.task_data),
        }

    def step(self, action: Any, data: dict[str, Any] | None = None):
        session = self._ensure_session()
        action_name = _action_name(action).strip().upper()
        payload: dict[str, Any] = {"game_id": self.game_id}
        if self.guid:
            payload["guid"] = self.guid
        if data:
            payload.update({key: value for key, value in dict(data).items() if value is not None})
        print(f"[ARC3] step: POST /api/cmd/{action_name} payload_keys={list(payload.keys())}")
        http_response = session.post(
            f"{self.api_url}/api/cmd/{action_name}",
            json=payload,
            timeout=30,
        )
        if http_response.status_code >= 400:
            body = http_response.text[:500]
            print(f"[ARC3] step HTTP {http_response.status_code}: {body[:200]}")
            raise RuntimeError(
                f"remote_api_step_failed action={action_name!r} "
                f"status={http_response.status_code} body={body[:200]!r}"
            )
        try:
            response = dict(http_response.json())
        except Exception as exc:
            body = http_response.text[:500]
            print(f"[ARC3] step JSON parse failed for {action_name!r}: {exc}")
            print(f"[ARC3] step raw body (first 500 chars): {body!r}")
            raise
        action_input = response.get("action_input", {})
        action_id = action_input.get("id", -1) if isinstance(action_input, dict) else -1
        print(f"[ARC3] step response: action_input.id={action_id} state={response.get('state')!r}")
        previous_levels = self.levels_completed
        self.guid = str(response.get("guid", self.guid))
        self.state = str(response.get("state", self.state))
        self.levels_completed = int(response.get("levels_completed", self.levels_completed) or 0)
        if isinstance(response.get("task_data"), dict):
            self.task_data = dict(response.get("task_data") or {})
        self._available_actions = self._resolve_actions(response)
        observation = {
            "grid": _normalize_grid(response.get("frame")),
            "goal": _goal_from_payload(response),
            "available_actions": list(self._available_actions),
            "state": self.state,
            "levels_completed": self.levels_completed,
            "task_data": dict(self.task_data),
        }
        reward = float(self.levels_completed - previous_levels)
        done = self.state not in ACTIVE_STATES
        info = {
            "levels_completed": self.levels_completed,
            "state": self.state,
            "transport": "remote_api_compat",
            "card_id": self.card_id,
            "scorecard_url": _scorecard_url(self.api_url, self.card_id) if self.card_id else "",
        }
        return observation, reward, done, info

    def close(self) -> None:
        if self._session is None or not self.card_id:
            return
        try:
            self._session.post(
                f"{self.api_url}/api/scorecard/close",
                json={"card_id": self.card_id},
                timeout=30,
            )
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass
        self._session = None


class K3DAgent:
    """K3D sovereign ARC-3 agent wrapped around the official SDK when present."""

    def __init__(
        self,
        game_id: str = "ls20",
        *,
        max_steps: int = 10000,
        rendering: bool = False,
        allow_remote_compat: bool = False,
        env: Any | None = None,
        knowledgeverse: Any | None = None,
        frame_encoder: ARC3FrameEncoder | None = None,
    ) -> None:
        self.game_id = str(game_id)
        self.max_steps = int(max_steps)
        self.sdk = sdk_status()
        self.transport = "official_sdk"
        self._arc_agi, self._game_action_type, self.sdk_error = _load_arc_sdk()
        self.env = env
        if self.env is None:
            if self.sdk_error is None and self._arc_agi is not None:
                arcade_cls = getattr(self._arc_agi, "Arcade", None)
                make_fn = getattr(self._arc_agi, "make", None)
                try:
                    if arcade_cls is not None:
                        arcade = arcade_cls()
                        self.env = arcade.make(self.game_id, render_mode="terminal") if rendering else arcade.make(self.game_id)
                    elif callable(make_fn):
                        self.env = make_fn(self.game_id, render_mode="terminal") if rendering else make_fn(self.game_id)
                except Exception as exc:
                    self.sdk_error = f"{self.sdk_error or ''} env_create_failed: {exc}".strip()
            if self.env is None:
                if allow_remote_compat:
                    self.transport = "remote_api_compat"
                    self.env = _RemoteArcCompatEnv(self.game_id, api_url=DEFAULT_API_URL, rendering=rendering)
                    self.game_id = str(getattr(self.env, "game_id", self.game_id))
                else:
                    raise ImportError(self.sdk_error or "official ARC game SDK unavailable")
        self.frame_encoder = frame_encoder
        if self.frame_encoder is None:
            try:
                self.frame_encoder = ARC3FrameEncoder()
            except Exception:
                self.frame_encoder = None
        self._knowledgeverse = knowledgeverse
        self._delegate: K3DARC3Agent | None = None
        self._episode_galaxy: Any | None = None
        self._delegate_attempted = False
        self._allow_remote_compat = bool(allow_remote_compat)
        self.policy_error: str | None = None
        self.policy_warning: str | None = None
        self.world_model: dict[str, dict[str, Any]] = {}
        self.frame_history: list[dict[str, Any]] = []
        self.step_count = 0
        self._last_decision: dict[str, Any] = {}

    def _delegate_summary(self) -> dict[str, Any]:
        delegate = self._delegate
        if delegate is None:
            return {
                "action_distribution": {},
                "spatial_plan_targets": [],
                "visited_cells_count": 0,
                "known_object_count": 0,
                "frame_color_diagnostics": [],
            }
        action_distribution: dict[str, int] = {}
        for row in list(getattr(delegate, "action_history", []) or []):
            action_name = str(row.get("action", "")).strip()
            if not action_name:
                continue
            action_distribution[action_name] = int(action_distribution.get(action_name, 0)) + 1
        spatial_plan_targets = sorted(
            {
                str(row.get("spatial_plan_target", "")).strip()
                for row in list(getattr(delegate, "action_history", []) or [])
                if str(row.get("spatial_plan_target", "")).strip()
            }
        )
        return {
            "action_distribution": action_distribution,
            "spatial_plan_targets": spatial_plan_targets,
            "visited_cells_count": int(len(getattr(delegate, "_visited_cells", set()) or set())),
            "known_object_count": int(len(getattr(delegate, "_known_objects", {}) or {})),
            "frame_color_diagnostics": list(getattr(delegate, "_frame_color_diagnostics", []) or []),
        }

    def _ensure_delegate(self) -> K3DARC3Agent | None:
        if self._delegate is not None:
            return self._delegate
        if self._delegate_attempted:
            return None
        if self.policy_error is not None:
            return None
        self._delegate_attempted = True
        try:
            if self._knowledgeverse is None:
                self._knowledgeverse = Knowledgeverse()
            self._delegate = K3DARC3Agent(max_actions=self.max_steps, knowledgeverse=self._knowledgeverse)
            self._episode_galaxy = getattr(self._delegate, "_episode_galaxy", None)
        except Exception as exc:
            message = str(exc)
            import traceback

            print(f"[ARC3] K3DARC3Agent init failed: {message}")
            traceback.print_exc()
            if self._allow_remote_compat and "sovereign_build_feed_missing" in message:
                self.policy_warning = "sovereign_build_feed_missing: proceeding with spatial primitives only"
                self.policy_error = None
            else:
                self.policy_error = message
            self._delegate = None
        return self._delegate

    def _reset_attempt_runtime(self) -> None:
        self.world_model = {}
        self.frame_history = []
        self.step_count = 0
        self._last_decision = {}
        delegate = self._ensure_delegate()
        if delegate is not None and hasattr(delegate, "reset_attempt_state"):
            delegate.reset_attempt_state()

    def observe(self, frame) -> dict[str, Any]:
        normalized = _normalize_grid(frame)
        observation: dict[str, Any] = {
            "grid": normalized,
            "height": len(normalized),
            "width": len(normalized[0]) if normalized and normalized[0] else 0,
        }
        if self.frame_encoder is not None:
            try:
                observation["embedding"] = self.frame_encoder.encode(normalized)
            except Exception as exc:
                observation["encoder_error"] = str(exc)
        return observation

    def decide_action(self, observation: dict[str, Any]) -> Any:
        frame = _normalize_grid(observation.get("grid", [[]]))
        goal_frame = observation.get("goal_frame")
        available_actions = _normalize_available_actions(
            list(observation.get("available_actions") or list(range(len(ACTION_NAMES))))
        )
        task_data = dict(observation.get("task_data") or {})
        episode_context = dict(observation.get("episode_context") or {})
        levels_completed = int(observation.get("levels_completed", 0) or 0)

        delegate = self._ensure_delegate()
        if delegate is None:
            reason = self.policy_error or self.policy_warning or "arc3_sovereign_delegate_unavailable"
            print(f"[ARC3] decide_action: delegate unavailable - {reason}")
            raise RuntimeError(str(reason))
        packet = delegate.choose_action(
            frame,
            goal_frame=goal_frame,
            task_data=task_data,
            available_actions=available_actions,
            game_id=self.game_id,
            levels_completed=levels_completed,
            episode_context=episode_context,
        )
        self._last_decision = dict(packet)
        action_name = str(packet.get("action", "")).strip().upper()
        if not action_name:
            raise RuntimeError("arc3_sovereign_action_missing")
        payload = {}
        if action_name == "ACTION6":
            if "x" not in packet or "y" not in packet:
                raise RuntimeError("arc3_click_payload_missing")
            payload = {"x": int(packet["x"]), "y": int(packet["y"])}
        return _to_sdk_action(self._game_action_type, action_name, payload)

    def update_world_model(self, prev_frame, action: Any, next_frame) -> None:
        prev_grid = _normalize_grid(prev_frame)
        next_grid = _normalize_grid(next_frame)
        changed_cells: list[dict[str, int]] = []
        rows = min(len(prev_grid), len(next_grid))
        cols = min(len(prev_grid[0]) if prev_grid else 0, len(next_grid[0]) if next_grid else 0)
        for row_idx in range(rows):
            for col_idx in range(cols):
                if int(prev_grid[row_idx][col_idx]) != int(next_grid[row_idx][col_idx]):
                    changed_cells.append(
                        {
                            "row": int(row_idx),
                            "col": int(col_idx),
                            "before": int(prev_grid[row_idx][col_idx]),
                            "after": int(next_grid[row_idx][col_idx]),
                        }
                    )
        action_name = _action_name(action)
        self.world_model[action_name] = {
            "cells_changed": len(changed_cells),
            "changed_cells": changed_cells[:64],
        }

    def _run_probe_action(self, obs: Any, frame: list[list[int]]) -> dict[str, Any] | None:
        """Send ACTION3 as the zero-context probe and seed episode memory from the transition."""
        probe_action_name = "ACTION3"
        probe_action = _to_sdk_action(self._game_action_type, probe_action_name, {})
        try:
            if self.transport == "remote_api_compat":
                next_obs, reward, done, info = self.env.step(probe_action, data=None)
            else:
                next_obs, reward, done, info = self.env.step(probe_action)
        except Exception as exc:
            print(f"[ARC3] Probe step failed: {exc}")
            return None

        next_frame = _normalize_grid(next_obs.get("grid") if isinstance(next_obs, dict) else next_obs)
        self.update_world_model(frame, probe_action, next_frame)

        prev_centroid_before = _focus_centroid(frame)
        prev_centroid_after = _focus_centroid(next_frame)
        agent_moved = (
            prev_centroid_before is not None
            and prev_centroid_after is not None
            and (
                abs(float(prev_centroid_after[0]) - float(prev_centroid_before[0])) > 0.5
                or abs(float(prev_centroid_after[1]) - float(prev_centroid_before[1])) > 0.5
            )
        )
        print(f"[ARC3] Probe ACTION3 (Move Left): agent_moved={agent_moved}")
        if prev_centroid_before is not None:
            print(f"[ARC3] Centroid before: {prev_centroid_before}")
        if prev_centroid_after is not None:
            print(f"[ARC3] Centroid after: {prev_centroid_after}")

        delegate = self._ensure_delegate()
        if delegate is not None:
            delegate._episode_galaxy.seed_frame(
                step_count=0,
                grid=frame,
                action_taken="",
                lives=-1,
                budget_pct=-1.0,
                levels_completed=0,
            )
            delegate._episode_galaxy.seed_outcome(
                step_count=0,
                action=probe_action_name,
                prev_grid=frame,
                next_grid=next_frame,
                reward=float(reward),
                lives_delta=0,
                levels_delta=int(info.get("levels_completed", 0)),
            )
            try:
                delegate._episode_galaxy.run_micro_sleeptime()
            except Exception:
                pass

        self.frame_history.append(
            {
                "step": int(self.step_count),
                "attempt_step": 0,
                "action": probe_action_name,
                "reward": float(reward),
                "levels_completed": int(info.get("levels_completed", 0)),
                "changed_cells": int(self.world_model.get(probe_action_name, {}).get("cells_changed", 0)),
                "probe": True,
            }
        )
        return {
            "obs": next_obs,
            "frame": next_frame,
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info or {}),
        }

    def run_level(self, max_steps: int | None = None, *, consolidate_episode: bool = True) -> dict[str, Any]:
        obs = self.env.reset()
        frame = _normalize_grid(obs.get("grid") if isinstance(obs, dict) else obs)
        limit = self.max_steps if max_steps is None else int(max_steps)
        levels_completed = 0
        reward_total = 0.0
        attempt_steps = 0
        episode_context_seen = False
        max_episode_rule_count = 0
        max_episode_object_count = 0
        last_info: dict[str, Any] = {}

        while attempt_steps < limit:
            delegate = self._ensure_delegate()
            episode_context: dict[str, Any] = {}
            if delegate is not None:
                budget_snapshot = _movement_budget_snapshot(frame) or {}
                budget_pct = float(budget_snapshot.get("fraction", -1.0) or -1.0)
                lives_before = _lives_remaining(frame)
                previous_action = str(self.frame_history[-1]["action"]) if self.frame_history else ""
                delegate._episode_galaxy.seed_frame(
                    step_count=int(self.step_count),
                    grid=frame,
                    action_taken=previous_action,
                    lives=int(lives_before) if lives_before is not None else -1,
                    budget_pct=budget_pct,
                    levels_completed=int(levels_completed),
                )
                episode_context = delegate._episode_galaxy.get_episode_context(last_n=10)
                episode_context_seen = episode_context_seen or bool(episode_context.get("recent_outcomes") or episode_context.get("rules"))
                max_episode_rule_count = max(max_episode_rule_count, len(list(episode_context.get("rules") or [])))
                max_episode_object_count = max(max_episode_object_count, len(dict(episode_context.get("objects") or {})))
            observation = self.observe(frame)
            observation["goal_frame"] = _goal_from_payload(obs)
            task_data = dict(obs.get("task_data", {}) if isinstance(obs, dict) else {})
            task_data["world_model"] = dict(self.world_model)
            observation["task_data"] = task_data
            observation["available_actions"] = list(obs.get("available_actions") or list(range(len(ACTION_NAMES)))) if isinstance(obs, dict) else list(range(len(ACTION_NAMES)))
            observation["levels_completed"] = int(levels_completed)
            observation["episode_context"] = episode_context
            try:
                action = self.decide_action(observation)
            except Exception as exc:
                if self.policy_error is None:
                    self.policy_error = str(exc)
                break
            step_payload: dict[str, Any] | None = None
            if _action_name(action) == "ACTION6":
                step_payload = {
                    "x": int(getattr(action, "x", 0)),
                    "y": int(getattr(action, "y", 0)),
                }
            reasoning_payload = self._last_decision.get("reasoning")
            if reasoning_payload is not None:
                if step_payload is None:
                    step_payload = {}
                step_payload["reasoning"] = reasoning_payload
            prev_lives = _lives_remaining(frame)
            prev_levels_completed = int(levels_completed)
            try:
                if self.transport == "remote_api_compat":
                    next_obs, reward, done, info = self.env.step(action, data=step_payload)
                else:
                    next_obs, reward, done, info = self.env.step(action)
            except Exception as step_exc:
                import traceback

                print(f"[ARC3] env.step() FAILED at step {self.step_count}: {step_exc}")
                traceback.print_exc()
                action_str = _action_name(action) if action is not None else "None"
                print(f"[ARC3] Failed action: {action_str!r}, payload: {step_payload!r}")
                self.frame_history.append(
                    {
                        "step": int(self.step_count),
                        "attempt_step": int(attempt_steps),
                        "action": str(action_str),
                        "reward": 0.0,
                        "levels_completed": int(levels_completed),
                        "changed_cells": 0,
                        "step_error": str(step_exc),
                    }
                )
                self.step_count += 1
                attempt_steps += 1
                continue
            if self.transport != "remote_api_compat":
                if isinstance(next_obs, tuple):
                    # Gym-style return from SDK
                    result = next_obs
                    if len(result) == 5:
                        next_obs, reward, terminated, truncated, info = result
                        done = bool(terminated or truncated)
                    elif len(result) == 4:
                        next_obs, reward, done, info = result
                elif not isinstance(info, dict):
                    info = {}
            next_frame = _normalize_grid(next_obs.get("grid") if isinstance(next_obs, dict) else next_obs)
            self.update_world_model(frame, action, next_frame)
            levels_completed = int(info.get("levels_completed", levels_completed) or levels_completed)
            last_info = dict(info or {})
            if delegate is not None:
                current_lives = _lives_remaining(next_frame)
                lives_delta = 0
                if prev_lives is not None and current_lives is not None:
                    lives_delta = int(prev_lives) - int(current_lives)
                delegate.learn_from_outcome(
                    levels_completed=levels_completed,
                    frame=next_frame,
                    action=_action_name(action),
                    prev_frame=frame,
                    reward=float(reward),
                    lives_delta=int(lives_delta),
                    levels_delta=int(levels_completed - prev_levels_completed),
                )
            self.frame_history.append(
                {
                    "step": int(self.step_count),
                    "attempt_step": int(attempt_steps),
                    "action": _action_name(action),
                    "reward": float(reward),
                    "levels_completed": int(levels_completed),
                    "changed_cells": int(self.world_model.get(_action_name(action), {}).get("cells_changed", 0)),
                }
            )
            reward_total += float(reward)
            self.step_count += 1
            attempt_steps += 1
            frame = next_frame
            obs = next_obs
            if done:
                break

        delegate = self._ensure_delegate()
        consolidation = None
        if consolidate_episode and delegate is not None:
            consolidation = delegate._episode_galaxy.consolidate_to_house(delegate.kv, float(reward_total))

        return {
            "game_id": self.game_id,
            "steps": int(attempt_steps),
            "session_steps": int(self.step_count),
            "levels_completed": int(levels_completed),
            "score": float(reward_total),
            "sdk_available": bool(self.sdk.get("package_available")),
            "game_surface_available": bool(self.sdk.get("game_surface_available")),
            "game_action_available": bool(self.sdk.get("game_action_available")),
            "sdk_error": self.sdk_error,
            "policy_error": self.policy_error,
            "policy_warning": self.policy_warning,
            "transport": self.transport,
            "episode_context_seen": bool(episode_context_seen),
            "max_episode_rule_count": int(max_episode_rule_count),
            "max_episode_object_count": int(max_episode_object_count),
            "episode_consolidation": dict(consolidation or {}),
            "card_id": str(last_info.get("card_id") or getattr(self.env, "card_id", "")),
            "scorecard_url": str(last_info.get("scorecard_url") or (_scorecard_url(DEFAULT_API_URL, getattr(self.env, "card_id", "")) if getattr(self.env, "card_id", "") else "")),
            **self._delegate_summary(),
        }

    def run_until_level_complete(
        self,
        target_levels: int = 1,
        max_attempts: int = 20,
        steps_per_attempt: int = 10000,
    ) -> dict[str, Any]:
        attempt = 0
        best_levels = -1
        best_result: dict[str, Any] = {}
        attempt_summaries: list[dict[str, Any]] = []
        crystallized_rule_ids: set[str] = set()

        while attempt < int(max_attempts):
            attempt += 1
            self._reset_attempt_runtime()
            result = self.run_level(max_steps=int(steps_per_attempt), consolidate_episode=False)
            levels_completed = int(result.get("levels_completed", 0) or 0)
            deep_summary: dict[str, Any] = {}
            if levels_completed < int(target_levels):
                episode_galaxy = self._episode_galaxy
            else:
                episode_galaxy = None
            if episode_galaxy is not None and hasattr(episode_galaxy, "run_deep_consolidation"):
                try:
                    deep_summary = dict(episode_galaxy.run_deep_consolidation() or {})
                except Exception as exc:
                    deep_summary = {"error": str(exc)}
            for rule_id in list(deep_summary.get("rule_ids") or []):
                text = str(rule_id).strip()
                if text:
                    crystallized_rule_ids.add(text)
            attempt_summary = {
                "attempt": int(attempt),
                "levels_completed": int(levels_completed),
                "steps": int(result.get("steps", 0) or 0),
                "score": float(result.get("score", 0.0) or 0.0),
                "deep_consolidation": dict(deep_summary),
            }
            attempt_summaries.append(attempt_summary)
            if levels_completed > best_levels:
                best_levels = levels_completed
                best_result = dict(result)
                best_result["first_completion_attempt"] = int(attempt) if levels_completed >= int(target_levels) else None
            if levels_completed >= int(target_levels):
                break

        final_consolidation: dict[str, Any] = {}
        episode_galaxy = self._episode_galaxy
        delegate = self._ensure_delegate()
        if episode_galaxy is not None and delegate is not None:
            final_consolidation = dict(episode_galaxy.consolidate_to_house(delegate.kv, float(best_result.get("score", 0.0) or 0.0)) or {})

        output = dict(best_result or {})
        output["autonomous"] = True
        output["attempts_used"] = int(attempt)
        output["attempt_summaries"] = attempt_summaries
        output["levels_completed_per_attempt"] = [int(row["levels_completed"]) for row in attempt_summaries]
        output["rules_crystallized_count"] = len(crystallized_rule_ids)
        output["crystallized_rule_ids"] = sorted(crystallized_rule_ids)
        output["final_episode_consolidation"] = final_consolidation
        if output.get("first_completion_attempt") is None and int(output.get("levels_completed", 0) or 0) >= int(target_levels):
            output["first_completion_attempt"] = int(attempt)
        return output

    def close(self) -> None:
        if self._delegate is not None and hasattr(self._delegate, "close"):
            try:
                self._delegate.close()
            except Exception:
                pass
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def run_ls20_test() -> dict[str, Any]:
    """Run one truthful ls20 attempt through the available ARC surface."""
    try:
        agent = K3DAgent("ls20", allow_remote_compat=False)
    except Exception as exc:
        result = {
            "game_id": "ls20",
            "steps": 0,
            "levels_completed": 0,
            "score": 0.0,
            "sdk_available": bool(sdk_status().get("package_available")),
            "game_surface_available": bool(sdk_status().get("game_surface_available")),
            "game_action_available": bool(sdk_status().get("game_action_available")),
            "sdk_error": str(exc),
            "policy_error": None,
            "policy_warning": None,
            "transport": "unavailable",
        }
        print(f"ARC-3 ls20 test: {json.dumps(result, ensure_ascii=False)}")
        return result
    try:
        result = agent.run_level()
    finally:
        agent.close()
    print(f"ARC-3 ls20 test: {json.dumps(result, ensure_ascii=False)}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", "--game-id", dest="game_id", default="ls20")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--no-remote-compat", action="store_true")
    parser.add_argument("--autonomous", action="store_true", help="Retry until target level is complete. Episode Galaxy persists across attempts.")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--target-levels", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    agent = K3DAgent(
        args.game_id,
        max_steps=args.max_steps,
        allow_remote_compat=not bool(args.no_remote_compat),
    )
    try:
        if bool(args.autonomous):
            result = agent.run_until_level_complete(
                target_levels=int(args.target_levels),
                max_attempts=int(args.max_attempts),
                steps_per_attempt=int(args.max_steps),
            )
        else:
            result = agent.run_level(max_steps=args.max_steps)
    finally:
        agent.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI surface
    raise SystemExit(main())


__all__ = ["K3DAgent", "run_ls20_test", "sdk_status"]
