"""Ephemeral ARC-3 episode memory for sovereign in-game learning."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from typing import Any

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from knowledge3d.knowledgeverse.route_contract import apply_route_capable_contract


def _normalize_grid(value: Any) -> list[list[int]]:
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        return [[int(cell) for cell in row] for row in value]
    return [[]]


def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _grid_hash(grid: list[list[int]]) -> str:
    payload = json.dumps(_normalize_grid(grid), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _background_color(grid: list[list[int]]) -> int:
    counts = Counter(int(value) for row in grid for value in row)
    if not counts:
        return 0
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _focus_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    if not grid or not grid[0]:
        return None
    background = _background_color(grid)
    counts = Counter(int(value) for row in grid for value in row if int(value) != background)
    if not counts:
        return None
    preferred_colors = [0, 1, 6, 9, 11, 12, 15]
    candidate_colors = [
        color
        for color in preferred_colors
        if color in counts and counts[color] <= 128
    ]
    if not candidate_colors:
        rarest_count = min(counts.values())
        candidate_colors = [
            color
            for color, count in counts.items()
            if count == rarest_count or count <= max(rarest_count * 2, 64)
        ]
    cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if int(value) in set(candidate_colors)
    ]
    if not cells:
        return None
    avg_row = sum(float(row_index) for row_index, _ in cells) / float(len(cells))
    avg_col = sum(float(col_index) for _, col_index in cells) / float(len(cells))
    return avg_row, avg_col


def _foreground_colors(grid: list[list[int]]) -> list[int]:
    if not grid or not grid[0]:
        return []
    background = _background_color(grid)
    return sorted(
        {
            int(value)
            for row in grid
            for value in row
            if int(value) != background
        }
    )


def _grid_dimensions(grid: list[list[int]]) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows and isinstance(grid[0], list) else 0
    return rows, cols


def _movement_delta(action: str) -> tuple[int, int] | None:
    action_name = str(action or "").strip().upper()
    if action_name == "ACTION1":
        return (-1, 0)
    if action_name == "ACTION2":
        return (1, 0)
    if action_name == "ACTION3":
        return (0, -1)
    if action_name == "ACTION4":
        return (0, 1)
    return None


def _action_name_to_index(action_name: str) -> int | None:
    try:
        value = int(str(action_name or "").strip().upper().replace("ACTION", "")) - 1
    except Exception:
        return None
    if 0 <= int(value) <= 6:
        return int(value)
    return None


def _adjacent_color(grid: list[list[int]], action: str) -> int | None:
    if not grid or not grid[0]:
        return None
    centroid = _focus_centroid(grid)
    delta = _movement_delta(action)
    if centroid is None or delta is None:
        return None
    row = max(0, min(len(grid) - 1, int(round(float(centroid[0]))) + int(delta[0])))
    col = max(0, min(len(grid[0]) - 1, int(round(float(centroid[1]))) + int(delta[1])))
    try:
        return int(grid[row][col])
    except Exception:
        return None


def _changed_cells(prev_grid: list[list[int]], next_grid: list[list[int]]) -> int:
    rows = min(len(prev_grid), len(next_grid))
    cols = min(
        len(prev_grid[0]) if rows and prev_grid[0] else 0,
        len(next_grid[0]) if rows and next_grid[0] else 0,
    )
    count = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if int(prev_grid[row_idx][col_idx]) != int(next_grid[row_idx][col_idx]):
                count += 1
    return int(count)


def _detect_gpu_count() -> int:
    try:
        import torch  # type: ignore
    except Exception:
        return 0
    try:
        return max(0, int(torch.cuda.device_count()))
    except Exception:
        return 0


class ARC3EpisodeGalaxy:
    """VRAM-era episodic memory controller for ARC-3 sessions."""

    def __init__(self, game_id: str, knowledgeverse: Any) -> None:
        self.game_id = str(game_id or "unknown")
        self.knowledgeverse = knowledgeverse
        self.frames: deque[dict[str, Any]] = deque(maxlen=4096)
        self.outcomes: deque[dict[str, Any]] = deque(maxlen=4096)
        self._objects_by_color: dict[int, dict[str, Any]] = {}
        self._rules_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._pending_futures: list[Future[Any]] = []
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="arc3_micro_sleep")
        self._gpu_count = _detect_gpu_count()
        self.device_roles = self._resolve_device_roles(self._gpu_count)
        self._last_sleep_errors: list[str] = []
        self._seed_game_mechanics_priors()
        loaded = self._load_persisted_rules()
        if loaded > 0:
            print(f"[ARC3-LEARN] Loaded {int(loaded)} persisted rules from House")

    def _seed_game_mechanics_priors(self) -> None:
        rule = {
            "type": "ARC3_RULE",
            "game_id": self.game_id,
            "condition": "agent_adjacent_to_untested_object",
            "action": "ACTION5",
            "predicted_outcome": "interact_probe",
            "confidence": 0.7,
            "evidence_count": 3,
            "source": "game_mechanics_prior",
            "galaxy_family": "GRAMMAR",
        }
        self._rules_by_key[("ACTION5", "agent_adjacent_to_untested_object")] = dict(rule)
        self._upsert_live_rule(rule, source="game_mechanics_prior")

    def _rule_entry(self, rule: dict[str, Any], *, source: str) -> dict[str, Any] | None:
        condition = str(rule.get("condition", "")).strip()
        action_name = str(rule.get("action", "")).strip().upper()
        predicted_outcome = str(rule.get("predicted_outcome", "")).strip() or "neutral"
        confidence = float(rule.get("confidence", 0.0) or 0.0)
        evidence = max(1, int(rule.get("evidence_count", 0) or 0))
        action_index = _action_name_to_index(action_name)
        if not condition or action_index is None:
            return None
        condition_tokens = condition.replace("_", " ").strip()
        action_label = action_name
        query_anchor = (
            f"arc3 game rule {condition_tokens} {action_name.lower()} {predicted_outcome} "
            f"answer action episode rule live gameplay"
        ).strip()
        entry = {
            "id": f"arc3_rule:{self.game_id}:{condition}:{action_name}",
            "name": f"ARC3 Rule {condition_tokens} -> {action_name}",
            "domain": "arc3_game_rule",
            "category": "game_rule",
            "content": (
                f"arc3 game rule when {condition_tokens} then {action_name.lower()} "
                f"predicts {predicted_outcome}"
            ).strip(),
            "description": (
                f"Live ARC-3 episode rule for {self.game_id}: {condition_tokens} -> {action_name}"
            ).strip(),
            "rpn_program": f"{condition.upper()} {action_name} {predicted_outcome.upper()}",
            "confidence": float(confidence),
            "tags": ["arc3", "game_rule", "live_episode", action_name.lower()],
            "meta_refs": [
                "answer_kind:action",
                f"action_index:{int(action_index)}",
                f"action_name:{action_name}",
                f"condition:{condition}",
                f"predicted_outcome:{predicted_outcome}",
                f"confidence:{confidence:.3f}",
                f"evidence_count:{int(evidence)}",
            ],
            "metadata": {
                "source": str(source or "arc3_episode_live"),
                "game_id": self.game_id,
                "condition": condition,
                "predicted_outcome": predicted_outcome,
                "rule_confidence": float(confidence),
                "evidence_count": int(evidence),
                "query_anchor": query_anchor,
                "answer_kind": "action",
                "action_index": int(action_index),
                "action_name": action_name,
                "action_label": action_label,
            },
        }
        return apply_route_capable_contract(
            entry,
            {
                "route_family": "GAME_2D",
                "selection_role": "validator",
                "layer_id": 4,
                "answer_eligible": True,
                "route_policy": {"branch_topk": 0},
            },
        )

    def _upsert_live_rule(self, rule: dict[str, Any], *, source: str) -> str | None:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None or not hasattr(galaxy_manager, "upsert_entry"):
            return None
        entry = self._rule_entry(rule, source=source)
        if entry is None:
            return None
        return str(galaxy_manager.upsert_entry("Grammar", entry))

    @staticmethod
    def _meta_refs_to_map(meta_refs: Any) -> dict[str, str]:
        values = list(meta_refs or []) if isinstance(meta_refs, (list, tuple)) else []
        parsed: dict[str, str] = {}
        for value in values:
            text = str(value or "").strip()
            if ":" not in text:
                continue
            key, payload = text.split(":", 1)
            parsed[str(key).strip()] = str(payload).strip()
        return parsed

    def _rule_from_entry(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id.startswith("arc3_rule:"):
            return None
        metadata = entry.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        meta_ref_map = self._meta_refs_to_map(entry.get("meta_refs"))
        id_parts = entry_id.split(":", 3)
        id_game = id_parts[1] if len(id_parts) >= 2 else ""
        id_condition = id_parts[2] if len(id_parts) >= 3 else ""
        id_action = id_parts[3] if len(id_parts) >= 4 else ""
        condition = str(
            metadata_dict.get("condition")
            or meta_ref_map.get("condition")
            or id_condition
        ).strip()
        action_name = str(
            metadata_dict.get("action_name")
            or meta_ref_map.get("action_name")
            or id_action
        ).strip().upper()
        if not condition or not action_name.startswith("ACTION"):
            return None
        game_id = str(metadata_dict.get("game_id", id_game)).strip()
        if game_id not in {"", self.game_id}:
            return None
        predicted_outcome = str(
            metadata_dict.get("predicted_outcome")
            or meta_ref_map.get("predicted_outcome")
            or "neutral"
        ).strip() or "neutral"
        try:
            confidence = float(
                metadata_dict.get("rule_confidence", entry.get("confidence", 0.0)) or 0.0
            )
        except Exception:
            confidence = 0.0
        try:
            evidence = int(
                metadata_dict.get("evidence_count")
                or meta_ref_map.get("evidence_count")
                or 1
            )
        except Exception:
            evidence = 1
        return {
            "type": "ARC3_RULE",
            "game_id": self.game_id,
            "condition": condition,
            "action": action_name,
            "predicted_outcome": predicted_outcome,
            "confidence": float(confidence),
            "evidence_count": max(1, int(evidence)),
            "source": str(metadata_dict.get("source", "house_persisted") or "house_persisted"),
            "galaxy_family": "GRAMMAR",
        }

    def _load_persisted_rules(self) -> int:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None or not hasattr(galaxy_manager, "get_galaxy"):
            return 0
        try:
            grammar = galaxy_manager.get_galaxy("Grammar")
        except Exception:
            return 0
        rows = list(getattr(grammar, "entries", []) or [])
        loaded = 0
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            hydrated = self._rule_from_entry(raw)
            if hydrated is None:
                continue
            key = (str(hydrated["action"]), str(hydrated["condition"]))
            source = str(hydrated.get("source", "")).strip().lower()
            if key in self._rules_by_key and source == "game_mechanics_prior":
                continue
            self._rules_by_key[key] = dict(hydrated)
            loaded += 1
        return int(loaded)

    def _object_entry(self, record: dict[str, Any]) -> dict[str, Any] | None:
        color = record.get("color")
        if not isinstance(color, int):
            return None
        behavior = str(record.get("behavior", "unknown") or "unknown").strip().lower()
        evidence = max(1, int(record.get("evidence_count", 0) or 0))
        confidence = float(record.get("confidence", 0.0) or 0.0)
        name = f"ARC3 Object color {int(color)}"
        content = (
            f"arc3 object color {int(color)} behaves as {behavior} "
            f"evidence {int(evidence)} confidence {float(confidence):.3f}"
        )
        entry = {
            "id": f"arc3_object:{self.game_id}:color_{int(color)}",
            "name": name,
            "domain": "arc3_object",
            "category": "live_object",
            "content": content,
            "description": f"Live ARC-3 object model for color {int(color)} in {self.game_id}",
            "rpn_program": f"OBJECT COLOR_{int(color)} BEHAVIOR_{behavior.upper()}",
            "confidence": float(confidence),
            "tags": ["arc3", "object", behavior],
            "meta_refs": [
                f"color:{int(color)}",
                f"behavior:{behavior}",
                f"evidence_count:{int(evidence)}",
                f"blocking_count:{int(record.get('blocking_count', 0) or 0)}",
                f"death_count:{int(record.get('death_count', 0) or 0)}",
                f"reward_count:{int(record.get('reward_count', 0) or 0)}",
            ],
            "metadata": {
                "source": "arc3_episode_live_object",
                "game_id": self.game_id,
                "color": int(color),
                "behavior": behavior,
                "evidence_count": int(evidence),
                "blocking_count": int(record.get("blocking_count", 0) or 0),
                "death_count": int(record.get("death_count", 0) or 0),
                "reward_count": int(record.get("reward_count", 0) or 0),
                "confidence": float(confidence),
                "answer_kind": "state",
            },
        }
        return apply_route_capable_contract(
            entry,
            {
                "route_family": "GAME_2D",
                "selection_role": "observer",
                "layer_id": 4,
                "answer_eligible": False,
                "route_policy": {"branch_topk": 0},
            },
        )

    def _upsert_live_object(self, record: dict[str, Any]) -> str | None:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None or not hasattr(galaxy_manager, "upsert_entry"):
            return None
        entry = self._object_entry(record)
        if entry is None:
            return None
        return str(galaxy_manager.upsert_entry("Grammar", entry))

    def _observation_entry(self, outcome: dict[str, Any]) -> dict[str, Any] | None:
        action_name = str(outcome.get("action", "")).strip().upper()
        step_count = int(outcome.get("step_count", len(self.outcomes) - 1) or 0)
        if bool(outcome.get("is_level_complete", False)):
            kind = "level_complete"
        elif bool(outcome.get("is_death", False)):
            kind = "death"
        elif bool(outcome.get("is_blocked", False)):
            kind = "blocked"
        else:
            return None
        adjacent_color = outcome.get("adjacent_color")
        color_value = int(adjacent_color) if isinstance(adjacent_color, int) else -1
        content_tokens = [
            "arc3 significant observation",
            f"step {int(step_count)}",
            action_name.lower() if action_name else "",
            kind,
        ]
        if color_value >= 0:
            content_tokens.append(f"color {int(color_value)}")
        entry = {
            "id": f"arc3_observation:{self.game_id}:{int(step_count)}:{action_name or 'ACTION0'}:{kind}",
            "name": f"ARC3 Observation {kind} {action_name or 'ACTION?'}",
            "domain": "arc3_observation",
            "category": "live_observation",
            "content": " ".join(token for token in content_tokens if token).strip(),
            "description": f"Significant ARC-3 outcome ({kind}) captured during gameplay.",
            "rpn_program": f"OBSERVE STEP_{int(step_count)} {action_name or 'ACTION0'} {kind.upper()}",
            "confidence": 1.0,
            "tags": ["arc3", "observation", kind, action_name.lower() if action_name else "action_unknown"],
            "meta_refs": [
                f"step:{int(step_count)}",
                f"action:{action_name or 'ACTION0'}",
                f"kind:{kind}",
                f"adjacent_color:{int(color_value)}" if color_value >= 0 else "adjacent_color:unknown",
            ],
            "metadata": {
                "source": "arc3_episode_observation",
                "game_id": self.game_id,
                "step_count": int(step_count),
                "action_name": action_name,
                "kind": kind,
                "adjacent_color": color_value if color_value >= 0 else None,
                "answer_kind": "event",
            },
        }
        return apply_route_capable_contract(
            entry,
            {
                "route_family": "GAME_2D",
                "selection_role": "observer",
                "layer_id": 4,
                "answer_eligible": False,
                "route_policy": {"branch_topk": 0},
            },
        )

    def _upsert_observation_star(self, outcome: dict[str, Any]) -> str | None:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None or not hasattr(galaxy_manager, "upsert_entry"):
            return None
        entry = self._observation_entry(outcome)
        if entry is None:
            return None
        return str(galaxy_manager.upsert_entry("Grammar", entry))

    def _grammar_star_count(self) -> int:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None or not hasattr(galaxy_manager, "get_galaxy"):
            return 0
        try:
            grammar = galaxy_manager.get_galaxy("Grammar")
        except Exception:
            return 0
        rows = list(getattr(grammar, "entries", []) or [])
        return int(len(rows))

    def _latest_step(self) -> int:
        if self.outcomes:
            return int(self.outcomes[-1].get("step_count", len(self.outcomes) - 1) or 0)
        if self.frames:
            return int(self.frames[-1].get("step_count", len(self.frames) - 1) or 0)
        return 0

    def _emit_learning_log(self) -> None:
        recent = list(self.outcomes)[-5:]
        last_actions = ",".join(str(r.get("action", "?")) for r in recent) if recent else "none"
        last_blocked = ",".join(str(bool(r.get("is_blocked", False))) for r in recent) if recent else "none"
        print(
            "[ARC3-LEARN] "
            f"step={int(self._latest_step())} "
            f"rules={len(self._rules_by_key)} "
            f"objects={len(self._objects_by_color)} "
            f"galaxy_stars={int(self._grammar_star_count())} "
            f"last_actions={last_actions} "
            f"last_blocked={last_blocked}"
        )

    @staticmethod
    def _resolve_device_roles(gpu_count: int) -> dict[str, int]:
        if int(gpu_count) >= 4:
            return {
                "inference": 0,
                "crystallize": 1,
                "reinforce": 2,
                "object": 3,
            }
        return {
            "inference": 0,
            "crystallize": 0,
            "reinforce": 0,
            "object": 0,
        }

    def _drain_pending_futures(self) -> None:
        if not self._pending_futures:
            return
        remaining: list[Future[Any]] = []
        errors: list[str] = []
        resolved_any = False
        for future in self._pending_futures:
            if not future.done():
                remaining.append(future)
                continue
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
            resolved_any = True
        self._last_sleep_errors = errors[-8:]
        self._pending_futures = remaining
        if resolved_any and not self._pending_futures:
            self._emit_learning_log()

    def seed_frame(
        self,
        step_count: int,
        grid: list[list[int]],
        action_taken: str = "",
        lives: int = -1,
        budget_pct: float = -1.0,
        levels_completed: int = 0,
    ) -> None:
        self._drain_pending_futures()
        normalized = _normalize_grid(grid)
        rows, cols = _grid_dimensions(normalized)
        centroid = _focus_centroid(normalized)
        frame_record = {
            "type": "ARC3_FRAME",
            "game_id": self.game_id,
            "step_count": int(step_count),
            "grid_hash": _grid_hash(normalized),
            "grid_height": int(rows),
            "grid_width": int(cols),
            "agent_row": float(centroid[0]) if centroid is not None else -1.0,
            "agent_col": float(centroid[1]) if centroid is not None else -1.0,
            "foreground_colors": _foreground_colors(normalized),
            "action_taken": str(action_taken or ""),
            "budget_pct": float(budget_pct),
            "lives_remaining": int(lives),
            "level": int(levels_completed),
        }
        self.frames.append(frame_record)
        avatar_color = _adjacent_color(normalized, "ACTION6")
        if avatar_color is not None:
            record = self._objects_by_color.setdefault(
                int(avatar_color),
                {
                    "type": "ARC3_OBJECT",
                    "game_id": self.game_id,
                    "color": int(avatar_color),
                    "behavior": "avatar",
                    "evidence_count": 0,
                    "blocking_count": 0,
                    "death_count": 0,
                    "reward_count": 0,
                    "confidence": 0.0,
                },
            )
            record["behavior"] = "avatar"
            record["evidence_count"] = int(record.get("evidence_count", 0)) + 1
            record["confidence"] = float(record["evidence_count"]) / float(record["evidence_count"] + 1)

    def seed_outcome(
        self,
        step_count: int,
        action: str,
        prev_grid: list[list[int]],
        next_grid: list[list[int]],
        reward: float,
        lives_delta: int,
        levels_delta: int,
    ) -> None:
        normalized_prev = _normalize_grid(prev_grid)
        normalized_next = _normalize_grid(next_grid)
        prev_centroid = _focus_centroid(normalized_prev)
        next_centroid = _focus_centroid(normalized_next)
        changed = _changed_cells(normalized_prev, normalized_next)
        moved = False
        if prev_centroid is not None and next_centroid is not None:
            moved = abs(float(prev_centroid[0]) - float(next_centroid[0])) > 0.5 or abs(
                float(prev_centroid[1]) - float(next_centroid[1])
            ) > 0.5
        outcome = {
            "type": "ARC3_OUTCOME",
            "game_id": self.game_id,
            "step_count": int(step_count),
            "action": str(action or ""),
            "cells_changed": int(changed),
            "agent_moved": bool(moved),
            "reward": float(reward),
            "is_blocked": bool(
                _movement_delta(action) is not None
                and not bool(moved)
                and int(levels_delta) <= 0
                and int(lives_delta) <= 0
            ),
            "is_death": bool(int(lives_delta) > 0),
            "is_level_complete": bool(int(levels_delta) > 0),
            "prev_frame_hash": _grid_hash(normalized_prev),
            "next_frame_hash": _grid_hash(normalized_next),
            "adjacent_color": _adjacent_color(normalized_prev, action),
        }
        self.outcomes.append(outcome)
        self._update_object_records(outcome, normalized_prev, normalized_next)
        if (
            bool(outcome.get("is_blocked", False))
            or bool(outcome.get("is_death", False))
            or bool(outcome.get("is_level_complete", False))
        ):
            self._upsert_observation_star(outcome)

    def _update_object_records(
        self,
        outcome: dict[str, Any],
        prev_grid: list[list[int]],
        next_grid: list[list[int]],
    ) -> None:
        for color in _foreground_colors(prev_grid) + _foreground_colors(next_grid):
            color_key = int(color)
            if color_key in self._objects_by_color:
                continue
            self._objects_by_color[color_key] = {
                "type": "ARC3_OBJECT",
                "game_id": self.game_id,
                "color": color_key,
                "behavior": "unknown",
                "evidence_count": 0,
                "blocking_count": 0,
                "death_count": 0,
                "reward_count": 0,
                "confidence": 0.0,
            }
            self._upsert_live_object(self._objects_by_color[color_key])
        target_color = outcome.get("adjacent_color")
        if target_color is None:
            return
        created = int(target_color) not in self._objects_by_color
        record = self._objects_by_color.setdefault(
            int(target_color),
            {
                "type": "ARC3_OBJECT",
                "game_id": self.game_id,
                "color": int(target_color),
                "behavior": "unknown",
                "evidence_count": 0,
                "blocking_count": 0,
                "death_count": 0,
                "reward_count": 0,
                "confidence": 0.0,
            },
        )
        previous_behavior = str(record.get("behavior", "unknown"))
        previous_counts = (
            int(record.get("evidence_count", 0) or 0),
            int(record.get("blocking_count", 0) or 0),
            int(record.get("death_count", 0) or 0),
            int(record.get("reward_count", 0) or 0),
        )
        record["evidence_count"] = int(record.get("evidence_count", 0)) + 1
        if bool(outcome.get("is_blocked", False)):
            record["blocking_count"] = int(record.get("blocking_count", 0)) + 1
            record["behavior"] = "static_obstacle"
        elif bool(outcome.get("is_death", False)):
            record["death_count"] = int(record.get("death_count", 0)) + 1
            record["behavior"] = "hazard"
        elif bool(outcome.get("is_level_complete", False)):
            record["reward_count"] = int(record.get("reward_count", 0)) + 1
            record["behavior"] = "goal"
        elif float(outcome.get("reward", 0.0) or 0.0) > 0.0:
            record["reward_count"] = int(record.get("reward_count", 0)) + 1
            record["behavior"] = "collectible"
        elif bool(outcome.get("agent_moved", False)):
            record["behavior"] = "walkable"
        record["confidence"] = float(record["evidence_count"]) / float(record["evidence_count"] + 1)
        current_counts = (
            int(record.get("evidence_count", 0) or 0),
            int(record.get("blocking_count", 0) or 0),
            int(record.get("death_count", 0) or 0),
            int(record.get("reward_count", 0) or 0),
        )
        if created or previous_behavior != str(record.get("behavior", "unknown")) or previous_counts != current_counts:
            self._upsert_live_object(record)

    def seed_object(self, obj: dict[str, Any]) -> None:
        color = obj.get("color")
        if not isinstance(color, int):
            return
        color_key = int(color)
        record = self._objects_by_color.setdefault(
            color_key,
            {
                "type": "ARC3_OBJECT",
                "game_id": self.game_id,
                "color": color_key,
                "behavior": "unknown",
                "evidence_count": 0,
                "blocking_count": 0,
                "death_count": 0,
                "reward_count": 0,
                "confidence": 0.0,
            },
        )
        record["evidence_count"] = int(record.get("evidence_count", 0)) + 1
        semantic_hint = str(obj.get("semantic_hint", "")).strip().lower()
        if semantic_hint and str(record.get("behavior", "unknown")) in {"", "unknown", "walkable"}:
            record["behavior"] = semantic_hint
        for key in ("position_label", "approximate_position", "interaction_result"):
            value = obj.get(key)
            if value is None:
                continue
            record[key] = value
        if "centroid" in obj and isinstance(obj.get("centroid"), (tuple, list)) and len(obj["centroid"]) == 2:
            row_value, col_value = obj["centroid"]
            record["centroid"] = (float(row_value), float(col_value))
        if "size" in obj:
            try:
                record["size"] = int(obj.get("size", 0) or 0)
            except Exception:
                pass
        if "moves_with_avatar" in obj:
            record["moves_with_avatar"] = bool(obj.get("moves_with_avatar"))
        if "interaction_tested" in obj:
            record["interaction_tested"] = bool(obj.get("interaction_tested"))
        record["confidence"] = float(record["evidence_count"]) / float(record["evidence_count"] + 1)
        self._upsert_live_object(record)

    def seed_rule(
        self,
        *,
        state: str,
        action: int | str,
        outcome: str,
        confidence: float,
        evidence_count: int = 3,
        source: str = "episode_learning",
    ) -> None:
        condition = str(state or "").strip()
        if not condition:
            return
        if isinstance(action, int):
            action_index = int(action)
            if 0 <= action_index <= 6:
                action_name = f"ACTION{action_index + 1}"
            else:
                return
        else:
            action_name = str(action or "").strip().upper()
            if not action_name.startswith("ACTION"):
                return
        self._rules_by_key[(action_name, condition)] = {
            "type": "ARC3_RULE",
            "game_id": self.game_id,
            "condition": condition,
            "action": action_name,
            "predicted_outcome": str(outcome or "").strip() or "neutral",
            "confidence": float(confidence),
            "evidence_count": max(1, int(evidence_count)),
            "source": str(source or "episode_learning"),
            "galaxy_family": "GRAMMAR",
        }
        self._upsert_live_rule(self._rules_by_key[(action_name, condition)], source=str(source or "episode_learning"))

    def _crystallize_rules(self, *, full_history: bool = False) -> None:
        recent = list(self.outcomes) if bool(full_history) else list(self.outcomes)[-20:]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for outcome in recent:
            adjacent_color = outcome.get("adjacent_color")
            if adjacent_color is None:
                continue
            action_name = str(outcome.get("action", "")).strip().upper()
            if not action_name.startswith("ACTION"):
                continue
            key = (action_name, f"agent_adjacent_to_color_{int(adjacent_color)}")
            grouped.setdefault(key, []).append(outcome)
        for (action, condition), rows in grouped.items():
            if not rows:
                continue
            outcome_counts = Counter()
            for row in rows:
                if bool(row.get("is_level_complete", False)):
                    outcome_counts["level_complete"] += 1
                elif bool(row.get("is_death", False)):
                    outcome_counts["death"] += 1
                elif bool(row.get("is_blocked", False)):
                    outcome_counts["blocked"] += 1
                elif bool(row.get("agent_moved", False)):
                    outcome_counts["moved"] += 1
                else:
                    outcome_counts["neutral"] += 1
            predicted_outcome, majority = max(outcome_counts.items(), key=lambda item: (item[1], item[0]))
            total = len(rows)
            confidence = (float(majority) / float(max(1, total))) * min(1.0, float(total) / 5.0)
            self._rules_by_key[(action, condition)] = {
                "type": "ARC3_RULE",
                "game_id": self.game_id,
                "condition": condition,
                "action": action,
                "predicted_outcome": predicted_outcome,
                "confidence": float(confidence),
                "evidence_count": int(total),
                "source": "micro_sleeptime",
                "galaxy_family": "GRAMMAR",
            }
            self._upsert_live_rule(self._rules_by_key[(action, condition)], source="micro_sleeptime")
            if predicted_outcome in {"blocked", "death"}:
                for alt_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                    if alt_action == action:
                        continue
                    alt_confidence = max(0.05, float(confidence) * 0.6)
                    avoidance = {
                        "type": "ARC3_RULE",
                        "game_id": self.game_id,
                        "condition": condition,
                        "action": alt_action,
                        "predicted_outcome": f"avoid_{predicted_outcome}",
                        "confidence": float(alt_confidence),
                        "evidence_count": int(total),
                        "source": "micro_sleeptime_avoidance",
                        "galaxy_family": "GRAMMAR",
                    }
                    self._rules_by_key[(alt_action, condition)] = dict(avoidance)
                    self._upsert_live_rule(avoidance, source="micro_sleeptime_avoidance")

    def query_rule_for_state(
        self,
        *,
        avatar_centroid: tuple[float, float] | None,
        adjacent_colors: list[int] | set[int] | tuple[int, ...],
        frame_state: str = "gameplay",
        blocked_actions: set[int] | None = None,
        has_adjacent_untested: bool = False,
    ) -> int | None:
        if avatar_centroid is None or str(frame_state or "gameplay") != "gameplay":
            return None
        adjacent = {
            int(color)
            for color in list(adjacent_colors or [])
            if isinstance(color, int)
        }
        if not adjacent:
            return None
        blocked = {int(index) for index in set(blocked_actions or set())}
        outcome_rank = {
            "level_complete": 0,
            "collectible": 1,
            "moved": 2,
            "neutral": 3,
            "blocked": 4,
            "death": 5,
        }
        candidates: list[tuple[int, float, int, int]] = []
        for rule in self._rules_by_key.values():
            condition = str(rule.get("condition", "")).strip()
            color_prefix = "agent_adjacent_to_color_"
            object_prefix = "agent_near_color_"
            matches = False
            if condition == "agent_adjacent_to_untested_object":
                matches = bool(has_adjacent_untested)
            elif condition.startswith(color_prefix):
                try:
                    color = int(condition[len(color_prefix) :])
                except Exception:
                    continue
                matches = color in adjacent
            elif condition.startswith(object_prefix) and condition.endswith("_object"):
                try:
                    color = int(condition[len(object_prefix) : -len("_object")])
                except Exception:
                    continue
                matches = color in adjacent
            if not matches:
                continue
            action_name = str(rule.get("action", "")).strip().upper()
            if not action_name.startswith("ACTION"):
                continue
            try:
                action_index = int(action_name.replace("ACTION", "")) - 1
            except Exception:
                continue
            if action_index in blocked or not (0 <= action_index <= 6):
                continue
            predicted = str(rule.get("predicted_outcome", "")).strip().lower()
            if predicted in {"blocked", "death"}:
                continue
            candidates.append(
                (
                    int(outcome_rank.get(predicted, 3)),
                    -float(rule.get("confidence", 0.0) or 0.0),
                    -int(rule.get("evidence_count", 0) or 0),
                    int(action_index),
                )
            )
        if not candidates:
            return None
        _, _, _, action_index = min(candidates)
        return int(action_index)

    def _reinforce_routes(self) -> None:
        shadow_copy = getattr(self.knowledgeverse, "shadow_copy", None)
        if shadow_copy is None or not hasattr(shadow_copy, "record_event"):
            return
        recent = list(self.outcomes)[-8:]
        reward_total = sum(float(row.get("reward", 0.0) or 0.0) for row in recent)
        blocked_total = sum(1 for row in recent if bool(row.get("is_blocked", False)))
        shadow_copy.record_event(
            event_type="arc3_micro_reinforcement",
            event_data={
                "game_id": self.game_id,
                "recent_reward_total": float(reward_total),
                "blocked_events": int(blocked_total),
                "episode_steps": len(self.outcomes),
            },
        )

    def _classify_objects(self) -> None:
        for record in self._objects_by_color.values():
            previous_behavior = str(record.get("behavior", "unknown"))
            evidence_count = int(record.get("evidence_count", 0))
            blocking_count = int(record.get("blocking_count", 0))
            death_count = int(record.get("death_count", 0))
            reward_count = int(record.get("reward_count", 0))
            if death_count > 0:
                record["behavior"] = "hazard"
            elif reward_count > 0 and str(record.get("behavior", "")) != "goal":
                record["behavior"] = "collectible"
            elif blocking_count > 0:
                record["behavior"] = "static_obstacle"
            elif evidence_count > 0 and str(record.get("behavior", "")) == "unknown":
                record["behavior"] = "walkable"
            record["confidence"] = float(evidence_count) / float(evidence_count + 1)
            if previous_behavior != str(record.get("behavior", "unknown")):
                self._upsert_live_object(record)

    def run_micro_sleeptime(self) -> None:
        self._drain_pending_futures()
        if self._pending_futures:
            return
        self._pending_futures = [
            self._executor.submit(self._crystallize_rules),
            self._executor.submit(self._reinforce_routes),
            self._executor.submit(self._classify_objects),
        ]

    def get_episode_context(self, last_n: int = 10) -> dict[str, Any]:
        return {
            "objects": {
                int(color): str(record.get("behavior", "unknown"))
                for color, record in sorted(self._objects_by_color.items())
                if str(record.get("behavior", "unknown")).strip()
            },
            "recent_outcomes": [dict(row) for row in list(self.outcomes)[-max(1, int(last_n)):]],
        }

    def _strong_rules(self) -> list[dict[str, Any]]:
        return [
            dict(rule)
            for rule in self._rules_by_key.values()
            if float(rule.get("confidence", 0.0) or 0.0) >= 0.5
            and int(rule.get("evidence_count", 0) or 0) >= 3
        ]

    def _persist_strong_rules(
        self,
        *,
        min_confidence: float = 0.5,
        min_evidence: int = 3,
    ) -> list[str]:
        galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None:
            return []
        rule_ids: list[str] = []
        sync_context = (
            galaxy_manager.bulk_disk_sync()
            if hasattr(galaxy_manager, "bulk_disk_sync")
            else nullcontext()
        )
        with sync_context:
            for rule in self._rules_by_key.values():
                confidence = float(rule.get("confidence", 0.0) or 0.0)
                evidence = int(rule.get("evidence_count", 0) or 0)
                predicted_outcome = str(rule.get("predicted_outcome", "")).strip().lower()
                required_evidence = 2 if predicted_outcome in {"blocked", "death"} else int(min_evidence)
                if confidence < float(min_confidence) or evidence < int(required_evidence):
                    continue
                entry = self._rule_entry(rule, source="arc3_episode_galaxy")
                if entry is None:
                    continue
                galaxy_manager.upsert_entry("Grammar", entry)
                rule_ids.append(str(entry["id"]))
        return rule_ids

    def run_deep_consolidation(self) -> dict[str, Any]:
        """Between-attempt full crystallization across all accumulated history."""
        self._drain_pending_futures()
        self._crystallize_rules(full_history=True)
        self._classify_objects()
        persisted = self._persist_strong_rules(min_confidence=0.6, min_evidence=3)
        return {
            "rules_persisted": len(persisted),
            "rule_ids": list(persisted),
            "object_count": len(self._objects_by_color),
            "episode_frames": len(self.frames),
            "episode_outcomes": len(self.outcomes),
        }

    def consolidate_to_house(self, knowledgeverse: Any, final_score: float) -> dict[str, int]:
        self._drain_pending_futures()
        galaxy_manager = getattr(knowledgeverse, "galaxy_manager", None)
        if galaxy_manager is None:
            self.frames.clear()
            self.outcomes.clear()
            self._rules_by_key.clear()
            return {"rules_persisted": 0, "session_entries": 0}
        sync_context = (
            galaxy_manager.bulk_disk_sync()
            if hasattr(galaxy_manager, "bulk_disk_sync")
            else nullcontext()
        )
        rules_persisted = 0
        session_entries = 0
        with sync_context:
            rules_persisted += len(self._persist_strong_rules(min_confidence=0.5, min_evidence=3))
            session_star = MeaningCentricStar(
                star_id=f"arc3_game_room:{self.game_id}",
                meaning_class="arc3_game_room",
                domain="reality",
                galaxy_ref="Reality",
                meaning_rpn=f"GAME {self.game_id} SESSION SCORE {float(final_score):.3f}",
                behavior_rpn=f"RULES_DISCOVERED {int(rules_persisted)}",
                taxonomy_refs=["arc3", "game_room", "living_memory"],
                reality_refs=["arc3", self.game_id],
                meta_refs=[
                    f"score:{float(final_score):.3f}",
                    f"rules_discovered:{int(rules_persisted)}",
                    f"episode_steps:{len(self.outcomes)}",
                ],
                confidence=1,
                polarity=1,
            )
            galaxy_manager.store_meaning_star(
                "Reality",
                session_star,
                category="arc3_game_room",
                metadata={
                    "source": "arc3_episode_galaxy",
                    "game_id": self.game_id,
                    "final_score": float(final_score),
                    "episode_steps": len(self.outcomes),
                },
            )
            session_entries += 1
            completion_rows = [row for row in self.outcomes if bool(row.get("is_level_complete", False))]
            if completion_rows:
                winning_actions = [
                    str(row.get("action", "")).strip()
                    for row in self.outcomes
                    if str(row.get("action", "")).strip()
                ]
                trace_star = MeaningCentricStar(
                    star_id=f"arc3_winning_trace:{self.game_id}",
                    meaning_class="arc3_winning_trace",
                    domain="reality",
                    galaxy_ref="Reality",
                    meaning_rpn=" ".join(winning_actions[:64]).strip(),
                    behavior_rpn="LEVEL_COMPLETE",
                    taxonomy_refs=["arc3", "winning_trace"],
                    reality_refs=["arc3", self.game_id, "level_complete"],
                    meta_refs=[
                        f"levels_completed:{sum(1 for row in completion_rows if bool(row.get('is_level_complete', False)))}",
                    ],
                    confidence=1,
                    polarity=1,
                )
                galaxy_manager.store_meaning_star(
                    "Reality",
                    trace_star,
                    category="arc3_winning_trace",
                    metadata={
                        "source": "arc3_episode_galaxy",
                        "game_id": self.game_id,
                        "action_count": len(winning_actions),
                    },
                )
                session_entries += 1
        self.frames.clear()
        self.outcomes.clear()
        self._objects_by_color.clear()
        self._rules_by_key.clear()
        return {"rules_persisted": int(rules_persisted), "session_entries": int(session_entries)}

    def close(self) -> None:
        self._drain_pending_futures()
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["ARC3EpisodeGalaxy"]
