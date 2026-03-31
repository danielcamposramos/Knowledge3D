"""Run the K3D sovereign ARC-AGI-3 agent against the remote game API."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")
STORAGE_ROOT = Path("/K3D/Knowledge3D.local")
ACTIVE_STATES = {"IN_PROGRESS", "NOT_FINISHED"}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_live_log_path() -> Path:
    return LOG_ROOT / f"arc3_live_{_ts()}.jsonl"


def scorecard_url(api_url: str, card_id: str) -> str:
    base = str(api_url).rstrip("/")
    return f"{base}/scorecards/{card_id}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def level_progress(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def normalize_frame(frame: Any) -> list[list[int]]:
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool))

    def _extract_grid(value: Any) -> list[list[int]] | None:
        if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            if all(all(_is_scalar(cell) for cell in row) for row in value):
                return [[int(cell) for cell in row] for row in value]
            for item in value:
                extracted = _extract_grid(item)
                if extracted:
                    return extracted
        if isinstance(value, dict):
            for key in ("grid", "frame", "image", "board", "layers"):
                if key in value:
                    extracted = _extract_grid(value[key])
                    if extracted:
                        return extracted
            for item in value.values():
                extracted = _extract_grid(item)
                if extracted:
                    return extracted
        return None

    return _extract_grid(frame) or [[]]


def normalize_goal_frame(reset_response: Any) -> list[list[int]]:
    if not isinstance(reset_response, dict):
        return [[]]

    goal = normalize_frame(reset_response.get("goal", [[]]))
    if goal and goal != [[]]:
        return goal

    task_data = reset_response.get("task_data") or {}
    if not isinstance(task_data, dict):
        try:
            task_data = dict(task_data)
        except Exception:
            task_data = {}

    for split_name in ("train", "test"):
        split_rows = list(task_data.get(split_name) or [])
        for example in split_rows:
            if not isinstance(example, dict):
                continue
            candidate = normalize_frame(example.get("output", [[]]))
            if candidate and candidate != [[]]:
                return candidate
    return [[]]


def _clone_frame(frame: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in frame]


def run_live_arc3(
    *,
    game_id: str,
    max_actions: int,
    log_path: str | Path | None,
    api_url: str,
    storage_root: str | Path = STORAGE_ROOT,
) -> dict[str, Any]:
    from benchmarks.arc_agi_3 import K3DARC3Agent
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        secrets_path = Path("/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt")
        if secrets_path.exists():
            api_key = secrets_path.read_text().strip()
    if not api_key:
        raise RuntimeError("ARC_API_KEY not set and /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt not found")

    try:
        import requests
    except Exception as exc:  # pragma: no cover - environment-dependent runtime
        raise RuntimeError(f"requests not installed: {exc}") from exc

    resolved_log_path = Path(log_path) if log_path else default_live_log_path()
    session = requests.Session()
    session.headers.update(
        {
            "X-API-Key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    scorecard = session.post(
        f"{api_url}/api/scorecard/open",
        json={"game_ids": [game_id], "tags": ["k3d-sovereign"]},
        timeout=30,
    ).json()
    card_id = str(scorecard.get("card_id", ""))
    card_url = str(scorecard.get("scorecard_url") or scorecard.get("url") or scorecard_url(api_url, card_id))

    reset = session.post(
        f"{api_url}/api/cmd/RESET",
        json={"card_id": card_id, "game_id": game_id, "reasoning": "K3D init"},
        timeout=30,
    ).json()
    guid = reset.get("guid", "")
    frame = normalize_frame(reset.get("frame", [[]]))
    goal_frame = normalize_goal_frame(reset)
    task_data = reset.get("task_data") or {}
    available_actions = list(reset.get("available_actions") or [])
    state = reset.get("state", "IN_PROGRESS")

    kv = Knowledgeverse(storage_root=storage_root)
    agent = K3DARC3Agent(max_actions=max_actions, knowledgeverse=kv)
    action_count = 0
    start = time.time()
    levels_completed = level_progress(reset.get("levels_completed", 0))

    try:
        while state in ACTIVE_STATES and action_count < max_actions:
            action = agent.choose_action(
                frame,
                goal_frame=goal_frame,
                task_data=task_data,
                available_actions=available_actions,
                game_id=game_id,
                levels_completed=levels_completed,
            )
            payload: dict[str, Any] = {
                "game_id": game_id,
                "guid": guid,
                "reasoning": {
                    "agent": "k3d-sovereign",
                    "confidence": action["confidence"],
                    "converged": action["converged"],
                },
            }
            if action["action"] == "ACTION6":
                payload["x"] = int(action.get("x", max(0, len(frame[0]) // 2) if frame and frame[0] else 0))
                payload["y"] = int(action.get("y", max(0, len(frame) // 2)))
            if action["action"] == "RESET":
                response = session.post(
                    f"{api_url}/api/cmd/RESET",
                    json={
                        "card_id": card_id,
                        "game_id": game_id,
                        "reasoning": {
                            "agent": "k3d-sovereign",
                            "confidence": action["confidence"],
                            "converged": action["converged"],
                        },
                    },
                    timeout=30,
                ).json()
                goal_frame = normalize_goal_frame(response)
                task_data = response.get("task_data") or task_data
            else:
                response = session.post(
                    f"{api_url}/api/cmd/{action['action']}",
                    json=payload,
                    timeout=30,
                ).json()
            frame = normalize_frame(response.get("frame", frame))
            guid = response.get("guid", guid)
            state = response.get("state", state)
            available_actions = list(response.get("available_actions") or available_actions)
            levels_completed = level_progress(response.get("levels_completed", levels_completed))
            outcome_signal = agent.learn_from_outcome(
                levels_completed=levels_completed,
                frame=frame,
            )
            action_count += 1

            log_row = {
                "timestamp": time.time(),
                "game_id": game_id,
                "card_id": card_id,
                "card_url": card_url,
                "action_count": action_count,
                "action": action["action"],
                "action_index": action["action_index"],
                "label": action["label"],
                "confidence": action["confidence"],
                "converged": action["converged"],
                "outcome_signal": outcome_signal,
                "state": state,
                "levels_completed": levels_completed,
                "available_actions": list(available_actions),
                "x": action.get("x"),
                "y": action.get("y"),
                "click_reason": str(action.get("click_reason", "")),
                "frame_state": str(action.get("frame_state", "")),
                "fresh_context": bool(action.get("fresh_context", False)),
                "movement_budget": dict(action.get("movement_budget") or {}),
                "lives_remaining": action.get("lives_remaining"),
                "reference_box_visible": bool(action.get("reference_box_visible", False)),
                "flash_semantics": str(action.get("flash_semantics", "")),
                "target_label": str(action.get("target_label", "")),
                "attempt_actions": int(action.get("attempt_actions", 0)),
                "frame_unchanged": bool(action.get("frame_unchanged", False)),
                "blocked_actions": list(action.get("blocked_actions") or []),
                "program_type": str(action.get("task_result", {}).get("program_type", "")),
                "program_id": str(action.get("task_result", {}).get("program_id", "")),
                "result_answer_index": action.get("task_result", {}).get("answer_index"),
                "result_action_name": str(action.get("task_result", {}).get("action_name", "")),
                "result_error": str(action.get("task_result", {}).get("error", "")),
                "frame": _clone_frame(frame),
            }
            append_jsonl(resolved_log_path, log_row)
            print(
                f"[{action_count:03d}] {action['label']:12s} conf={action['confidence']:.3f} "
                f"levels={levels_completed} state={state}"
            )
            if state in {"WIN", "GAME_OVER"}:
                break
    finally:
        agent.close()
        session.post(f"{api_url}/api/scorecard/close", json={"card_id": card_id}, timeout=30)

    elapsed = time.time() - start
    return {
        "game_id": game_id,
        "card_id": card_id,
        "card_url": card_url,
        "state": state,
        "actions": action_count,
        "levels_completed": levels_completed,
        "elapsed_seconds": round(elapsed, 2),
        "log_path": str(resolved_log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="K3D ARC-AGI-3 sovereign agent")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--log-path", default=str(default_live_log_path()))
    parser.add_argument("--api-url", default="https://three.arcprize.org")
    parser.add_argument("--storage-root", default=str(STORAGE_ROOT))
    args = parser.parse_args()

    try:
        summary = run_live_arc3(
            game_id=args.game_id,
            max_actions=args.max_actions,
            log_path=args.log_path,
            api_url=args.api_url,
            storage_root=args.storage_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Done: {summary['actions']} actions in {summary['elapsed_seconds']:.1f}s state={summary['state']}")
    print(f"Log: {summary['log_path']}")
    print(f"Scorecard URL: {summary['card_url']}")
    if summary["state"] == "WIN":
        print("WIN!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
