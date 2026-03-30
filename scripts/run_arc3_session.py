"""K3D ARC-AGI-3 living session with one brain across many games."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_arc3_agent import (
    ACTIVE_STATES,
    _clone_frame,
    append_jsonl,
    level_progress,
    normalize_frame,
    normalize_goal_frame,
    scorecard_url,
)


LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")
STORAGE_ROOT = Path("/K3D/Knowledge3D.local")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _session_log_dir(log_dir: Path | None = None) -> Path:
    return Path(log_dir) if log_dir is not None else (LOG_ROOT / f"arc3_session_{_ts()}")


def _inter_game_consolidation(
    knowledgeverse,
    game_result: dict[str, Any],
) -> list[dict[str, Any]]:
    if str(game_result.get("state") or "").upper() == "WIN":
        outcome = "win"
    elif int(game_result.get("levels_completed", 0)) > 0:
        outcome = "progress"
    else:
        outcome = "stalled"
    results: list[dict[str, Any]] = []
    for _ in range(3):
        if hasattr(knowledgeverse, "jarvis_sleep_consolidation"):
            summary = dict(knowledgeverse.jarvis_sleep_consolidation() or {})
            summary.setdefault("session_outcome", outcome)
            results.append(summary)
        else:
            results.append({"updated": False, "session_outcome": outcome})
    return results


def run_single_game(
    agent,
    *,
    game_id: str,
    api_url: str,
    log_dir: Path,
) -> dict[str, Any]:
    try:
        import os
        import requests
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"requests not installed: {exc}") from exc

    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        secrets_path = Path("/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt")
        if secrets_path.exists():
            api_key = secrets_path.read_text().strip()
    if not api_key:
        raise RuntimeError("ARC_API_KEY not set and /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt not found")

    log_dir.mkdir(parents=True, exist_ok=True)
    game_log_path = log_dir / f"{game_id}.jsonl"
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
        json={"game_ids": [game_id], "tags": ["k3d-living-session"]},
        timeout=30,
    ).json()
    card_id = str(scorecard.get("card_id", ""))
    card_url = str(scorecard.get("scorecard_url") or scorecard.get("url") or scorecard_url(api_url, card_id))

    reset = session.post(
        f"{api_url}/api/cmd/RESET",
        json={"card_id": card_id, "game_id": game_id, "reasoning": "K3D living session init"},
        timeout=30,
    ).json()
    guid = reset.get("guid", "")
    frame = normalize_frame(reset.get("frame", [[]]))
    goal_frame = normalize_goal_frame(reset)
    task_data = reset.get("task_data") or {}
    available_actions = list(reset.get("available_actions") or [])
    state = reset.get("state", "IN_PROGRESS")
    levels_completed = level_progress(reset.get("levels_completed", 0))
    action_count = 0

    try:
        while state in ACTIVE_STATES and action_count < int(agent.max_actions):
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
                    "agent": "k3d-living-session",
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
                            "agent": "k3d-living-session",
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

            append_jsonl(
                game_log_path,
                {
                    "timestamp": datetime.now().timestamp(),
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
                    "program_type": str(action.get("task_result", {}).get("program_type", "")),
                    "program_id": str(action.get("task_result", {}).get("program_id", "")),
                    "result_answer_index": action.get("task_result", {}).get("answer_index"),
                    "result_action_name": str(action.get("task_result", {}).get("action_name", "")),
                    "result_error": str(action.get("task_result", {}).get("error", "")),
                    "frame": _clone_frame(frame),
                },
            )
            if state in {"WIN", "GAME_OVER"}:
                break
    finally:
        session.post(f"{api_url}/api/scorecard/close", json={"card_id": card_id}, timeout=30)
        session.close()

    return {
        "game_id": game_id,
        "card_id": card_id,
        "card_url": card_url,
        "state": state,
        "actions": action_count,
        "levels_completed": levels_completed,
        "log_path": str(game_log_path),
    }


def run_arc3_session(
    *,
    game_ids: list[str],
    max_actions_per_game: int = 500,
    api_url: str = "https://three.arcprize.org",
    log_dir: Path | None = None,
    storage_root: str | Path = STORAGE_ROOT,
) -> dict[str, Any]:
    from benchmarks.arc_agi_3 import K3DARC3Agent
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

    resolved_log_dir = _session_log_dir(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    kv = Knowledgeverse(storage_root=storage_root)

    results: list[dict[str, Any]] = []
    for game_id in [str(value).strip() for value in game_ids if str(value).strip()]:
        agent = K3DARC3Agent(
            max_actions=max_actions_per_game,
            log_path=resolved_log_dir / f"{game_id}_agent.jsonl",
            knowledgeverse=kv,
        )
        try:
            result = run_single_game(agent, game_id=game_id, api_url=api_url, log_dir=resolved_log_dir)
        finally:
            agent.close()
        result["inter_game_consolidation"] = _inter_game_consolidation(kv, result)
        results.append(result)

    summary = {
        "games": results,
        "log_dir": str(resolved_log_dir),
    }
    (resolved_log_dir / "session_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="K3D ARC-AGI-3 living session")
    parser.add_argument("--game-id", dest="game_ids", action="append", required=True)
    parser.add_argument("--max-actions-per-game", type=int, default=500)
    parser.add_argument("--api-url", default="https://three.arcprize.org")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--storage-root", default=str(STORAGE_ROOT))
    args = parser.parse_args()

    summary = run_arc3_session(
        game_ids=list(args.game_ids or []),
        max_actions_per_game=args.max_actions_per_game,
        api_url=args.api_url,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        storage_root=args.storage_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
