#!/usr/bin/env python3
"""Standalone ARC-3 API diagnostic for remote compatibility negotiation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests


API_URL = "https://three.arcprize.org"
GAME_ID = "ls20"


def _resolve_api_key() -> str:
    key = os.environ.get("ARC_API_KEY", "").strip()
    if key:
        return key
    path = Path("/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    api_key = _resolve_api_key()
    if not api_key:
        print("WARNING: ARC_API_KEY not set and secrets file missing")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
        }
    )
    if api_key:
        session.headers["X-API-Key"] = api_key
        print(f"Using API key: {api_key[:8]}...")

    print("\n=== 1. Resolve full game_id ===")
    response = session.get(f"{API_URL}/api/games", timeout=30)
    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    games = list(response.json() or [])
    resolved_game_id = GAME_ID
    for game in games:
        if not isinstance(game, dict):
            continue
        candidate = str(game.get("game_id", "")).strip()
        if candidate == GAME_ID or candidate.startswith(f"{GAME_ID}-"):
            resolved_game_id = candidate
            print(json.dumps(game, indent=2))
            break
    print(f"resolved_game_id: {resolved_game_id}")

    print("\n=== 2. Opening scorecard ===")
    response = session.post(
        f"{API_URL}/api/scorecard/open",
        json={"tags": ["k3d-diagnostic"]},
        timeout=30,
    )
    print(f"HTTP {response.status_code}")
    scorecard = response.json()
    print(json.dumps(scorecard, indent=2))
    card_id = str(scorecard.get("card_id", ""))

    print("\n=== 3. RESET ===")
    response = session.post(
        f"{API_URL}/api/cmd/RESET",
        json={"card_id": card_id, "game_id": resolved_game_id},
        timeout=30,
    )
    print(f"HTTP {response.status_code}")
    reset_payload = response.json()
    diag_payload = {key: value for key, value in reset_payload.items() if key != "frame"}
    print(json.dumps(diag_payload, indent=2))
    print(f"frame: {type(reset_payload.get('frame')).__name__}")
    if isinstance(reset_payload.get("frame"), list):
        frame = reset_payload["frame"]
        print(f"frame shape: {len(frame)} rows × {len(frame[0]) if frame else 0} cols")
    guid = str(reset_payload.get("guid", ""))

    print("\n=== 4. ACTION3 ===")
    payload = {"game_id": resolved_game_id}
    if guid:
        payload["guid"] = guid
    response = session.post(
        f"{API_URL}/api/cmd/ACTION3",
        json=payload,
        timeout=30,
    )
    print(f"/api/cmd/ACTION3 → HTTP {response.status_code}")
    if response.status_code >= 400:
        print(response.text[:500])
    else:
        step_payload = response.json()
        print(json.dumps(step_payload, indent=2))
        action_input = step_payload.get("action_input", {})
        action_id = action_input.get("id", -1) if isinstance(action_input, dict) else -1
        print(f"action_input.id: {action_id}")
        guid = str(step_payload.get("guid", guid))

    print("\n=== 5. Close scorecard ===")
    session.post(
        f"{API_URL}/api/scorecard/close",
        json={"card_id": card_id},
        timeout=30,
    )
    print("Done")


if __name__ == "__main__":
    main()
