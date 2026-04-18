"""Persistent storage for lightweight TRM routing weights/topology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TRMWeightStore:
    """Read/write persistent TRM routing state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "specialist_bias": {},
                "routing_topology": {},
                "update_count": 0,
                "navigator_recent_traces": [],
                "navigator_training_state": {},
            }
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "version": 1,
                "specialist_bias": {},
                "routing_topology": {},
                "update_count": 0,
                "navigator_recent_traces": [],
                "navigator_training_state": {},
            }
        if not isinstance(raw, dict):
            return {
                "version": 1,
                "specialist_bias": {},
                "routing_topology": {},
                "update_count": 0,
                "navigator_recent_traces": [],
                "navigator_training_state": {},
            }
        return {
            "version": int(raw.get("version", 1)),
            "specialist_bias": dict(raw.get("specialist_bias", {})),
            "routing_topology": dict(raw.get("routing_topology", {})),
            "update_count": int(raw.get("update_count", 0)),
            "navigator_recent_traces": list(raw.get("navigator_recent_traces", [])),
            "navigator_training_state": dict(raw.get("navigator_training_state", {})),
        }

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)
