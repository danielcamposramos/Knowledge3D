from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


SPATIAL_COMMANDS: Dict[str, Dict[str, str]] = {
    "navigation": {
        "goto [location]": "Navigate to a specific location in the 3D space",
        "move [direction] [distance]": "Move in a direction (up, down, left, right, forward, back) by specified units",
        "teleport to [x,y,z]": "Instantly move to exact coordinates",
        "follow [user/object]": "Follow a specific user or object",
        "orbit around [object]": "Orbit around a specified object",
    },
    "exploration": {
        "show me [concept/topic]": "Display visual representation of a concept",
        "find related to [concept]": "Find and display related concepts",
        "expand [area]": "Expand a specific area to show more detail",
        "hide [object]": "Hide a specific object from view",
    },
    "interaction": {
        "touch [object]": "Simulate touching an object to reveal properties",
        "talk to [avatar]": "Initiate conversation with another avatar",
        "give [object] to [recipient]": "Transfer an object to another entity",
    },
}


@dataclass
class ConversationContext:
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_location: Optional[Tuple[float, float, float]] = None
    last_action: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    def update(self, message: str, response: Dict[str, Any]) -> None:
        self.last_action = response.get("action") or response.get("type")
        self.history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "message": message,
                "response": response,
                "location": self.current_location,
            }
        )

    def get_relevant_context(self, current_message: str) -> Dict[str, Any]:
        return {
            "last_action": self.last_action,
            "current_location": self.current_location,
            "history_tail": self.history[-3:],
        }


class SpatialParser:
    loc_re = re.compile(r"\b(?:goto|go to)\s+(?P<loc>[\w\- ]+)", re.I)
    teleport_re = re.compile(r"\bteleport(?:\s+to)?\s*\[(?P<x>-?\d+(?:\.\d+)?),(?P<y>-?\d+(?:\.\d+)?),(?P<z>-?\d+(?:\.\d+)?)\]", re.I)
    move_re = re.compile(r"\bmove\s+(?P<dir>up|down|left|right|forward|back)\s+(?P<dist>\d+(?:\.\d+)?)", re.I)
    follow_re = re.compile(r"\bfollow\s+(?P<target>[\w\- ]+)", re.I)
    orbit_re = re.compile(r"\borbit\s+around\s+(?P<target>[\w\- ]+)", re.I)
    show_re = re.compile(r"\bshow\s+me\s+(?P<topic>.+)", re.I)
    find_re = re.compile(r"\bfind\s+related\s+to\s+(?P<topic>.+)", re.I)
    expand_re = re.compile(r"\bexpand\s+(?P<area>.+)", re.I)
    hide_re = re.compile(r"\bhide\s+(?P<obj>.+)", re.I)
    touch_re = re.compile(r"\btouch\s+(?P<obj>.+)", re.I)
    talk_re = re.compile(r"\btalk\s+to\s+(?P<avatar>.+)", re.I)
    give_re = re.compile(r"\bgive\s+(?P<obj>.+)\s+to\s+(?P<recipient>.+)", re.I)

    def parse(self, message: str, ctx: ConversationContext) -> Dict[str, Any]:
        m = self.teleport_re.search(message)
        if m:
            return {
                "intent": "navigation",
                "action": "teleport",
                "coords": (float(m.group("x")), float(m.group("y")), float(m.group("z"))),
            }
        m = self.move_re.search(message)
        if m:
            return {
                "intent": "navigation",
                "action": "move",
                "direction": m.group("dir").lower(),
                "distance": float(m.group("dist")),
            }
        m = self.loc_re.search(message)
        if m:
            return {"intent": "navigation", "action": "goto", "location": m.group("loc").strip()}
        m = self.follow_re.search(message)
        if m:
            return {"intent": "navigation", "action": "follow", "target": m.group("target").strip()}
        m = self.orbit_re.search(message)
        if m:
            return {"intent": "navigation", "action": "orbit", "target": m.group("target").strip()}
        m = self.show_re.search(message)
        if m:
            return {"intent": "exploration", "action": "show", "topic": m.group("topic").strip()}
        m = self.find_re.search(message)
        if m:
            return {"intent": "exploration", "action": "find_related", "topic": m.group("topic").strip()}
        m = self.expand_re.search(message)
        if m:
            return {"intent": "exploration", "action": "expand", "area": m.group("area").strip()}
        m = self.hide_re.search(message)
        if m:
            return {"intent": "exploration", "action": "hide", "object": m.group("obj").strip()}
        m = self.touch_re.search(message)
        if m:
            return {"intent": "interaction", "action": "touch", "object": m.group("obj").strip()}
        m = self.talk_re.search(message)
        if m:
            return {"intent": "interaction", "action": "talk", "avatar": m.group("avatar").strip()}
        m = self.give_re.search(message)
        if m:
            return {
                "intent": "interaction",
                "action": "give",
                "object": m.group("obj").strip(),
                "recipient": m.group("recipient").strip(),
            }
        return {"intent": "unknown", "action": "unknown"}


class EnhancedChatProcessor:
    def __init__(self) -> None:
        self.spatial_parser = SpatialParser()

    def suggestions(self, text: str, limit: int = 5) -> List[str]:
        keys: List[str] = []
        for cat in SPATIAL_COMMANDS.values():
            keys.extend(cat.keys())
        t = text.lower().strip()
        ranked = sorted(keys, key=lambda k: self._simple_distance(t, k.lower()))
        return ranked[:limit]

    @staticmethod
    def _simple_distance(a: str, b: str) -> int:
        # very rough: shorter edit-like distance by token mismatch
        as_ = a.split()
        bs_ = b.split()
        d = 0
        for i in range(min(len(as_), len(bs_))):
            if as_[i] != bs_[i]:
                d += 1
        d += abs(len(as_) - len(bs_))
        return d

    def process_message(self, message: str, ctx: ConversationContext) -> Dict[str, Any]:
        parsed = self.spatial_parser.parse(message, ctx)
        intent = parsed.get("intent")
        action = parsed.get("action")

        if intent == "navigation":
            return self._handle_navigation(parsed, ctx)
        if intent == "exploration":
            return self._handle_exploration(parsed, ctx)
        if intent == "interaction":
            return self._handle_interaction(parsed, ctx)

        return {
            "type": "chat_response",
            "ok": False,
            "message": "I didn't understand that. Try one of these:",
            "suggestions": self.suggestions(message),
        }

    def _handle_navigation(self, p: Dict[str, Any], ctx: ConversationContext) -> Dict[str, Any]:
        act = p.get("action")
        if act == "teleport":
            coords = p.get("coords")
            if isinstance(coords, tuple) and len(coords) == 3:
                ctx.current_location = coords
            return {"type": "navigation", "action": "teleport", "target": coords, "message": f"Teleporting to {coords}"}
        if act == "move":
            return {
                "type": "navigation",
                "action": "move",
                "direction": p.get("direction"),
                "distance": p.get("distance"),
                "message": f"Moving {p.get('direction')} by {p.get('distance')} units",
            }
        if act == "goto":
            return {"type": "navigation", "action": "goto", "target": p.get("location"), "message": f"Navigating to {p.get('location')}"}
        if act == "follow":
            return {"type": "navigation", "action": "follow", "target": p.get("target"), "message": f"Following {p.get('target')}"}
        if act == "orbit":
            return {"type": "navigation", "action": "orbit", "target": p.get("target"), "message": f"Orbiting around {p.get('target')}"}
        return {"type": "chat_response", "ok": False, "message": "Unknown navigation action"}

    def _handle_exploration(self, p: Dict[str, Any], ctx: ConversationContext) -> Dict[str, Any]:
        act = p.get("action")
        if act == "show":
            return {"type": "exploration", "action": "show", "topic": p.get("topic"), "message": f"Showing {p.get('topic')}"}
        if act == "find_related":
            return {"type": "exploration", "action": "find_related", "topic": p.get("topic"), "message": f"Finding related to {p.get('topic')}"}
        if act == "expand":
            return {"type": "exploration", "action": "expand", "area": p.get("area"), "message": f"Expanding {p.get('area')}"}
        if act == "hide":
            return {"type": "exploration", "action": "hide", "object": p.get("object"), "message": f"Hiding {p.get('object')}"}
        return {"type": "chat_response", "ok": False, "message": "Unknown exploration action"}

    def _handle_interaction(self, p: Dict[str, Any], ctx: ConversationContext) -> Dict[str, Any]:
        act = p.get("action")
        if act == "touch":
            return {"type": "interaction", "action": "touch", "object": p.get("object"), "message": f"Touching {p.get('object')}"}
        if act == "talk":
            return {"type": "interaction", "action": "talk", "avatar": p.get("avatar"), "message": f"Talking to {p.get('avatar')}"}
        if act == "give":
            return {
                "type": "interaction",
                "action": "give",
                "object": p.get("object"),
                "recipient": p.get("recipient"),
                "message": f"Giving {p.get('object')} to {p.get('recipient')}",
            }
        return {"type": "chat_response", "ok": False, "message": "Unknown interaction action"}


