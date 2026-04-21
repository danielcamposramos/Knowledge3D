"""Chat WINE — Tablet surface for multi-turn free-text conversation.

Mirrors math_wine.py / question_wine.py shape. Chat reads the House through
the internal `chat_specialist`; it does NOT emit stars (no ingest). If a turn
crosses into ingest (user uploads a doc), that is a PROCEDURALIZE handoff,
not a chat-side concern.

See: TEMP/CLAUDE_CHAT_WINE_SPEC_04.20.2026.md
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from knowledge3d.bridge.headless_tablet import (
    ROUTE_POLICY_ALL_LIVE_GALAXIES,
    SURFACE_KIND_CHAT,
    TabletEnvelope,
    TabletIngest,
)

# Default galaxies resident in VRAM during chat reasoning.
# Chat is broad — keep the full default set loaded. No selection, no capping.
# (See feedback_no_knowledge_caps.md — quantity caps NEVER; LOD + frustum cull
# handle the working-memory management on GPU.)
CHAT_ROUTE_GALAXIES: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
)

# Maximum bytes per turn content field (UTF-8 encoded). Input gate.
CHAT_MAX_CONTENT_BYTES: int = 16 * 1024  # 16 KiB per message — tune later
CHAT_MAX_MESSAGES: int = 64              # per-turn history cap — tune later

_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "system"})


def _validate_chat_input(
    messages: Any,
    context: Any,
) -> dict[str, Any] | None:
    """Validate chat input. Returns error dict on failure, None on success.

    Pure Python type/length checks — NOT reasoning logic.
    Per feedback_python_dispatch_is_not_a_line_item.md: these are I/O gates.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return {"status": "error", "error": "empty_messages"}
    if len(messages) > CHAT_MAX_MESSAGES:
        return {"status": "error", "error": "too_many_messages"}
    for msg in messages:
        if not isinstance(msg, dict):
            return {"status": "error", "error": "bad_message_shape"}
        role = str(msg.get("role", "")).strip().lower()
        if role not in _VALID_ROLES:
            return {"status": "error", "error": "bad_message_shape"}
        content = msg.get("content")
        if not isinstance(content, str):
            return {"status": "error", "error": "bad_message_shape"}
        try:
            encoded = content.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return {"status": "error", "error": "bad_encoding"}
        if len(encoded) > CHAT_MAX_CONTENT_BYTES:
            return {"status": "error", "error": "content_too_large"}
    return None


def build_chat_route(
    *,
    specialist: str = "chat",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]:
    """Return the galaxy route descriptor consumed by the chat specialist.

    Symmetric to math_wine.build_math_route(). Contains the resident galaxy
    set plus any route hints the specialist needs. No semantic gravity
    tuning here — that lives inside the specialist on GPU.
    """
    route: dict[str, Any] = {
        "specialist": str(specialist or "chat"),
        "route_policy": str(route_policy or ROUTE_POLICY_ALL_LIVE_GALAXIES),
    }
    if domain_hint is not None and str(domain_hint).strip():
        route["domain_hint"] = str(domain_hint).strip()
    galaxy_names = [str(name) for name in (galaxies or CHAT_ROUTE_GALAXIES) if str(name).strip()]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_chat_task(
    messages: Sequence[Mapping[str, str]],
    *,
    context: Mapping[str, Any] | None = None,
    stream: bool = False,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (task_payload, route_payload) for a single chat turn.

    Args:
        messages: Sequence of {"role": "user"|"assistant"|"system",
                  "content": str}. Client sends full history each turn
                  (stateless server).
        context: Optional dict of prior-turn references.
        stream: If True, the specialist MAY emit partial envelopes (MVP
                treats all turns as non-streaming; flag reserved).
    """
    envelope = TabletIngest.chat_task(
        messages,
        context=context,
        stream=stream,
    )
    return dict(envelope.task), build_chat_route(
        specialist=envelope.specialist,
        galaxies=list(galaxies or CHAT_ROUTE_GALAXIES),
        route_policy=route_policy,
    )


def chat_envelope(
    messages: Sequence[Mapping[str, str]],
    *,
    context: Mapping[str, Any] | None = None,
    stream: bool = False,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return the full Tablet envelope dict for a chat turn.

    This is the factory TabletIngest.chat_task delegates to. Shape
    mirrors math_wine structure — returns a plain dict matching the
    TabletEnvelope wire format expected by the daemon.
    """
    msg_list = list(messages)
    # Derive the prompt text from the last user message.
    prompt = ""
    for msg in reversed(msg_list):
        if isinstance(msg, dict) and str(msg.get("role", "")).strip().lower() == "user":
            candidate = str(msg.get("content", "")).strip()
            if candidate:
                prompt = candidate
                break

    task_payload: dict[str, Any] = {
        "surface_kind": SURFACE_KIND_CHAT,
        "task_id": str(task_id or ""),
        "query": prompt,
        "prompt": prompt,
        "messages": msg_list,
        "context": dict(context) if context else {},
        "stream": bool(stream),
    }
    return {
        "surface_kind": SURFACE_KIND_CHAT,
        "task_id": task_id,
        "task": task_payload,
        "route": build_chat_route(),
    }
