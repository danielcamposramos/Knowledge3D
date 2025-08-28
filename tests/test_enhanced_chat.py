import json

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from knowledge3d.bridge.enhanced_chat_processor import (
    EnhancedChatProcessor,
    ConversationContext,
)


def test_parse_navigation_goto():
    proc = EnhancedChatProcessor()
    ctx = ConversationContext()
    resp = proc.process_message("goto Mars", ctx)
    assert resp["type"] == "navigation"
    assert resp["action"] == "goto"
    assert resp["target"].lower() == "mars"


def test_parse_navigation_move_and_teleport():
    proc = EnhancedChatProcessor()
    ctx = ConversationContext()
    r1 = proc.process_message("move up 5", ctx)
    assert r1["action"] == "move" and r1["direction"] == "up" and r1["distance"] == 5
    r2 = proc.process_message("teleport to [1,2,3]", ctx)
    assert r2["action"] == "teleport" and tuple(r2["target"]) == (1.0, 2.0, 3.0)


def test_context_updates():
    proc = EnhancedChatProcessor()
    ctx = ConversationContext()
    resp = proc.process_message("goto lab", ctx)
    ctx.update("goto lab", resp)
    rel = ctx.get_relevant_context("where am i?")
    assert rel["last_action"] == "goto"
    assert len(rel["history_tail"]) == 1


def test_suggestions_for_unknown():
    proc = EnhancedChatProcessor()
    ctx = ConversationContext()
    resp = proc.process_message("fly sideways", ctx)
    assert resp["type"] == "chat_response"
    assert resp.get("suggestions")
