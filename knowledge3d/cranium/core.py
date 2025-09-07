from __future__ import annotations

"""
CraniumCore — unified logic facade

Presents a single, integrated interface for K3D's logic:
- ask(): LLM generation (Transformers), optionally with RAG
- parse(): map text -> structured intent via EnhancedChatProcessor
- act(): run RPN‑gated actions and return a chat response payload

This class does not force a "router" mental model; it composes skills
behind one cohesive API so upper layers can treat the agent as one brain.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


try:  # Optional import; keep core lightweight
    from .memory import ShortTermGalaxy  # type: ignore
except Exception:  # pragma: no cover
    ShortTermGalaxy = None  # type: ignore

try:
    from ..core.faith_engine import FaithEngine  # type: ignore
except Exception:  # pragma: no cover
    FaithEngine = None  # type: ignore


@dataclass
class CraniumConfig:
    llm_model: Optional[str] = None  # HF model path; falls back to default in LLMSkill


class CraniumCore:
    def __init__(self, cfg: Optional[CraniumConfig] = None) -> None:
        self.cfg = cfg or CraniumConfig()
        # Lazy import skills to avoid heavyweight deps on import
        self._llm = None
        self._parser = None
        self._stm = ShortTermGalaxy() if ShortTermGalaxy else None
        self._faith = FaithEngine() if FaithEngine else None

    def init_llm(self) -> None:
        if self._llm is not None:
            return
        try:
            from ..skills.llm import LLMSkill, LLMConfig  # type: ignore
            llm_cfg = LLMConfig()
            if self.cfg.llm_model:
                llm_cfg.model = self.cfg.llm_model
            self._llm = LLMSkill(llm_cfg)
        except Exception:
            self._llm = None

    def init_parser(self) -> None:
        if self._parser is not None:
            return
        try:
            from ..bridge.enhanced_chat_processor import EnhancedChatProcessor  # type: ignore
            self._parser = EnhancedChatProcessor()
        except Exception:
            self._parser = None

    # --- Unified calls ---
    def ask(self, question: str, context: Optional[List[Tuple[str, str]]] = None, max_tokens: int = 384) -> str:
        self.init_llm()
        if self._llm is None:
            return "[llm unavailable]"
        if context:
            return self._llm.answer_with_rag(question, context, max_tokens=max_tokens)
        return self._llm.generate(question, system=None, max_tokens=max_tokens)

    def parse(self, text: str) -> Dict[str, Any]:
        self.init_parser()
        if self._parser is None:
            return {"intent": "unknown", "action": "unknown"}
        # Minimal context for now; live_server holds the full ConversationContext
        try:
            return self._parser.process_message(text, type("Ctx", (), {"update": lambda *a, **k: None, "get_relevant_context": lambda *a, **k: {}})())
        except Exception:
            return {"intent": "unknown", "action": "unknown"}

    # --- Multimodal observe ---
    def observe_text(self, text: str, label: Optional[str] = None) -> None:
        if self._stm:
            try:
                self._stm.add_text(text, label=label)
            except Exception:
                pass

    def observe_image(self, image_path: str, label: Optional[str] = None) -> None:
        if self._stm:
            try:
                self._stm.add_image(image_path, label=label)
            except Exception:
                pass

    def observe_audio(self, audio_path: str, label: Optional[str] = None) -> None:
        if self._stm:
            try:
                self._stm.add_audio(audio_path, label=label)
            except Exception:
                pass

    # --- Acting and reflection ---
    def act(self, message: str, contexts: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """Unified action: parse intent, decide, and produce a response payload.

        - Uses EnhancedChatProcessor for spatial intents when available
        - Uses FaithEngine (if available) to gate navigation vs. chat
        - Uses short‑term memory contexts for chat; falls back to LLM skill
        """
        parsed = self.parse(message)
        intent = parsed.get("intent")
        action = parsed.get("action")

        # Confidence heuristic: prefer explicit navigation verbs
        nav_conf = 0.0
        if intent == "navigation":
            nav_conf = 0.85
        elif intent in ("exploration", "interaction"):
            nav_conf = 0.6
        chat_conf = 0.75 if intent not in ("navigation",) else 0.15

        choice = None
        if self._faith:
            choice = self._faith.decide({"navigate": nav_conf, "chat": chat_conf}, threshold=0.7)
        # Default to chat when undecided
        if choice is None:
            choice = "navigate" if nav_conf >= 0.7 else "chat"

        if choice == "navigate" and intent in {"navigation", "exploration", "interaction"}:
            # Return parsed as an action payload; caller (bridge) will execute
            return {"type": parsed.get("intent"), **parsed}

        # Otherwise, answer in chat using spatial memory first
        reply = ""
        # Prefer STM contexts if present; else use provided contexts; else empty
        rag_contexts: List[Tuple[str, str]] = []
        if self._stm:
            try:
                rag_contexts = self._stm.get_contexts(message, k=6)
            except Exception:
                rag_contexts = []
        if not rag_contexts and contexts:
            rag_contexts = contexts

        # Try K3D-native spatial text first
        try:
            from ..skills.spatial_text import compose_answer  # type: ignore
            if rag_contexts:
                reply = compose_answer(message, rag_contexts)
        except Exception:
            reply = ""

        if not reply:
            if rag_contexts:
                # Minimal composition when spatial_text is unavailable
                lines = [f"- {lab}: {txt[:160]}" for lab, txt in rag_contexts[:4]]
                reply = ("I will answer from nearby memory:\n" + "\n".join(lines))
            else:
                reply = (
                    "I prefer to ground answers in my House memory. "
                    "Let’s explore a few topics first, then ask again."
                )

        return {"type": "chat_response", "ok": True, "message": reply}

    def reflect(self) -> str:
        """Return a concise self-reflection summary based on STM contents."""
        if not self._stm:
            return "I am present but my short-term galaxy is disabled."
        try:
            return self._stm.reflect()
        except Exception:
            return "I attempted to reflect but encountered an internal issue."

    def sleep_consolidate(self, out_gltf: Optional[str] = None) -> str:
        """Consolidate STM into the House GLTF as durable objects.

        Returns a small status string; out_gltf defaults to viewer/public/memory_house.gltf
        """
        if not self._stm:
            return "sleep: no STM available"
        try:
            return self._stm.consolidate(out_gltf=out_gltf)
        except Exception as e:
            return f"sleep: consolidation error: {e}"
