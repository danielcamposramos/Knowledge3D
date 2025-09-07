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


@dataclass
class CraniumConfig:
    llm_model: Optional[str] = None  # HF model path; falls back to default in LLMSkill


class CraniumCore:
    def __init__(self, cfg: Optional[CraniumConfig] = None) -> None:
        self.cfg = cfg or CraniumConfig()
        # Lazy import skills to avoid heavyweight deps on import
        self._llm = None
        self._parser = None

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

