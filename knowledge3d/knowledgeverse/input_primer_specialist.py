"""Input primer specialist: pre-routing normalization for chat and MCQ inputs.

This specialist does not perform routing. It only normalizes user-facing text
into deterministic forms before downstream TRM routing/selection.
"""

from __future__ import annotations

import re
from typing import Any

from knowledge3d.knowledgeverse.specialist_base import SpecialistBase


_WS_RE = re.compile(r"\s+")
_OPT_LABEL_RE = re.compile(r"^\s*[\(\[]?\s*([A-Da-d]|[0-9]{1,2})\s*[\)\].:\-]\s*")


class InputPrimerSpecialist(SpecialistBase):
    """
    Lightweight text normalization stage before routing.

    Guarantees:
    - Stable whitespace and punctuation normalization.
    - Option text normalization for multiple-choice benchmarks.
    - No routing decisions (router remains TRM/Navigator).
    """

    def __init__(self, *, parent: SpecialistBase | None = None, **kwargs: Any):
        super().__init__(
            name="InputPrimerSpecialist",
            domain="input_normalization",
            parent=parent,
            **kwargs,
        )

    def normalize_chat_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower() or "user"
            content = self.normalize_text(str(msg.get("content", "")))
            out.append({"role": role, "content": content})
        return out

    def prepare_multiple_choice(self, question_text: str, options: list[str]) -> dict[str, Any]:
        normalized_question = self.normalize_text(question_text)
        normalized_options = [self._normalize_option_text(opt) for opt in options]

        # Build a deterministic option context block to improve specialist matching.
        option_lines = []
        for idx, option in enumerate(normalized_options):
            label = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
            option_lines.append(f"({label}) {option}")
        normalized_prompt = (
            f"{normalized_question}\n"
            f"Options:\n" + "\n".join(option_lines)
        ).strip()

        return {
            "question_text": normalized_prompt,
            "options": normalized_options,
            "original_options": list(options),
        }

    def normalize_text(self, text: str) -> str:
        text = str(text)
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = _WS_RE.sub(" ", text).strip()
        return text

    def _normalize_option_text(self, option: str) -> str:
        cleaned = self.normalize_text(option)
        cleaned = _OPT_LABEL_RE.sub("", cleaned)
        return cleaned.strip()

