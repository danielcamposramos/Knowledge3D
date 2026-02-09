"""Optional Ollama-backed teacher augmentation for curriculum generation."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class OllamaAugmenter:
    """
    Produce additional training perspectives from local Ollama models.

    The augmenter is intentionally optional. If Ollama or models are unavailable,
    deterministic fallbacks are used so training remains reproducible.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        vision_model: str = "llava",
        language_model: str = "llama3.2",
        multimodal_model: str = "llava",
        timeout_s: int = 30,
    ):
        self.enabled = bool(enabled)
        self.vision_model = vision_model
        self.language_model = language_model
        self.multimodal_model = multimodal_model
        self.timeout_s = int(timeout_s)

    def augment_aliases(self, operation: str, aliases: list[str]) -> list[str]:
        """Expand alias prompts with teacher-generated variations."""
        base = [str(a) for a in aliases if str(a).strip()]
        if not self.enabled:
            return self._deterministic_alias_expansion(operation, base)

        prompt = (
            "Generate 6 short rephrasings for this operation description as a JSON array of strings. "
            f"operation={operation}; aliases={base}"
        )
        teacher = self._ollama_generate(self.language_model, prompt)
        if teacher:
            parsed = self._parse_json_array(teacher)
            if parsed:
                merged = base + parsed
                return self._dedupe(merged)
        return self._deterministic_alias_expansion(operation, base)

    def augment_visual_examples(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate teacher perspectives for visual transformation examples."""
        if not self.enabled:
            return self._fallback_visual_augmentation(examples)

        compact = self._compact_examples(examples)
        visual_prompt = (
            "Infer the visual transformation rule from these input/output examples. "
            "Return one concise sentence.\n"
            f"{compact}"
        )
        language_prompt = (
            "Provide 5 alternate phrasings for this visual rule as a JSON array.\n"
            f"{compact}"
        )
        procedural_prompt = (
            "Propose a compact procedural pseudo-RPN transform for this rule. "
            "Return only one line.\n"
            f"{compact}"
        )
        visual_description = self._ollama_generate(self.vision_model, visual_prompt) or "visual transform rule"
        phrasing_raw = self._ollama_generate(self.language_model, language_prompt) or "[]"
        procedural_program = self._ollama_generate(self.multimodal_model, procedural_prompt) or "GRID CLONE"
        variations = self._parse_json_array(phrasing_raw) or []
        if not variations:
            variations = ["apply the inferred transformation"]
        return {
            "visual_description": visual_description.strip(),
            "language_variations": variations,
            "procedural_program": procedural_program.strip(),
            "teacher_source": "ollama",
        }

    def _ollama_generate(self, model: str, prompt: str) -> str | None:
        try:
            proc = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        text = (proc.stdout or "").strip()
        return text or None

    def _deterministic_alias_expansion(self, operation: str, aliases: list[str]) -> list[str]:
        op = str(operation).lower().replace("_", " ")
        deterministic = [
            f"perform {op} on the grid",
            f"apply {op} transformation",
            f"execute {op} operation",
        ]
        return self._dedupe(aliases + deterministic)

    def _fallback_visual_augmentation(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        compact = self._compact_examples(examples)
        return {
            "visual_description": f"inferred transform from examples: {compact[:120]}",
            "language_variations": [
                "apply the same transformation as the examples",
                "infer the visual rule and transform the input",
            ],
            "procedural_program": "GRID TRANSFORM_FROM_EXAMPLES",
            "teacher_source": "deterministic_fallback",
        }

    def _compact_examples(self, examples: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for idx, ex in enumerate(examples[:3]):
            inp = ex.get("input")
            out = ex.get("output")
            parts.append(f"E{idx+1}: in={inp} out={out}")
        return " | ".join(parts)

    def _parse_json_array(self, raw: str) -> list[str] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        out = [str(item).strip() for item in parsed if str(item).strip()]
        return out or None

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value.strip())
        return out

