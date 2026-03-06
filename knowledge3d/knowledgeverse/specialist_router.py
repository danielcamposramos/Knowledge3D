"""Centralized specialist routing for Knowledgeverse query flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SpecialistRoute:
    """Resolved routing decision used by Cranium/Knowledgeverse query surfaces."""

    specialist: str
    domain: str
    galaxy_names: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "domain": self.domain,
            "galaxy_names": list(self.galaxy_names),
            "reason": self.reason,
        }


class SpecialistRouter:
    """Single source of truth for specialist + galaxy routing."""

    _ALWAYS_ON_AUXILIARY_GALAXIES: dict[str, list[str]] = {
        "visual": ["Tool"],
        "math": ["Tool"],
        "physics": ["Tool"],
        "audio": ["Tool"],
        "grammar": ["Tool"],
        "cartographer": ["Tool"],
        "any": ["Tool"],
    }

    _SPECIALIST_GALAXIES: dict[str, list[str]] = {
        "visual": ["Drawing", "Tool", "Grammar"],
        "math": ["Math", "Grammar"],
        "physics": ["Reality", "3DObjects", "Tool", "Math", "Grammar"],
        "audio": ["Audio", "Tool", "Drawing", "Grammar"],
        "grammar": ["Grammar", "Tool"],
        "cartographer": ["Math", "Reality", "3DObjects", "Tool", "Grammar", "Drawing"],
        "any": [],
    }

    _DOMAIN_SPECIALIST: dict[str, str] = {
        "visual": "visual",
        "math": "math",
        "physics": "physics",
        "audio": "audio",
        "signal": "audio",
        "logic": "grammar",
        "grammar": "grammar",
        "language": "grammar",
        "multi": "cartographer",
        "cartographer": "cartographer",
        "any": "any",
    }

    _VISUAL_HINTS = {
        "grid",
        "pattern",
        "transform",
        "flip",
        "rotate",
        "color",
        "shape",
        "pixel",
        "arc",
        "visual",
        "mesh",
        "voxel",
        "ray",
        "3d",
    }
    _MATH_HINTS = {
        "derivative",
        "integral",
        "equation",
        "solve",
        "compute",
        "calculate",
        "area",
        "factor",
        "algebra",
        "geometry",
        "calculus",
        "sum",
        "product",
    }
    _PHYSICS_HINTS = {
        "force",
        "mass",
        "energy",
        "velocity",
        "acceleration",
        "momentum",
        "newton",
        "physics",
        "gravity",
        "collision",
        "thermo",
        "electromagnetism",
        "field",
    }
    _AUDIO_HINTS = {
        "audio",
        "sound",
        "spectrogram",
        "waveform",
        "phoneme",
        "voice",
        "signal",
        "frequency",
        "music",
        "noise",
    }
    _LOGIC_HINTS = {
        "proof",
        "therefore",
        "implies",
        "logic",
        "syllogism",
        "predicate",
        "grammar",
        "syntax",
    }

    def __init__(self, specialist_bias: dict[str, float] | None = None):
        # Learned specialist biases persisted across runs.
        self._specialist_bias: dict[str, float] = {
            "visual": 0.0,
            "math": 0.0,
            "physics": 0.0,
            "audio": 0.0,
            "grammar": 0.0,
            "cartographer": 0.0,
            "any": 0.0,
        }
        if specialist_bias:
            for key, value in specialist_bias.items():
                if key in self._specialist_bias:
                    self._specialist_bias[key] = float(value)

    def route(
        self,
        query: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Resolve specialist and galaxy targets for a query."""
        explicit_galaxies = [str(name) for name in galaxy_names] if galaxy_names else None
        requested = (specialist or "auto").strip().lower()
        hint = (domain_hint or "").strip().lower()

        if requested != "auto":
            resolved_specialist = self._normalize_specialist(requested)
            resolved_domain = hint or resolved_specialist
            reason = "explicit_specialist"
        else:
            if hint:
                resolved_domain = hint
                resolved_specialist = self._specialist_from_domain(hint)
                reason = "domain_hint"
            else:
                resolved_domain = self._infer_domain(query)
                resolved_specialist = self._specialist_from_domain(resolved_domain)
                reason = "query_inference"

        base_galaxies = (
            explicit_galaxies
            if explicit_galaxies is not None
            else list(self._SPECIALIST_GALAXIES.get(resolved_specialist, ["Grammar"]))
        )
        resolved_galaxies = self._merge_auxiliary_galaxies(resolved_specialist, base_galaxies)

        return SpecialistRoute(
            specialist=resolved_specialist,
            domain=resolved_domain,
            galaxy_names=resolved_galaxies,
            reason=reason,
        ).as_dict()

    def _normalize_specialist(self, specialist: str) -> str:
        if specialist in self._SPECIALIST_GALAXIES:
            return specialist
        return "grammar"

    def _specialist_from_domain(self, domain: str) -> str:
        return self._DOMAIN_SPECIALIST.get(domain, "grammar")

    def _merge_auxiliary_galaxies(self, specialist: str, galaxies: Sequence[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for name in list(galaxies) + self._ALWAYS_ON_AUXILIARY_GALAXIES.get(specialist, []):
            token = str(name).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            merged.append(token)
        return merged

    def _infer_domain(self, query: str) -> str:
        lowered = query.lower()
        tokens = self._tokenize_query(lowered)

        domain_scores = {
            "visual": float(len(tokens & self._VISUAL_HINTS)),
            "math": float(len(tokens & self._MATH_HINTS)),
            "physics": float(len(tokens & self._PHYSICS_HINTS)),
            "audio": float(len(tokens & self._AUDIO_HINTS)),
            "logic": float(len(tokens & self._LOGIC_HINTS)),
        }
        # Numeric and symbolic cues provide extra evidence for math.
        if self._has_math_symbol_cue(lowered):
            domain_scores["math"] += 1.0

        # Inject learned specialist bias into domain scoring.
        for domain in ("visual", "math", "physics", "audio", "logic"):
            specialist = self._specialist_from_domain(domain)
            domain_scores[domain] += float(self._specialist_bias.get(specialist, 0.0))

        positives = [(domain, score) for domain, score in domain_scores.items() if score > 0.0]
        if not positives:
            return "grammar"

        positives.sort(key=lambda item: item[1], reverse=True)
        if len(positives) >= 2:
            top_score = positives[0][1]
            second_score = positives[1][1]
            # Multi-domain trigger when two domains are both strong.
            if second_score >= max(1.0, top_score * 0.75):
                return "multi"
        return positives[0][0]

    def _tokenize_query(self, lowered_query: str) -> set[str]:
        tokens: set[str] = set()
        current: list[str] = []
        for ch in lowered_query:
            if ch.isalnum() or ch == "_":
                current.append(ch)
                continue
            if current:
                tokens.add("".join(current))
                current = []
        if current:
            tokens.add("".join(current))
        return tokens

    def _has_math_symbol_cue(self, lowered_query: str) -> bool:
        for ch in lowered_query:
            if ch.isdigit() or ch in "+-*/=^":
                return True
        return False

    def set_specialist_bias(self, specialist: str, value: float) -> None:
        if specialist not in self._specialist_bias:
            return
        self._specialist_bias[specialist] = float(value)

    def adjust_specialist_bias(self, specialist: str, delta: float) -> None:
        if specialist not in self._specialist_bias:
            return
        updated = self._specialist_bias[specialist] + float(delta)
        # Keep bounded to avoid runaway routing collapse.
        self._specialist_bias[specialist] = max(-1.0, min(1.0, updated))

    def get_specialist_bias(self) -> dict[str, float]:
        return dict(self._specialist_bias)
