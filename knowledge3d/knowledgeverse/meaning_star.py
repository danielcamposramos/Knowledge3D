"""Meaning-centric star schema for House/Galaxy knowledge storage."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping


TRIT_VALUES = {-1, 0, 1}
EmbeddingVector = tuple[float, ...]


def compute_star_id(
    meaning_rpn: str,
    meaning_class: str,
    domain: str,
    galaxy_ref: str = "",
) -> str:
    """Content-addressed identity: same concept definition yields the same star id."""
    content = f"{str(meaning_rpn).strip()}|{str(meaning_class).strip()}|{str(domain).strip()}"
    galaxy_ref = str(galaxy_ref).strip()
    if galaxy_ref:
        content = f"{content}|{galaxy_ref}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _coerce_trit(value: int | float | None) -> int:
    try:
        resolved = int(value)
    except Exception:
        resolved = 0
    return max(-1, min(1, resolved))


def _coerce_refs(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _coerce_embedding(values: Any, dim: int) -> EmbeddingVector | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return None
    if not isinstance(values, (list, tuple)):
        return None
    try:
        resolved = [float(value) for value in values]
    except Exception:
        return None
    if not resolved:
        return None
    if len(resolved) < dim:
        resolved.extend(0.0 for _ in range(dim - len(resolved)))
    return tuple(resolved[:dim])


@dataclass
class SurfaceForm:
    """Language-specific surface form stored as references, not inline copies."""

    word_ref: str
    char_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.word_ref = str(self.word_ref).strip()
        self.char_refs = _coerce_refs(self.char_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_ref": self.word_ref,
            "char_refs": list(self.char_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SurfaceForm":
        payload = dict(payload or {})
        return cls(
            word_ref=str(payload.get("word_ref", "")).strip(),
            char_refs=_coerce_refs(payload.get("char_refs")),
        )


@dataclass
class MeaningCentricStar:
    """Atomic meaning-centric knowledge unit for House/Galaxy storage."""

    star_id: str = ""
    meaning_class: str = "concept"
    meaning_rpn: str = ""
    domain: str = ""
    taxonomy_refs: list[str] = field(default_factory=list)
    surface_forms: dict[str, SurfaceForm] = field(default_factory=dict)
    visual_rpn: str | None = None
    visual_refs: list[str] = field(default_factory=list)
    audio_rpn: str | None = None
    audio_refs: list[str] = field(default_factory=list)
    pronunciations: dict[str, str] = field(default_factory=dict)
    behavior_rpn: str | None = None
    reality_refs: list[str] = field(default_factory=list)
    grammar_refs: list[str] = field(default_factory=list)
    meta_refs: list[str] = field(default_factory=list)
    house_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    house_room: str = ""
    galaxy_ref: str = ""
    confidence: int = 0
    polarity: int = 0
    embedding_64: EmbeddingVector | None = None
    embedding_128: EmbeddingVector | None = None
    embedding_512: EmbeddingVector | None = None
    embedding_2048: EmbeddingVector | None = None
    component_refs: list[str] = field(default_factory=list)
    composite_of: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.meaning_class = str(self.meaning_class).strip() or "concept"
        self.meaning_rpn = str(self.meaning_rpn).strip()
        self.domain = str(self.domain).strip()
        self.star_id = str(self.star_id).strip() or compute_star_id(
            self.meaning_rpn,
            self.meaning_class,
            self.domain,
            self.galaxy_ref,
        )
        self.taxonomy_refs = _coerce_refs(self.taxonomy_refs)
        self.surface_forms = {
            str(language).strip().lower(): (
                value if isinstance(value, SurfaceForm) else SurfaceForm.from_dict(value)
            )
            for language, value in dict(self.surface_forms).items()
            if str(language).strip()
        }
        self.visual_refs = _coerce_refs(self.visual_refs)
        self.visual_rpn = _coerce_optional_text(self.visual_rpn)
        self.audio_rpn = _coerce_optional_text(self.audio_rpn)
        self.audio_refs = _coerce_refs(self.audio_refs)
        self.pronunciations = {
            str(language).strip().lower(): str(ref).strip()
            for language, ref in dict(self.pronunciations).items()
            if str(language).strip() and str(ref).strip()
        }
        self.behavior_rpn = _coerce_optional_text(self.behavior_rpn)
        self.reality_refs = _coerce_refs(self.reality_refs)
        self.grammar_refs = _coerce_refs(self.grammar_refs)
        self.meta_refs = _coerce_refs(self.meta_refs)
        self.house_room = str(self.house_room).strip()
        self.galaxy_ref = str(self.galaxy_ref).strip()
        raw_position = list(self.house_position) if isinstance(self.house_position, (list, tuple)) else [0.0, 0.0, 0.0]
        while len(raw_position) < 3:
            raw_position.append(0.0)
        self.house_position = tuple(float(raw_position[index]) for index in range(3))
        self.confidence = _coerce_trit(self.confidence)
        self.polarity = _coerce_trit(self.polarity)
        self.embedding_64 = _coerce_embedding(self.embedding_64, 64)
        self.embedding_128 = _coerce_embedding(self.embedding_128, 128)
        self.embedding_512 = _coerce_embedding(self.embedding_512, 512)
        self.embedding_2048 = _coerce_embedding(self.embedding_2048, 2048)
        self.component_refs = _coerce_refs(self.component_refs)
        self.composite_of = _coerce_refs(self.composite_of)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "star_id": self.star_id,
            "meaning_class": self.meaning_class,
            "meaning_rpn": self.meaning_rpn,
            "domain": self.domain,
            "taxonomy_refs": list(self.taxonomy_refs),
            "surface_forms": {
                language: surface_form.to_dict()
                for language, surface_form in self.surface_forms.items()
            },
            "visual_rpn": self.visual_rpn,
            "visual_refs": list(self.visual_refs),
            "audio_rpn": self.audio_rpn,
            "audio_refs": list(self.audio_refs),
            "pronunciations": dict(self.pronunciations),
            "behavior_rpn": self.behavior_rpn,
            "reality_refs": list(self.reality_refs),
            "grammar_refs": list(self.grammar_refs),
            "meta_refs": list(self.meta_refs),
            "house_position": [float(value) for value in self.house_position],
            "house_room": self.house_room,
            "confidence": int(self.confidence),
            "polarity": int(self.polarity),
            "embedding_64": list(self.embedding_64) if self.embedding_64 is not None else None,
            "embedding_128": list(self.embedding_128) if self.embedding_128 is not None else None,
            "embedding_512": list(self.embedding_512) if self.embedding_512 is not None else None,
            "embedding_2048": list(self.embedding_2048) if self.embedding_2048 is not None else None,
            "component_refs": list(self.component_refs),
            "composite_of": list(self.composite_of),
        }
        if self.galaxy_ref:
            payload["galaxy_ref"] = self.galaxy_ref
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "MeaningCentricStar":
        payload = dict(payload or {})
        return cls(
            star_id=str(payload.get("star_id", "")).strip(),
            meaning_class=str(payload.get("meaning_class", "concept")).strip(),
            meaning_rpn=str(payload.get("meaning_rpn", "")).strip(),
            domain=str(payload.get("domain", "")).strip(),
            taxonomy_refs=_coerce_refs(payload.get("taxonomy_refs")),
            surface_forms={
                str(language).strip().lower(): SurfaceForm.from_dict(raw_value)
                for language, raw_value in dict(payload.get("surface_forms", {}) or {}).items()
                if str(language).strip()
            },
            visual_rpn=payload.get("visual_rpn"),
            visual_refs=_coerce_refs(payload.get("visual_refs")),
            audio_rpn=payload.get("audio_rpn"),
            audio_refs=_coerce_refs(payload.get("audio_refs")),
            pronunciations=dict(payload.get("pronunciations", {}) or {}),
            behavior_rpn=payload.get("behavior_rpn"),
            reality_refs=_coerce_refs(payload.get("reality_refs")),
            grammar_refs=_coerce_refs(payload.get("grammar_refs")),
            meta_refs=_coerce_refs(payload.get("meta_refs")),
            house_position=tuple(payload.get("house_position", (0.0, 0.0, 0.0))),
            house_room=str(payload.get("house_room", "")).strip(),
            galaxy_ref=str(payload.get("galaxy_ref", "")).strip(),
            confidence=_coerce_trit(payload.get("confidence")),
            polarity=_coerce_trit(payload.get("polarity")),
            embedding_64=payload.get("embedding_64"),
            embedding_128=payload.get("embedding_128"),
            embedding_512=payload.get("embedding_512"),
            embedding_2048=payload.get("embedding_2048"),
            component_refs=_coerce_refs(payload.get("component_refs")),
            composite_of=_coerce_refs(payload.get("composite_of")),
        )

    def to_galaxy_entry(
        self,
        *,
        entry_id: str | None = None,
        name: str | None = None,
        galaxy_name: str | None = None,
        category: str = "meaning_star",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry_metadata = dict(metadata or {})
        entry_metadata["meaning_star"] = self.to_dict()
        entry_metadata["meaning_star_id"] = self.star_id
        entry_metadata["meaning_class"] = self.meaning_class
        entry_metadata["house_room"] = self.house_room
        if self.galaxy_ref:
            entry_metadata["galaxy_ref"] = self.galaxy_ref
        entry_metadata["surface_form_languages"] = sorted(self.surface_forms.keys())
        preferred_name = name or self._preferred_name()
        return {
            "id": str(entry_id).strip() if entry_id is not None else self.star_id,
            "name": preferred_name,
            "domain": str(galaxy_name or self.domain or "meaning").strip().lower(),
            "category": str(category).strip() or "meaning_star",
            "content": preferred_name,
            "summary": preferred_name,
            "description": self.domain or preferred_name,
            "rpn_program": self.meaning_rpn,
            "metadata": entry_metadata,
        }

    @classmethod
    def from_galaxy_entry(cls, entry: Mapping[str, Any]) -> "MeaningCentricStar":
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        payload = None
        if isinstance(entry.get("meaning_star"), Mapping):
            payload = dict(entry.get("meaning_star"))
        elif isinstance(metadata, Mapping) and isinstance(metadata.get("meaning_star"), Mapping):
            payload = dict(metadata.get("meaning_star"))
        if payload is not None:
            return cls.from_dict(payload)
        return cls(
            star_id=str(entry.get("id", "")).strip(),
            meaning_class=str(metadata.get("meaning_class", "concept")).strip(),
            meaning_rpn=str(entry.get("rpn_program", "")).strip(),
            domain=str(entry.get("domain", "")).strip(),
            house_room=str(metadata.get("house_room", "")).strip(),
            galaxy_ref=str(metadata.get("galaxy_ref", "")).strip(),
        )

    def _preferred_name(self) -> str:
        for language in ("en", "pt", "ja"):
            surface_form = self.surface_forms.get(language)
            if surface_form is not None and surface_form.word_ref:
                return surface_form.word_ref
        return self.star_id


def wrap_galaxy_entry_with_meaning_star(
    entry: Mapping[str, Any],
    star: MeaningCentricStar,
) -> dict[str, Any]:
    """Attach meaning-centric star metadata without changing the host entry id."""
    wrapped = dict(entry)
    metadata = dict(wrapped.get("metadata") or {})
    metadata["meaning_star"] = star.to_dict()
    metadata["meaning_star_id"] = star.star_id
    metadata["meaning_class"] = star.meaning_class
    if star.galaxy_ref:
        metadata["galaxy_ref"] = star.galaxy_ref
    wrapped["metadata"] = metadata
    return wrapped


__all__ = [
    "MeaningCentricStar",
    "SurfaceForm",
    "EmbeddingVector",
    "TRIT_VALUES",
    "compute_star_id",
    "wrap_galaxy_entry_with_meaning_star",
]
