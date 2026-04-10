"""Boot-time symbol table for the sovereign algebraic system."""

from __future__ import annotations

import hashlib
from typing import Any

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


SYMBOL_REGISTRY: dict[int, tuple[str, float]] = {
    0x10: ("PI", 3.141592653589793),
    0x11: ("E", 2.718281828459045),
    0x20: ("G", 6.67430e-11),
    0x21: ("c", 299792458.0),
    0x22: ("h", 6.62607015e-34),
    0x23: ("hbar", 1.054571817e-34),
    0x24: ("k_B", 1.380649e-23),
    0x25: ("N_A", 6.02214076e23),
    0x26: ("e", 1.602176634e-19),
    0x27: ("eps0", 8.8541878128e-12),
    0x28: ("mu0", 1.25663706212e-6),
}

_SYMBOL_ALIASES = {
    "pi": 0x10,
    "e": 0x11,
    "g": 0x20,
    "c": 0x21,
    "h": 0x22,
    "hbar": 0x23,
    "ħ": 0x23,
    "k_b": 0x24,
    "kb": 0x24,
    "n_a": 0x25,
    "na": 0x25,
    "eps0": 0x27,
    "epsilon_0": 0x27,
    "epsilon0": 0x27,
    "ε0": 0x27,
    "mu0": 0x28,
    "μ0": 0x28,
}


def _stable_u32(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _extract_numeric(star: MeaningCentricStar) -> float | None:
    for ref in list(star.meta_refs or []) + list(star.reality_refs or []):
        try:
            return float(ref)
        except (TypeError, ValueError):
            continue
    return None


def _iter_meaning_stars(galaxy_manager: Any):
    if galaxy_manager is None:
        return
    for galaxy_name in ("Reality", "Math"):
        try:
            galaxy = galaxy_manager.get_galaxy(galaxy_name)
        except Exception:
            continue
        entries = list(getattr(galaxy, "entries", getattr(galaxy, "_extra_entries", [])) or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                yield MeaningCentricStar.from_galaxy_entry(entry)
            except Exception:
                continue


def _resolve_symbol_slot(star: MeaningCentricStar) -> int | None:
    candidates = [
        star.star_id,
        star.meaning_class,
        star.domain,
        *list(star.taxonomy_refs or []),
        *list(star.meta_refs or []),
        *list(star.reality_refs or []),
    ]
    for surface_form in (star.surface_forms or {}).values():
        candidates.append(getattr(surface_form, "word_ref", ""))
    normalized = {str(value).strip().lower() for value in candidates if str(value).strip()}
    for key, slot in _SYMBOL_ALIASES.items():
        if key in normalized:
            return slot
    return None


def build_symbol_table(galaxy_manager=None) -> tuple[list[float], list[int]]:
    """Return (values[256], star_id_indices[256]) for __constant__-memory upload."""
    values = [0.0] * 256
    star_ids = [0] * 256
    for sym_id, (_name, default_val) in SYMBOL_REGISTRY.items():
        values[sym_id] = float(default_val)

    if galaxy_manager is None:
        return values, star_ids

    for star in _iter_meaning_stars(galaxy_manager):
        if star.meaning_class not in {"physical_constant", "mathematical_constant", "variable"}:
            continue
        slot = _resolve_symbol_slot(star)
        if slot is None:
            continue
        numeric_val = _extract_numeric(star)
        if numeric_val is not None:
            values[slot] = float(numeric_val)
        star_ids[slot] = _stable_u32(star.star_id)
    return values, star_ids


__all__ = ["SYMBOL_REGISTRY", "build_symbol_table"]
