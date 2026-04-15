"""Shared bidirectional symlink helpers for meaning-centric ingestion."""

from __future__ import annotations

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm


def append_ref(star: MeaningCentricStar, field_path: str, ref_id: str) -> None:
    if field_path == "taxonomy_refs":
        if ref_id not in star.taxonomy_refs:
            star.taxonomy_refs.append(ref_id)
        return
    if field_path == "meta_refs":
        if ref_id not in star.meta_refs:
            star.meta_refs.append(ref_id)
        return
    if field_path == "char_refs":
        if ref_id not in star.char_refs:
            star.char_refs.append(ref_id)
        return
    if field_path == "mathematical_role":
        if ref_id not in star.taxonomy_refs:
            star.taxonomy_refs.append(ref_id)
        return
    if field_path == "grammar_refs":
        if ref_id not in star.grammar_refs:
            star.grammar_refs.append(ref_id)
        return
    if field_path == "reality_refs":
        if ref_id not in star.reality_refs:
            star.reality_refs.append(ref_id)
        return
    if field_path == "visual_refs":
        if ref_id not in star.visual_refs:
            star.visual_refs.append(ref_id)
        return
    if field_path == "audio_refs":
        if ref_id not in star.audio_refs:
            star.audio_refs.append(ref_id)
        return
    if field_path == "component_refs":
        if ref_id not in star.component_refs:
            star.component_refs.append(ref_id)
        return
    if field_path == "composite_of":
        if ref_id not in star.composite_of:
            star.composite_of.append(ref_id)
        return
    if field_path.startswith("surface_forms.") and field_path.endswith(".word_ref"):
        _, language, _ = field_path.split(".", 2)
        form = star.surface_forms.setdefault(language, SurfaceForm(word_ref="", char_refs=[]))
        form.word_ref = ref_id
        return
    if field_path.startswith("surface_forms.") and field_path.endswith(".char_refs"):
        _, language, _ = field_path.split(".", 2)
        form = star.surface_forms.setdefault(language, SurfaceForm(word_ref="", char_refs=[]))
        if ref_id not in form.char_refs:
            form.char_refs.append(ref_id)
        return
    raise ValueError(f"unsupported_symlink_field:{field_path}")


def link(left: MeaningCentricStar, right: MeaningCentricStar, forward_kind: str, backward_kind: str) -> None:
    append_ref(left, forward_kind, right.star_id)
    append_ref(right, backward_kind, left.star_id)


__all__ = ["append_ref", "link"]
