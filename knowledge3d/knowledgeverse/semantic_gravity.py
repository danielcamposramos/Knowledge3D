"""Semantic gravity for Galaxy working-memory organization.

This module is ingestion/sleep-time infrastructure. It must not be used to
rewrite House placement, which remains intentional and explicit.
"""

from __future__ import annotations

from math import sqrt
from typing import Mapping

from .meaning_star import MeaningCentricStar


Vec3 = tuple[float, float, float]


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    limit = min(len(left), len(right))
    return sum(float(left[index]) * float(right[index]) for index in range(limit))


def _norm(values: tuple[float, ...]) -> float:
    return sqrt(sum(float(value) * float(value) for value in values))


def _cosine_similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if left is None or right is None:
        return 0.0
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _current_position(star: MeaningCentricStar, positions: Mapping[str, Vec3] | None) -> Vec3:
    if positions and star.star_id in positions:
        position = positions[star.star_id]
        return (float(position[0]), float(position[1]), float(position[2]))
    return (
        float(star.house_position[0]),
        float(star.house_position[1]),
        float(star.house_position[2]),
    )


def ternary_semantic_force(star_a: MeaningCentricStar, star_b: MeaningCentricStar) -> int:
    """Return +1 attract, 0 neutral, -1 repel for Galaxy working-memory use."""
    if not star_a.star_id or not star_b.star_id:
        return 0
    if star_a.star_id == star_b.star_id:
        return 0

    shared_refs = set(star_a.taxonomy_refs) & set(star_b.taxonomy_refs)
    shared_refs |= set(star_a.component_refs) & set(star_b.component_refs)
    shared_refs |= set(star_a.grammar_refs) & set(star_b.grammar_refs)
    shared_refs |= set(star_a.reality_refs) & set(star_b.reality_refs)
    if shared_refs or star_a.star_id in star_b.taxonomy_refs or star_b.star_id in star_a.taxonomy_refs:
        return 1

    cosine = _cosine_similarity(star_a.embedding_128, star_b.embedding_128)
    if (
        star_a.polarity != 0
        and star_b.polarity != 0
        and star_a.polarity == -star_b.polarity
        and (star_a.domain == star_b.domain or cosine <= -0.30)
    ):
        return -1
    if cosine >= 0.35:
        return 1
    if cosine <= -0.35:
        return -1
    return 0


def meaning_mass(star: MeaningCentricStar) -> float:
    """Approximate semantic mass by connection richness across modalities."""
    mass = 1.0
    mass += float(len(star.taxonomy_refs))
    mass += float(len(star.surface_forms))
    mass += float(len(star.visual_refs))
    mass += float(len(star.audio_refs))
    mass += float(len(star.pronunciations))
    mass += float(len(star.reality_refs))
    mass += float(len(star.grammar_refs))
    mass += float(len(star.meta_refs))
    mass += float(len(star.component_refs))
    mass += float(len(star.composite_of))
    if star.visual_rpn:
        mass += 1.0
    if star.audio_rpn:
        mass += 1.0
    if star.behavior_rpn:
        mass += 1.0
    return mass


def semantic_gravity_force(
    star_a: MeaningCentricStar,
    star_b: MeaningCentricStar,
    distance: float,
) -> Vec3:
    """Compute scalar semantic force magnitude along the x-axis.

    This low-level helper only provides the magnitude term from the semantic
    gravity equation. Direction is injected by ``gravity_tick`` using the
    current working-memory positions.
    """
    scalar = float(ternary_semantic_force(star_a, star_b))
    if scalar == 0.0:
        return (0.0, 0.0, 0.0)
    safe_distance = max(float(distance), 1e-3)
    magnitude = scalar * meaning_mass(star_a) * meaning_mass(star_b) / (safe_distance * safe_distance)
    return (magnitude, 0.0, 0.0)


def gravity_tick(
    stars: list[MeaningCentricStar],
    dt: float = 0.01,
    damping: float = 0.95,
    *,
    positions: Mapping[str, Vec3] | None = None,
    velocities: Mapping[str, Vec3] | None = None,
) -> tuple[dict[str, Vec3], dict[str, Vec3]]:
    """Advance Galaxy working-memory positions by one semantic-gravity tick."""
    resolved_positions: dict[str, Vec3] = {
        star.star_id: _current_position(star, positions)
        for star in stars
    }
    resolved_velocities: dict[str, Vec3] = {
        star.star_id: (
            float((velocities or {}).get(star.star_id, (0.0, 0.0, 0.0))[0]),
            float((velocities or {}).get(star.star_id, (0.0, 0.0, 0.0))[1]),
            float((velocities or {}).get(star.star_id, (0.0, 0.0, 0.0))[2]),
        )
        for star in stars
    }
    net_forces: dict[str, list[float]] = {star.star_id: [0.0, 0.0, 0.0] for star in stars}

    for left_index, star_a in enumerate(stars):
        position_a = resolved_positions[star_a.star_id]
        for right_index in range(left_index + 1, len(stars)):
            star_b = stars[right_index]
            position_b = resolved_positions[star_b.star_id]
            delta_x = position_b[0] - position_a[0]
            delta_y = position_b[1] - position_a[1]
            delta_z = position_b[2] - position_a[2]
            distance = sqrt((delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z))
            magnitude = semantic_gravity_force(star_a, star_b, distance)[0]
            if abs(magnitude) <= 1e-12:
                continue
            if distance <= 1e-9:
                unit = (1.0, 0.0, 0.0)
            else:
                inverse = 1.0 / distance
                unit = (delta_x * inverse, delta_y * inverse, delta_z * inverse)
            force = (magnitude * unit[0], magnitude * unit[1], magnitude * unit[2])
            net_forces[star_a.star_id][0] += force[0]
            net_forces[star_a.star_id][1] += force[1]
            net_forces[star_a.star_id][2] += force[2]
            net_forces[star_b.star_id][0] -= force[0]
            net_forces[star_b.star_id][1] -= force[1]
            net_forces[star_b.star_id][2] -= force[2]

    next_positions: dict[str, Vec3] = {}
    next_velocities: dict[str, Vec3] = {}
    delta_t = max(float(dt), 0.0)
    damp = min(max(float(damping), 0.0), 1.0)
    for star in stars:
        star_id = star.star_id
        mass = max(meaning_mass(star), 1.0)
        acceleration = tuple(component / mass for component in net_forces[star_id])
        velocity = resolved_velocities[star_id]
        next_velocity = (
            (velocity[0] + (acceleration[0] * delta_t)) * damp,
            (velocity[1] + (acceleration[1] * delta_t)) * damp,
            (velocity[2] + (acceleration[2] * delta_t)) * damp,
        )
        position = resolved_positions[star_id]
        next_position = (
            position[0] + (next_velocity[0] * delta_t),
            position[1] + (next_velocity[1] * delta_t),
            position[2] + (next_velocity[2] * delta_t),
        )
        next_positions[star_id] = next_position
        next_velocities[star_id] = next_velocity
    return next_positions, next_velocities


__all__ = [
    "Vec3",
    "gravity_tick",
    "meaning_mass",
    "semantic_gravity_force",
    "ternary_semantic_force",
]
