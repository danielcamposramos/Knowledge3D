"""Bootstrap helpers for Reality Enabler physics nodes."""

from __future__ import annotations

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_nodes import RealityAtom, RealitySystem


def bootstrap_physics_atoms(galaxy: RealityGalaxy, *, encode_embedding: bool = False) -> None:
    """Add foundational physics atoms."""
    point_mass = RealityAtom(
        node_id="point_mass",
        visual_rpn="x y 0.05 circle fill",
        behavior_rpn="",
        metadata={"description": "Dimensionless point mass"},
    )
    galaxy.add_node(point_mass, encode_embedding=encode_embedding)


def bootstrap_constant_acceleration_system(
    galaxy: RealityGalaxy,
    *,
    encode_embedding: bool = False,
) -> None:
    """Create constant-acceleration system (1D explicit Euler)."""
    behavior_rpn = " ".join(
        [
            "v RECALL a RECALL dt RECALL * + v STORE",
            "x RECALL v RECALL dt RECALL * + x STORE",
        ]
    )

    system = RealitySystem(
        node_id="system:constant_accel_1d",
        component_refs=[],
        state={"x": 0.0, "v": 1.0, "a": -9.81, "dt": 0.01, "F": -9.81, "m": 1.0},
        behavior_rpn=behavior_rpn,
        law_rpn="F RECALL m RECALL / a RECALL - abs 1e-6 lt",  # optional invariant
        visual_rpn="x 0 move x 0 0.05 circle fill",
        metadata={
            "description": "1D constant acceleration (explicit Euler)",
            "equations": "v' = a, x' = v",
            "analytic_solution": "x(t)=x0+v0*t+0.5*a*t^2",
        },
    )
    galaxy.add_node(system, encode_embedding=encode_embedding)


def bootstrap_harmonic_oscillator_system(
    galaxy: RealityGalaxy,
    *,
    encode_embedding: bool = False,
) -> None:
    """Create 1D harmonic oscillator (explicit Euler)."""
    behavior_rpn = " ".join(
        [
            "x RECALL omega RECALL omega RECALL * * -1 * a STORE",
            "v RECALL a RECALL dt RECALL * + v STORE",
            "x RECALL v RECALL dt RECALL * + x STORE",
        ]
    )

    system = RealitySystem(
        node_id="system:harmonic_osc_1d",
        component_refs=[],
        state={"x": 1.0, "v": 0.0, "a": 0.0, "omega": 1.0, "dt": 0.001},
        behavior_rpn=behavior_rpn,
        law_rpn="v RECALL dup * x RECALL omega RECALL dup * * + 0.5 *",  # energy metric
        visual_rpn="x 0 move x 0 0.05 circle fill",
        metadata={
            "description": "1D harmonic oscillator (explicit Euler)",
            "equations": "v' = -omega^2 * x, x' = v",
        },
    )
    galaxy.add_node(system, encode_embedding=encode_embedding)
