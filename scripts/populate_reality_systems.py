from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.cranium.physics_demo import (  # noqa: E402
    ConstantAcceleration1D,
    HarmonicOscillator1D,
    Heat1D,
    Heat2D,
    Orbital2D,
)

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e28_reality_population_v1"
DEFAULT_GALAXY_DIR = Path("/K3D/Knowledge3D.local/galaxies")


def _system_entry(
    *,
    entry_id: str,
    name: str,
    domain: str,
    content: str,
    description: str,
    component_refs: list[str],
    behavior_rpn: str,
    law_rpn: str,
    visual_rpn: str,
    tags: list[str],
    source_class: type[Any],
    reusable_contexts: list[str],
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": domain,
        "category": "reality_system",
        "layer": 3,
        "content": content,
        "description": description,
        "behavior_rpn": behavior_rpn,
        "law_rpn": law_rpn,
        "visual_rpn": visual_rpn,
        "component_refs": list(component_refs),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "component_refs": list(component_refs),
            "py_origin": f"knowledge3d.cranium.physics_demo.{source_class.__name__}",
            "reusable_contexts": list(reusable_contexts),
            "surface_forms": {
                "en": name.lower(),
                "pt": name.lower(),
            },
        },
        "tags": list(tags),
    }


def _atom_entry(
    *,
    entry_id: str,
    name: str,
    domain: str,
    content: str,
    symbol: str,
    unit: str,
    behavior_rpn: str,
    visual_rpn: str,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": domain,
        "category": "reality_atom",
        "layer": 1,
        "content": content,
        "description": content,
        "behavior_rpn": behavior_rpn,
        "visual_rpn": visual_rpn,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "symbol": symbol,
            "unit": unit,
            "surface_forms": {
                "en": name.lower(),
                "pt": name.lower(),
            },
        },
        "tags": list(tags),
    }


REALITY_ATOMS: list[dict[str, Any]] = [
    _atom_entry(
        entry_id="reality_atom_position_1d",
        name="Position 1D",
        domain="physics_kinematics",
        content="Scalar position along one axis.",
        symbol="x",
        unit="m",
        behavior_rpn="x RECALL",
        visual_rpn="x RECALL 0 DRAW_MOVE x RECALL 1 ADD 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "kinematics", "position", "scalar"],
    ),
    _atom_entry(
        entry_id="reality_atom_velocity_1d",
        name="Velocity 1D",
        domain="physics_kinematics",
        content="Scalar velocity along one axis.",
        symbol="v",
        unit="m/s",
        behavior_rpn="v RECALL",
        visual_rpn="0 0 DRAW_MOVE v RECALL 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "kinematics", "velocity", "scalar"],
    ),
    _atom_entry(
        entry_id="reality_atom_acceleration_1d",
        name="Acceleration 1D",
        domain="physics_kinematics",
        content="Scalar acceleration along one axis.",
        symbol="a",
        unit="m/s^2",
        behavior_rpn="a RECALL",
        visual_rpn="0 0 DRAW_MOVE a RECALL 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "kinematics", "acceleration", "scalar"],
    ),
    _atom_entry(
        entry_id="reality_atom_timestep",
        name="Timestep",
        domain="physics_shared",
        content="Discrete simulation timestep.",
        symbol="dt",
        unit="s",
        behavior_rpn="dt RECALL",
        visual_rpn="0 0 DRAW_MOVE 1 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "shared", "time", "timestep"],
    ),
    _atom_entry(
        entry_id="reality_atom_angular_frequency",
        name="Angular Frequency",
        domain="physics_oscillation",
        content="Angular frequency controlling oscillator curvature.",
        symbol="omega",
        unit="rad/s",
        behavior_rpn="omega RECALL",
        visual_rpn="0 0 DRAW_MOVE omega RECALL 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "oscillation", "frequency"],
    ),
    _atom_entry(
        entry_id="reality_atom_position_2d",
        name="Position 2D",
        domain="physics_orbital",
        content="Two-dimensional position vector.",
        symbol="r",
        unit="m",
        behavior_rpn="x RECALL y RECALL",
        visual_rpn="x RECALL y RECALL DRAW_MOVE x RECALL 1 ADD y RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "position", "vector"],
    ),
    _atom_entry(
        entry_id="reality_atom_velocity_2d",
        name="Velocity 2D",
        domain="physics_orbital",
        content="Two-dimensional velocity vector.",
        symbol="v_vec",
        unit="m/s",
        behavior_rpn="vx RECALL vy RECALL",
        visual_rpn="0 0 DRAW_MOVE vx RECALL vy RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "velocity", "vector"],
    ),
    _atom_entry(
        entry_id="reality_atom_mass",
        name="Mass",
        domain="physics_orbital",
        content="Mass scalar participating in orbital dynamics.",
        symbol="m",
        unit="kg",
        behavior_rpn="m RECALL",
        visual_rpn="0 0 DRAW_MOVE 0 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "mass"],
    ),
    _atom_entry(
        entry_id="reality_atom_gravitational_parameter",
        name="Gravitational Parameter",
        domain="physics_orbital",
        content="Gravitational parameter mu = G*M.",
        symbol="mu",
        unit="m^3/s^2",
        behavior_rpn="mu RECALL",
        visual_rpn="0 0 DRAW_MOVE mu RECALL 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "gravity"],
    ),
    _atom_entry(
        entry_id="reality_atom_radial_magnitude",
        name="Radial Magnitude",
        domain="physics_orbital",
        content="Distance from the orbital center.",
        symbol="r_mag",
        unit="m",
        behavior_rpn="x RECALL x RECALL MUL y RECALL y RECALL MUL ADD SQRT",
        visual_rpn="0 0 DRAW_MOVE r_mag RECALL 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "radius"],
    ),
    _atom_entry(
        entry_id="reality_atom_temperature_scalar",
        name="Temperature Scalar",
        domain="physics_heat",
        content="Temperature value on a 1D thermal lattice.",
        symbol="T_center",
        unit="K",
        behavior_rpn="T_center RECALL",
        visual_rpn="0 T_center RECALL DRAW_MOVE 1 T_center RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "temperature", "scalar"],
    ),
    _atom_entry(
        entry_id="reality_atom_thermal_diffusivity",
        name="Thermal Diffusivity",
        domain="physics_heat",
        content="Thermal diffusivity constant alpha.",
        symbol="alpha",
        unit="m^2/s",
        behavior_rpn="alpha RECALL",
        visual_rpn="0 0 DRAW_MOVE alpha RECALL 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "diffusivity"],
    ),
    _atom_entry(
        entry_id="reality_atom_grid_spacing",
        name="Grid Spacing",
        domain="physics_heat",
        content="Uniform spacing dx for a 1D grid.",
        symbol="dx",
        unit="m",
        behavior_rpn="dx RECALL",
        visual_rpn="0 0 DRAW_MOVE dx RECALL 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "grid"],
    ),
    _atom_entry(
        entry_id="reality_atom_heat_laplacian_1d",
        name="Heat Laplacian 1D",
        domain="physics_heat",
        content="Second-difference stencil for one-dimensional heat flow.",
        symbol="lap_1d",
        unit="K",
        behavior_rpn="T_right RECALL T_center RECALL 2 MUL SUB T_left RECALL ADD",
        visual_rpn="0 0 DRAW_MOVE 1 1 DRAW_LINE 2 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "laplacian", "stencil"],
    ),
    _atom_entry(
        entry_id="reality_atom_temperature_field_2d",
        name="Temperature Field 2D",
        domain="physics_heat",
        content="Two-dimensional temperature field on a grid.",
        symbol="T_ij",
        unit="K",
        behavior_rpn="T_ij RECALL",
        visual_rpn="0 0 DRAW_MOVE 1 0 DRAW_LINE 1 1 DRAW_LINE 0 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "temperature", "field"],
    ),
    _atom_entry(
        entry_id="reality_atom_grid_spacing_x",
        name="Grid Spacing X",
        domain="physics_heat",
        content="Horizontal grid spacing dx for a 2D field.",
        symbol="dx",
        unit="m",
        behavior_rpn="dx RECALL",
        visual_rpn="0 0 DRAW_MOVE dx RECALL 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "grid", "x_axis"],
    ),
    _atom_entry(
        entry_id="reality_atom_grid_spacing_y",
        name="Grid Spacing Y",
        domain="physics_heat",
        content="Vertical grid spacing dy for a 2D field.",
        symbol="dy",
        unit="m",
        behavior_rpn="dy RECALL",
        visual_rpn="0 0 DRAW_MOVE 0 dy RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "grid", "y_axis"],
    ),
    _atom_entry(
        entry_id="reality_atom_heat_laplacian_2d",
        name="Heat Laplacian 2D",
        domain="physics_heat",
        content="Five-point stencil for two-dimensional heat flow.",
        symbol="lap_2d",
        unit="K",
        behavior_rpn="T_up RECALL T_down RECALL ADD T_left RECALL ADD T_right RECALL ADD T_center RECALL 4 MUL SUB",
        visual_rpn="1 0 DRAW_MOVE 0 1 DRAW_LINE 1 2 DRAW_LINE 2 1 DRAW_LINE 1 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "laplacian", "stencil", "2d"],
    ),
]


REALITY_SYSTEMS: list[dict[str, Any]] = [
    _system_entry(
        entry_id="reality_system_constant_acceleration_1d",
        name="Constant Acceleration 1D",
        domain="physics_kinematics",
        content="One-dimensional constant acceleration with Euler-style integration.",
        description="v(t+1) = v(t) + a*dt and x(t+1) = x(t) + v(t+1)*dt.",
        component_refs=[
            "reality_atom_position_1d",
            "reality_atom_velocity_1d",
            "reality_atom_acceleration_1d",
            "reality_atom_timestep",
        ],
        behavior_rpn="v RECALL a RECALL dt RECALL MUL ADD v STORE x RECALL v RECALL dt RECALL MUL ADD x STORE",
        law_rpn="v_prev RECALL a RECALL dt RECALL MUL ADD v RECALL SUB ABS tolerance RECALL LTE",
        visual_rpn="x RECALL 0 DRAW_MOVE x RECALL 1 ADD 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "kinematics", "motion", "euler"],
        source_class=ConstantAcceleration1D,
        reusable_contexts=["physics_sim", "kinematics", "projectile_motion", "free_fall"],
    ),
    _system_entry(
        entry_id="reality_system_harmonic_oscillator_1d",
        name="Harmonic Oscillator 1D",
        domain="physics_oscillation",
        content="One-dimensional harmonic oscillator encoded as a first-order update.",
        description="a = -omega^2 * x, then integrate velocity and position over dt.",
        component_refs=[
            "reality_atom_position_1d",
            "reality_atom_velocity_1d",
            "reality_atom_angular_frequency",
            "reality_atom_timestep",
        ],
        behavior_rpn="omega RECALL omega RECALL MUL x RECALL MUL NEG a STORE v RECALL a RECALL dt RECALL MUL ADD v STORE x RECALL v RECALL dt RECALL MUL ADD x STORE",
        law_rpn="omega RECALL omega RECALL MUL x RECALL MUL NEG a_expected STORE a_expected RECALL a RECALL SUB ABS tolerance RECALL LTE",
        visual_rpn="0 x RECALL DRAW_MOVE 1 v RECALL DRAW_LINE 2 x RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "oscillation", "harmonic", "euler"],
        source_class=HarmonicOscillator1D,
        reusable_contexts=["physics_sim", "oscillation", "signal_modeling", "wave_motion"],
    ),
    _system_entry(
        entry_id="reality_system_orbital_2d",
        name="Orbital 2D",
        domain="physics_orbital",
        content="Two-dimensional Newtonian orbit under a central gravitational field.",
        description="a = -mu * r / |r|^3 integrated over velocity and position components.",
        component_refs=[
            "reality_atom_position_2d",
            "reality_atom_velocity_2d",
            "reality_atom_mass",
            "reality_atom_gravitational_parameter",
            "reality_atom_radial_magnitude",
            "reality_atom_timestep",
        ],
        behavior_rpn="mu RECALL r_mag RECALL 3 POW DIV NEG ax STORE mu RECALL r_mag RECALL 3 POW DIV NEG ay STORE vx RECALL ax RECALL x RECALL MUL MUL dt RECALL MUL ADD vx STORE vy RECALL ay RECALL y RECALL MUL MUL dt RECALL MUL ADD vy STORE x RECALL vx RECALL dt RECALL MUL ADD x STORE y RECALL vy RECALL dt RECALL MUL ADD y STORE",
        law_rpn="r_mag RECALL 0 GT vx RECALL vx_prev RECALL SUB ABS vy RECALL vy_prev RECALL SUB ABS ADD tolerance RECALL LTE AND",
        visual_rpn="x RECALL y RECALL DRAW_MOVE 0 0 DRAW_LINE DRAW_STROKE",
        tags=["physics", "orbital", "gravity", "trajectory"],
        source_class=Orbital2D,
        reusable_contexts=["physics_sim", "orbital_dynamics", "celestial_mechanics", "navigation"],
    ),
    _system_entry(
        entry_id="reality_system_heat_1d",
        name="Heat 1D",
        domain="physics_heat",
        content="One-dimensional heat diffusion using an explicit finite-difference stencil.",
        description="T_next = T_center + alpha * dt / dx^2 * (T_right - 2*T_center + T_left).",
        component_refs=[
            "reality_atom_temperature_scalar",
            "reality_atom_thermal_diffusivity",
            "reality_atom_grid_spacing",
            "reality_atom_heat_laplacian_1d",
            "reality_atom_timestep",
        ],
        behavior_rpn="alpha RECALL dt RECALL MUL dx RECALL dx RECALL MUL DIV lap_1d RECALL MUL T_center RECALL ADD T_center STORE",
        law_rpn="T_right RECALL T_center RECALL 2 MUL SUB T_left RECALL ADD lap_1d RECALL SUB ABS tolerance RECALL LTE",
        visual_rpn="0 T_left RECALL DRAW_MOVE 1 T_center RECALL DRAW_LINE 2 T_right RECALL DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "diffusion", "pde"],
        source_class=Heat1D,
        reusable_contexts=["physics_sim", "heat_transfer", "diffusion", "finite_difference"],
    ),
    _system_entry(
        entry_id="reality_system_heat_2d",
        name="Heat 2D",
        domain="physics_heat",
        content="Two-dimensional heat diffusion using a five-point stencil.",
        description="T_next = T_center + alpha * dt * laplacian(T) across a rectangular grid.",
        component_refs=[
            "reality_atom_temperature_field_2d",
            "reality_atom_thermal_diffusivity",
            "reality_atom_grid_spacing_x",
            "reality_atom_grid_spacing_y",
            "reality_atom_heat_laplacian_2d",
            "reality_atom_timestep",
        ],
        behavior_rpn="alpha RECALL dt RECALL MUL lap_2d RECALL MUL T_ij RECALL ADD T_ij STORE",
        law_rpn="T_up RECALL T_down RECALL ADD T_left RECALL ADD T_right RECALL ADD T_center RECALL 4 MUL SUB lap_2d RECALL SUB ABS tolerance RECALL LTE",
        visual_rpn="0 1 DRAW_MOVE 1 0 DRAW_LINE 2 1 DRAW_LINE 1 2 DRAW_LINE 0 1 DRAW_LINE DRAW_STROKE",
        tags=["physics", "heat", "diffusion", "pde", "2d"],
        source_class=Heat2D,
        reusable_contexts=["physics_sim", "heat_transfer", "diffusion", "finite_difference", "field_simulation"],
    ),
]


def build_reality_system_entries() -> list[dict[str, Any]]:
    return [dict(row) for row in REALITY_ATOMS + REALITY_SYSTEMS]


def populate_reality_systems(*, galaxy_dir: Path = DEFAULT_GALAXY_DIR) -> dict[str, dict[str, int]]:
    galaxy_dir = Path(galaxy_dir)
    result = upsert_entries(
        galaxy_dir / "Reality.jsonl",
        build_reality_system_entries(),
    )
    return {"Reality.jsonl": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Reality galaxy with physics systems and atoms.")
    parser.add_argument(
        "--galaxy-dir",
        type=Path,
        default=DEFAULT_GALAXY_DIR,
        help="Directory containing galaxy JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = populate_reality_systems(galaxy_dir=args.galaxy_dir)
    stats = summary["Reality.jsonl"]
    print(
        "Reality.jsonl:"
        f" before={stats['before']}"
        f" after={stats['after']}"
        f" appended={stats['appended']}"
        f" replaced={stats['replaced']}"
        f" removed={stats['removed']}"
    )


if __name__ == "__main__":
    main()
