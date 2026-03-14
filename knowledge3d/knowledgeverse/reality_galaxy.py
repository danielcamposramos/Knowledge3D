"""Knowledgeverse Reality Galaxy bootstrap (procedural, additive, idempotent).

This module generates deterministic procedural entries for the Reality galaxy.
Entries are designed for ingestion/retrieval paths and remain sovereign-friendly:
no external runtime dependencies are required in the hot path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry(
    *,
    entry_id: str,
    name: str,
    category: str,
    rpn_program: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_tags = list(tags or [])
    base_meta = {
        "source": "week19_reality_bootstrap",
        "bootstrap": "week19_reality_enabler",
        "procedural": True,
    }
    lowered = " ".join([str(category or "").lower(), *[str(tag).lower() for tag in resolved_tags]])
    inferred_subject = "reality"
    inferred_subfield = str(category or "reality")
    if any(token in lowered for token in ("kinematics", "dynamics", "electromagnetism", "thermodynamics")):
        inferred_subject = "physics"
        if "kinematics" in lowered or "projectile" in lowered:
            inferred_subfield = "kinematics"
        elif "electromagnetism" in lowered or "circuits" in lowered:
            inferred_subfield = "electromagnetism"
        elif "thermodynamics" in lowered or "entropy" in lowered or "ideal_gas" in lowered:
            inferred_subfield = "thermodynamics"
        else:
            inferred_subfield = "dynamics"
    elif any(token in lowered for token in ("procedural", "cellular_automata", "lsystem", "fractal", "eca")):
        inferred_subject = "computer_science"
        inferred_subfield = "procedural_generation"
    base_meta.update(
        {
            "subject": inferred_subject,
            "subfield": inferred_subfield,
            "query_anchor": f"{name} {str(category).replace('_', ' ')} {' '.join(resolved_tags)}".strip(),
        }
    )
    if metadata:
        base_meta.update(metadata)
    return {
        "id": entry_id,
        "name": name,
        "domain": "reality",
        "category": category,
        "rpn_program": rpn_program,
        "tags": resolved_tags,
        "metadata": base_meta,
    }


def create_kinematics_primitives() -> list[dict[str, Any]]:
    specs = [
        (
            "reality_kinematics_position_update_euler",
            "Position Update (Euler)",
            "X V DT MUL ADD",
            ["kinematics", "integration", "euler"],
            "position update displacement x plus v dt motion euler integration constant velocity",
        ),
        (
            "reality_kinematics_velocity_update_euler",
            "Velocity Update (Euler)",
            "V A DT MUL ADD",
            ["kinematics", "integration", "euler"],
            "velocity update acceleration v plus a dt motion euler integration change in speed",
        ),
        (
            "reality_kinematics_constant_acceleration",
            "Constant Acceleration Displacement",
            "X0 V0 DT MUL ADD 0.5 A MUL DT DT MUL ADD",
            ["kinematics", "constant_acceleration", "motion"],
            "constant acceleration displacement x0 plus v0 dt plus one half a dt squared uniformly accelerated motion",
        ),
        (
            "reality_kinematics_average_velocity",
            "Average Velocity",
            "DISPLACEMENT TIME DIV",
            ["kinematics", "velocity", "rate"],
            "average velocity displacement divided by time motion rate change in position over time",
        ),
        (
            "reality_kinematics_acceleration_definition",
            "Acceleration Definition",
            "DELTA_V DELTA_T DIV",
            ["kinematics", "acceleration", "rate"],
            "acceleration change in velocity divided by change in time rate of motion",
        ),
        (
            "reality_kinematics_projectile_2d",
            "Projectile Motion (2D)",
            "X VX DT MUL ADD Y VY DT MUL ADD 0.5 G MUL DT DT MUL MUL SUB",
            ["kinematics", "projectile", "composed"],
            "projectile motion launch angle horizontal vertical gravity parabola time of flight range",
        ),
        (
            "reality_kinematics_projectile_range",
            "Projectile Range",
            "V V MUL THETA2 SIN MUL G DIV",
            ["kinematics", "projectile", "range"],
            "projectile range launch speed angle distance gravity horizontal range",
        ),
        (
            "reality_kinematics_projectile_time_of_flight",
            "Projectile Time of Flight",
            "2 V MUL THETA SIN MUL G DIV",
            ["kinematics", "projectile", "time"],
            "projectile time of flight launch speed angle gravity vertical motion duration",
        ),
        (
            "reality_kinematics_uniform_circular_speed",
            "Uniform Circular Speed",
            "2 PI MUL R MUL T DIV",
            ["kinematics", "circular_motion", "speed"],
            "uniform circular motion speed circumference divided by period orbit rotation",
        ),
        (
            "reality_kinematics_relative_velocity",
            "Relative Velocity",
            "V_OBJECT V_OBSERVER SUB",
            ["kinematics", "relative_motion", "velocity"],
            "relative velocity object speed minus observer speed motion reference frame",
        ),
    ]
    return [
        _entry(
            entry_id=entry_id,
            name=name,
            category="kinematics",
            rpn_program=program,
            tags=tags,
            metadata={
                "cross_modal": ["drawing", "math"] if "projectile" in tags or "motion" in tags else ["math"],
                "query_anchor": query_anchor,
                "canonical_family": "physics_kinematics",
            },
        )
        for entry_id, name, program, tags, query_anchor in specs
    ]


def create_dynamics_primitives() -> list[dict[str, Any]]:
    specs = [
        (
            "reality_dynamics_newton_second_law",
            "Newton Second Law",
            "F M DIV",
            "dynamics",
            ["fundamental", "dynamics", "force"],
            "newton second law acceleration equals force divided by mass force motion dynamics",
        ),
        (
            "reality_dynamics_kinetic_energy",
            "Kinetic Energy",
            "0.5 M MUL V V MUL MUL",
            "dynamics",
            ["energy", "dynamics"],
            "kinetic energy one half m v squared moving body energy",
        ),
        (
            "reality_dynamics_potential_energy_gravity",
            "Gravitational Potential Energy",
            "M G MUL H MUL",
            "dynamics",
            ["energy", "gravity"],
            "gravitational potential energy m g h height gravity field",
        ),
        (
            "reality_dynamics_momentum_linear",
            "Linear Momentum",
            "M V MUL",
            "dynamics",
            ["momentum", "dynamics"],
            "linear momentum mass times velocity impulse collision motion",
        ),
        (
            "reality_dynamics_impulse",
            "Impulse",
            "F DELTA_T MUL",
            "dynamics",
            ["impulse", "dynamics"],
            "impulse force times time change in momentum collision push",
        ),
        (
            "reality_dynamics_work",
            "Work",
            "F D MUL",
            "dynamics",
            ["work", "energy"],
            "work force times displacement mechanical energy transfer",
        ),
        (
            "reality_dynamics_power",
            "Power",
            "WORK TIME DIV",
            "dynamics",
            ["power", "energy"],
            "power work divided by time rate of energy transfer",
        ),
        (
            "reality_dynamics_energy_conservation",
            "Mechanical Energy Conservation",
            "KE1 PE1 ADD KE2 PE2 ADD EQ",
            "dynamics",
            ["conservation", "energy"],
            "energy conservation total mechanical energy initial equals final",
        ),
        (
            "reality_dynamics_momentum_conservation_1d",
            "Momentum Conservation (1D)",
            "M1 V1 MUL M2 V2 MUL ADD",
            "dynamics",
            ["conservation", "momentum"],
            "momentum conservation one dimensional collision total momentum before equals after",
        ),
        (
            "reality_dynamics_torque",
            "Torque",
            "R F MUL THETA SIN MUL",
            "dynamics",
            ["torque", "rotation"],
            "torque lever arm force sine angle rotational dynamics moment",
        ),
        (
            "reality_dynamics_angular_momentum",
            "Angular Momentum",
            "I OMEGA MUL",
            "dynamics",
            ["angular_momentum", "rotation"],
            "angular momentum moment of inertia times angular velocity rotation",
        ),
        (
            "reality_dynamics_hooke_law",
            "Hooke Law",
            "K X MUL NEG",
            "dynamics",
            ["spring", "restoring_force"],
            "hooke law spring restoring force minus kx oscillation elasticity",
        ),
        (
            "reality_dynamics_friction_force",
            "Friction Force",
            "MU N MUL",
            "dynamics",
            ["friction", "contact_force"],
            "friction coefficient times normal force contact surface sliding",
        ),
        (
            "reality_dynamics_elastic_collision_1d",
            "Elastic Collision (1D)",
            "M1 U1 MUL M2 U2 MUL ADD M1 M2 ADD DIV",
            "dynamics_collision",
            ["collision", "elastic", "conservation"],
            "elastic collision one dimensional momentum conservation kinetic energy conserved",
        ),
        (
            "reality_dynamics_inelastic_collision_1d",
            "Inelastic Collision (1D)",
            "M1 U1 MUL M2 U2 MUL ADD M1 M2 ADD DIV",
            "dynamics_collision",
            ["collision", "inelastic", "conservation"],
            "inelastic collision one dimensional combined mass shared velocity momentum conserved",
        ),
        (
            "reality_dynamics_center_of_mass_velocity",
            "Center of Mass Velocity",
            "M1 V1 MUL M2 V2 MUL ADD M1 M2 ADD DIV",
            "dynamics_collision",
            ["center_of_mass", "momentum", "system"],
            "center of mass velocity total momentum divided by total mass system motion",
        ),
    ]
    out: list[dict[str, Any]] = []
    for entry_id, name, program, category, tags, query_anchor in specs:
        out.append(
            _entry(
                entry_id=entry_id,
                name=name,
                category=category,
                rpn_program=program,
                tags=tags,
                metadata={
                    "fundamental_law": category == "dynamics",
                    "cross_modal": ["math"],
                    "query_anchor": query_anchor,
                    "canonical_family": "physics_dynamics",
                },
            )
        )
    return out


def create_electromagnetism_primitives() -> list[dict[str, Any]]:
    specs = [
        (
            "reality_em_coulomb_force",
            "Coulomb Force",
            "K Q1 MUL Q2 MUL R R MUL DIV",
            ["electromagnetism", "inverse_square", "charge"],
            "coulomb force electric charges inverse square separation distance",
        ),
        (
            "reality_em_electric_field_point",
            "Electric Field (Point Charge)",
            "K Q MUL R R MUL DIV",
            ["electromagnetism", "field", "charge"],
            "electric field point charge strength over distance squared field lines",
        ),
        (
            "reality_em_electric_potential_point",
            "Electric Potential (Point Charge)",
            "K Q MUL R DIV",
            ["electromagnetism", "potential", "charge"],
            "electric potential point charge work per unit charge voltage",
        ),
        (
            "reality_em_ohm_voltage",
            "Ohm Voltage",
            "I R MUL",
            ["electromagnetism", "circuits", "ohm"],
            "ohm law voltage current times resistance circuit",
        ),
        (
            "reality_em_ohm_current",
            "Ohm Current",
            "V R DIV",
            ["electromagnetism", "circuits", "ohm"],
            "ohm law current equals voltage divided by resistance circuit",
        ),
        (
            "reality_em_capacitor_charge",
            "Capacitor Charge",
            "C V MUL",
            ["electromagnetism", "circuits", "capacitor"],
            "capacitor charge capacitance times voltage stored electric charge",
        ),
        (
            "reality_em_lorentz_force",
            "Lorentz Force",
            "Q V MUL B MUL THETA SIN MUL",
            ["electromagnetism", "magnetic", "force"],
            "lorentz force charge moving in magnetic field velocity cross magnetic field",
        ),
        (
            "reality_em_gauss_law",
            "Gauss Law",
            "Q_ENC EPSILON0 DIV",
            ["electromagnetism", "flux", "gauss"],
            "gauss law electric flux enclosed charge divided by epsilon zero",
        ),
        (
            "reality_em_faraday_law",
            "Faraday Law",
            "DELTA_FLUX DELTA_T DIV NEG",
            ["electromagnetism", "induction", "faraday"],
            "faraday law induced emf equals negative rate of change of magnetic flux",
        ),
        (
            "reality_em_ampere_law",
            "Ampere Law",
            "MU0 I_ENC MUL",
            ["electromagnetism", "magnetic_field", "ampere"],
            "ampere law circulation magnetic field enclosed current",
        ),
        (
            "reality_em_resistor_power",
            "Resistor Power",
            "I I MUL R MUL",
            ["electromagnetism", "circuits", "power"],
            "electric power in resistor i squared r circuit dissipation",
        ),
        (
            "reality_em_series_resistance",
            "Series Resistance",
            "R1 R2 ADD",
            ["electromagnetism", "circuits", "series"],
            "series resistance add resistors in a single current path",
        ),
        (
            "reality_em_parallel_resistance",
            "Parallel Resistance",
            "R1 R2 MUL R1 R2 ADD DIV",
            ["electromagnetism", "circuits", "parallel"],
            "parallel resistance reciprocal sum equivalent resistance branch circuits",
        ),
    ]
    return [
        _entry(
            entry_id=entry_id,
            name=name,
            category="electromagnetism",
            rpn_program=program,
            tags=tags,
            metadata={
                "fundamental_law": True,
                "cross_modal": ["drawing", "math"] if "field" in tags or "magnetic" in tags else ["math"],
                "query_anchor": query_anchor,
                "canonical_family": "physics_electromagnetism",
            },
        )
        for entry_id, name, program, tags, query_anchor in specs
    ]


def create_thermodynamics_primitives() -> list[dict[str, Any]]:
    specs = [
        (
            "reality_thermo_ideal_gas_pressure",
            "Ideal Gas Pressure",
            "N R_GAS MUL T MUL V DIV",
            ["thermodynamics", "ideal_gas", "pressure"],
            "ideal gas law pressure number of moles gas constant temperature volume",
        ),
        (
            "reality_thermo_ideal_gas_volume",
            "Ideal Gas Volume",
            "N R_GAS MUL T MUL P DIV",
            ["thermodynamics", "ideal_gas", "volume"],
            "ideal gas law volume number of moles gas constant temperature pressure",
        ),
        (
            "reality_thermo_ideal_gas_temperature",
            "Ideal Gas Temperature",
            "P V MUL N R_GAS MUL DIV",
            ["thermodynamics", "ideal_gas", "temperature"],
            "ideal gas law temperature pressure volume moles gas constant",
        ),
        (
            "reality_thermo_heat_capacity",
            "Heat Capacity",
            "Q DELTA_T DIV",
            ["thermodynamics", "heat", "capacity"],
            "heat capacity heat divided by temperature change thermal response",
        ),
        (
            "reality_thermo_thermal_energy",
            "Thermal Energy",
            "M C_P MUL DELTA_T MUL",
            ["thermodynamics", "heat", "specific_heat"],
            "thermal energy mass specific heat temperature change heating",
        ),
        (
            "reality_thermo_entropy_change",
            "Entropy Change",
            "Q_REV T DIV",
            ["thermodynamics", "entropy", "reversible"],
            "entropy change reversible heat divided by temperature disorder state function",
        ),
        (
            "reality_thermo_first_law",
            "First Law of Thermodynamics",
            "Q W SUB",
            ["thermodynamics", "energy_conservation", "first_law"],
            "first law thermodynamics internal energy equals heat minus work",
        ),
        (
            "reality_thermo_carnot_efficiency",
            "Carnot Efficiency",
            "1 TCOLD THOT DIV SUB",
            ["thermodynamics", "heat_engine", "efficiency"],
            "carnot efficiency one minus cold over hot temperature ideal heat engine",
        ),
        (
            "reality_thermo_phase_change_heat",
            "Phase Change Heat",
            "M L MUL",
            ["thermodynamics", "latent_heat", "phase_change"],
            "phase change latent heat mass times latent heat melting boiling",
        ),
        (
            "reality_thermo_conduction_rate",
            "Thermal Conduction Rate",
            "K A MUL DELTA_T MUL L DIV",
            ["thermodynamics", "heat_transfer", "conduction"],
            "thermal conduction rate conductivity area temperature difference length",
        ),
    ]
    return [
        _entry(
            entry_id=entry_id,
            name=name,
            category="thermodynamics",
            rpn_program=program,
            tags=tags,
            metadata={
                "cross_modal": ["math"],
                "query_anchor": query_anchor,
                "canonical_family": "physics_thermodynamics",
            },
        )
        for entry_id, name, program, tags, query_anchor in specs
    ]


def create_procedural_system_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        _entry(
            entry_id="reality_proc_lsystem_expand",
            name="L-System Expansion Step",
            category="procedural_generation",
            rpn_program="AXIOM RULES ITER EXPAND_LSYSTEM",
            tags=["procedural", "lsystem", "generative"],
            metadata={"generative": True, "cross_modal": ["drawing", "grammar"]},
        )
    )
    out.append(
        _entry(
            entry_id="reality_proc_game_of_life_step",
            name="Cellular Automata Step (Life)",
            category="procedural_generation",
            rpn_program="GRID NEIGHBORS_COUNT APPLY_LIFE_RULES",
            tags=["procedural", "cellular_automata", "generative"],
            metadata={"generative": True, "cross_modal": ["drawing", "math"]},
        )
    )
    out.append(
        _entry(
            entry_id="reality_proc_mandelbrot_iteration",
            name="Mandelbrot Iteration",
            category="procedural_generation",
            rpn_program="Z DUP MUL C ADD",
            tags=["procedural", "fractal", "generative"],
            metadata={"generative": True, "cross_modal": ["drawing", "math"]},
        )
    )
    for depth in range(1, 26):
        out.append(
            _entry(
                entry_id=f"reality_proc_lsystem_tree_depth_{depth}",
                name=f"L-System Tree depth={depth}",
                category="procedural_generation_lsystem",
                rpn_program=f"F RULE_TREE {depth} EXPAND_LSYSTEM",
                tags=["procedural", "lsystem", "tree"],
                metadata={"generative": True, "cross_modal": ["drawing", "grammar"]},
            )
        )
    for rule in range(18, 111):
        out.append(
            _entry(
                entry_id=f"reality_proc_eca_rule_{rule}",
                name=f"Elementary Cellular Automata Rule {rule}",
                category="procedural_generation_ca",
                rpn_program=f"GRID RULE_{rule} APPLY_ECA",
                tags=["procedural", "cellular_automata", "rule"],
                metadata={"generative": True, "cross_modal": ["drawing", "math"]},
            )
        )
    for octave in range(1, 11):
        for persistence in (0.35, 0.5, 0.65, 0.8):
            out.append(
                _entry(
                    entry_id=f"reality_proc_noise_o{octave}_p{str(persistence).replace('.', 'p')}",
                    name=f"Fractal Noise oct={octave} p={persistence}",
                    category="procedural_generation_noise",
                    rpn_program=f"X Y {octave} {persistence} FRACTAL_NOISE2D",
                    tags=["procedural", "noise", "terrain"],
                    metadata={"generative": True, "cross_modal": ["drawing", "math"]},
                )
            )
    for feed in (0.02, 0.03, 0.04, 0.05):
        for kill in (0.045, 0.05, 0.055, 0.06):
            out.append(
                _entry(
                    entry_id=f"reality_proc_reaction_diffusion_f{str(feed).replace('.', 'p')}_k{str(kill).replace('.', 'p')}",
                    name=f"Reaction Diffusion f={feed} k={kill}",
                    category="procedural_generation_reaction_diffusion",
                    rpn_program=f"GRID_A GRID_B {feed} {kill} RD_STEP",
                    tags=["procedural", "reaction_diffusion", "pattern"],
                    metadata={"generative": True, "cross_modal": ["drawing", "math"]},
                )
            )
    return out


def default_reality_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    entries.extend(create_kinematics_primitives())
    entries.extend(create_dynamics_primitives())
    entries.extend(create_electromagnetism_primitives())
    entries.extend(create_thermodynamics_primitives())
    entries.extend(create_procedural_system_primitives())
    return entries


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def bootstrap_reality_galaxy(storage_root: str | Path = "../Knowledge3D.local") -> dict[str, int]:
    """Append deterministic Reality entries without resetting existing data."""
    galaxies_root = Path(storage_root) / "galaxies"
    path = galaxies_root / "Reality.jsonl"
    existing = _read_jsonl(path)
    existing_ids = {str(row.get("id", "")) for row in existing}
    generated = default_reality_entries()
    to_append = [row for row in generated if str(row.get("id", "")) and str(row.get("id", "")) not in existing_ids]
    if to_append:
        _append_jsonl(path, to_append)
    return {
        "before": len(existing),
        "generated": len(generated),
        "appended": len(to_append),
        "after": len(existing) + len(to_append),
    }
