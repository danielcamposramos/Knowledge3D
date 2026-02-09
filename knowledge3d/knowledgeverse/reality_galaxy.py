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
    base_meta = {
        "source": "week19_reality_bootstrap",
        "bootstrap": "week19_reality_enabler",
        "procedural": True,
    }
    if metadata:
        base_meta.update(metadata)
    return {
        "id": entry_id,
        "name": name,
        "domain": "reality",
        "category": category,
        "rpn_program": rpn_program,
        "tags": tags or [],
        "metadata": base_meta,
    }


def create_kinematics_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        _entry(
            entry_id="reality_kinematics_position_update_euler",
            name="Position Update (Euler)",
            category="kinematics",
            rpn_program="X V DT MUL ADD",
            tags=["kinematics", "integration", "euler"],
            metadata={"cross_modal": ["drawing", "math"]},
        )
    )
    out.append(
        _entry(
            entry_id="reality_kinematics_velocity_update_euler",
            name="Velocity Update (Euler)",
            category="kinematics",
            rpn_program="V A DT MUL ADD",
            tags=["kinematics", "integration", "euler"],
            metadata={"cross_modal": ["math"]},
        )
    )
    out.append(
        _entry(
            entry_id="reality_kinematics_projectile_2d",
            name="Projectile Motion (2D)",
            category="kinematics",
            rpn_program="X VX DT MUL ADD Y VY DT MUL ADD 0.5 G MUL DT DT MUL MUL SUB",
            tags=["kinematics", "projectile", "composed"],
            metadata={"cross_modal": ["drawing", "math"], "composition_depth": 3, "generative": True},
        )
    )
    for accel in range(-24, 25):
        out.append(
            _entry(
                entry_id=f"reality_kinematics_velocity_a_{accel:+d}",
                name=f"Velocity Update a={accel}",
                category="kinematics",
                rpn_program=f"V0 {accel} DT MUL ADD",
                tags=["kinematics", "velocity"],
                metadata={"cross_modal": ["math"]},
            )
        )
    for accel in range(-16, 17):
        for t_step in (0.25, 0.5, 1.0, 2.0):
            out.append(
                _entry(
                    entry_id=f"reality_kinematics_position_a_{accel:+d}_dt_{str(t_step).replace('.', 'p')}",
                    name=f"Position Update a={accel} dt={t_step}",
                    category="kinematics",
                    rpn_program=f"X0 V0 {t_step} MUL ADD 0.5 {accel} MUL {t_step} {t_step} MUL MUL ADD",
                    tags=["kinematics", "position", "motion"],
                    metadata={"cross_modal": ["drawing", "math"]},
                )
            )
    for angle in range(5, 90, 5):
        for speed in (10, 20, 30, 40):
            out.append(
                _entry(
                    entry_id=f"reality_kinematics_projectile_angle_{angle}_speed_{speed}",
                    name=f"Projectile angle={angle} speed={speed}",
                    category="kinematics",
                    rpn_program=f"{speed} {angle} DEG2RAD COS MUL VX_STORE {speed} {angle} DEG2RAD SIN MUL VY_STORE",
                    tags=["kinematics", "projectile", "launch"],
                    metadata={"cross_modal": ["drawing", "math"], "generative": True},
                )
            )
    return out


def create_dynamics_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    laws = [
        ("newton_second_law", "F M DIV", ["fundamental", "dynamics"]),
        ("kinetic_energy", "0.5 M MUL V V MUL MUL", ["energy", "dynamics"]),
        ("potential_energy_gravity", "M G MUL H MUL", ["energy", "gravity"]),
        ("momentum_linear", "M V MUL", ["momentum", "dynamics"]),
        ("impulse", "F DELTA_T MUL", ["impulse", "dynamics"]),
        ("work", "F D MUL", ["work", "energy"]),
        ("power", "WORK TIME DIV", ["power", "energy"]),
        ("energy_conservation", "KE1 PE1 ADD PE2 SUB", ["conservation", "energy"]),
        ("momentum_conservation_1d", "M1 V1 MUL M2 V2 MUL ADD", ["conservation", "momentum"]),
    ]
    for law_id, program, tags in laws:
        out.append(
            _entry(
                entry_id=f"reality_dynamics_{law_id}",
                name=law_id.replace("_", " ").title(),
                category="dynamics",
                rpn_program=program,
                tags=tags,
                metadata={"fundamental_law": True, "cross_modal": ["math"]},
            )
        )
    for mass in range(1, 41):
        for velocity in range(1, 13):
            out.append(
                _entry(
                    entry_id=f"reality_dynamics_ke_m{mass}_v{velocity}",
                    name=f"Kinetic Energy m={mass} v={velocity}",
                    category="dynamics_templates",
                    rpn_program=f"0.5 {mass} MUL {velocity} {velocity} MUL MUL",
                    tags=["energy", "template"],
                    metadata={"cross_modal": ["math"]},
                )
            )
    for m1 in range(1, 16):
        for m2 in range(1, 16):
            out.append(
                _entry(
                    entry_id=f"reality_dynamics_collision_1d_m{m1}_{m2}",
                    name=f"1D Collision Template m1={m1} m2={m2}",
                    category="dynamics_collision",
                    rpn_program=f"{m1} U1 MUL {m2} U2 MUL ADD {m1} {m2} ADD DIV",
                    tags=["collision", "conservation", "template"],
                    metadata={"cross_modal": ["math"], "generative": True},
                )
            )
    return out


def create_electromagnetism_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    core = [
        ("coulomb_force", "K Q1 MUL Q2 MUL R R MUL DIV", ["electromagnetism", "inverse_square"]),
        ("electric_field_point", "K Q MUL R R MUL DIV", ["electromagnetism", "field"]),
        ("electric_potential_point", "K Q MUL R DIV", ["electromagnetism", "potential"]),
        ("ohm_voltage", "I R MUL", ["electromagnetism", "circuits"]),
        ("ohm_current", "V R DIV", ["electromagnetism", "circuits"]),
        ("capacitor_charge", "C V MUL", ["electromagnetism", "circuits"]),
    ]
    for cid, program, tags in core:
        out.append(
            _entry(
                entry_id=f"reality_em_{cid}",
                name=cid.replace("_", " ").title(),
                category="electromagnetism",
                rpn_program=program,
                tags=tags,
                metadata={"fundamental_law": True, "cross_modal": ["drawing", "math"]},
            )
        )
    for q in range(1, 16):
        for r in range(1, 11):
            out.append(
                _entry(
                    entry_id=f"reality_em_field_q{q}_r{r}",
                    name=f"Electric Field q={q} r={r}",
                    category="electromagnetism_field_templates",
                    rpn_program=f"K {q} MUL {r} {r} MUL DIV",
                    tags=["electromagnetism", "field", "template"],
                    metadata={"cross_modal": ["drawing", "math"], "generative": True},
                )
            )
    for i in range(1, 13):
        for resistance in (1, 2, 5, 10, 20, 50):
            out.append(
                _entry(
                    entry_id=f"reality_em_ohm_i{i}_r{resistance}",
                    name=f"Ohm Template i={i} r={resistance}",
                    category="electromagnetism_circuit_templates",
                    rpn_program=f"{i} {resistance} MUL",
                    tags=["circuits", "ohm", "template"],
                    metadata={"cross_modal": ["math"]},
                )
            )
    return out


def create_thermodynamics_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base = [
        ("ideal_gas_pressure", "N R_GAS MUL T MUL V DIV", ["thermodynamics", "ideal_gas"]),
        ("ideal_gas_volume", "N R_GAS MUL T MUL P DIV", ["thermodynamics", "ideal_gas"]),
        ("heat_capacity", "Q DELTA_T DIV", ["thermodynamics", "heat"]),
        ("thermal_energy", "M C_P MUL DELTA_T MUL", ["thermodynamics", "heat"]),
        ("entropy_change", "Q_REV T DIV", ["thermodynamics", "entropy"]),
    ]
    for tid, program, tags in base:
        out.append(
            _entry(
                entry_id=f"reality_thermo_{tid}",
                name=tid.replace("_", " ").title(),
                category="thermodynamics",
                rpn_program=program,
                tags=tags,
                metadata={"cross_modal": ["math"]},
            )
        )
    for n in range(1, 21):
        for t in (273, 300, 350, 400, 450, 500):
            out.append(
                _entry(
                    entry_id=f"reality_thermo_ideal_n{n}_t{t}",
                    name=f"Ideal Gas Template n={n} T={t}",
                    category="thermodynamics_templates",
                    rpn_program=f"{n} R_GAS MUL {t} MUL V DIV",
                    tags=["ideal_gas", "template"],
                    metadata={"cross_modal": ["math"], "generative": True},
                )
            )
    for cp in (0.5, 1.0, 2.0, 4.0):
        for dt in (5, 10, 25, 50, 100):
            out.append(
                _entry(
                    entry_id=f"reality_thermo_heat_cp_{str(cp).replace('.', 'p')}_dt_{dt}",
                    name=f"Heat Transfer cp={cp} dt={dt}",
                    category="thermodynamics_heat_templates",
                    rpn_program=f"M {cp} MUL {dt} MUL",
                    tags=["heat", "template"],
                    metadata={"cross_modal": ["math"]},
                )
            )
    return out


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
