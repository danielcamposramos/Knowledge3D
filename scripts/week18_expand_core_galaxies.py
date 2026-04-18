#!/usr/bin/env python3
"""Expand core Knowledgeverse galaxies with deterministic foundational entries.

This script appends deduplicated JSONL entries into:
- Math.jsonl
- Audio.jsonl
- Reality.jsonl

The generated entries are procedural RPN-like templates designed for ingestion
and retrieval paths (not direct hot-path execution kernels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _new_entry(
    *,
    entry_id: str,
    name: str,
    domain: str,
    category: str,
    rpn_program: str,
    source: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": domain,
        "category": category,
        "rpn_program": rpn_program,
        "tags": tags or [],
        "metadata": {
            "source": source,
            "bootstrap": "week18_core_expansion",
        },
    }


def _build_math_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    # Arithmetic/identity primitives.
    ops = [
        ("add", "A B ADD"),
        ("sub", "A B SUB"),
        ("mul", "A B MUL"),
        ("div", "A B DIV"),
        ("pow", "A N POW"),
        ("sqrt", "A SQRT"),
        ("abs", "A ABS"),
        ("mod", "A B MOD"),
        ("sin", "X SIN"),
        ("cos", "X COS"),
        ("tan", "X TAN"),
        ("exp", "X EXP"),
        ("log", "X LOG"),
        ("floor", "X FLOOR"),
        ("ceil", "X CEIL"),
        ("round", "X ROUND"),
    ]
    for op_name, program in ops:
        entries.append(
            _new_entry(
                entry_id=f"math_op_{op_name}",
                name=f"Math Operation {op_name.upper()}",
                domain="math",
                category="operator",
                rpn_program=program,
                source="week18_generated_math",
                tags=["operator", "foundation"],
            )
        )

    # Derivative patterns for powers up to 64.
    for n in range(1, 65):
        entries.append(
            _new_entry(
                entry_id=f"math_derivative_power_{n}",
                name=f"Derivative of x^{n}",
                domain="math",
                category="calculus_derivative",
                rpn_program=f"{n} X {n-1} POW MUL",
                source="week18_generated_math",
                tags=["calculus", "derivative", "power_rule"],
            )
        )

    # Integral patterns for powers up to 64.
    for n in range(0, 65):
        denom = n + 1
        entries.append(
            _new_entry(
                entry_id=f"math_integral_power_{n}",
                name=f"Integral of x^{n}",
                domain="math",
                category="calculus_integral",
                rpn_program=f"X {denom} POW {denom} DIV",
                source="week18_generated_math",
                tags=["calculus", "integral", "power_rule"],
            )
        )

    # Linear equation solver templates: a*x + b = c.
    for a in range(1, 41):
        for b in range(0, 11):
            entries.append(
                _new_entry(
                    entry_id=f"math_linear_solve_a{a}_b{b}",
                    name=f"Solve linear equation a*x+b=c (a={a}, b={b})",
                    domain="math",
                    category="algebra_linear",
                    rpn_program=f"C {b} SUB {a} DIV",
                    source="week18_generated_math",
                    tags=["algebra", "linear_equation"],
                )
            )

    # Quadratic discriminant templates.
    for a in range(1, 11):
        for b in range(1, 11):
            for c in range(1, 4):
                entries.append(
                    _new_entry(
                        entry_id=f"math_quadratic_discriminant_a{a}_b{b}_c{c}",
                        name=f"Quadratic discriminant for (a={a}, b={b}, c={c})",
                        domain="math",
                        category="algebra_quadratic",
                        rpn_program="B 2 POW 4 A MUL C MUL SUB",
                        source="week18_generated_math",
                        tags=["algebra", "quadratic", "discriminant"],
                    )
                )

    # Geometry formulas.
    geometry = [
        ("circle_area", "PI R 2 POW MUL"),
        ("circle_circumference", "2 PI MUL R MUL"),
        ("triangle_area", "BASE HEIGHT MUL 2 DIV"),
        ("rectangle_area", "WIDTH HEIGHT MUL"),
        ("rectangle_perimeter", "WIDTH HEIGHT ADD 2 MUL"),
        ("sphere_volume", "4 PI MUL R 3 POW MUL 3 DIV"),
        ("sphere_surface_area", "4 PI MUL R 2 POW MUL"),
        ("cylinder_volume", "PI R 2 POW MUL HEIGHT MUL"),
        ("distance_2d", "X2 X1 SUB 2 POW Y2 Y1 SUB 2 POW ADD SQRT"),
        ("distance_3d", "DX 2 POW DY 2 POW ADD DZ 2 POW ADD SQRT"),
    ]
    for gid, program in geometry:
        entries.append(
            _new_entry(
                entry_id=f"math_geometry_{gid}",
                name=f"Geometry formula {gid}",
                domain="math",
                category="geometry",
                rpn_program=program,
                source="week18_generated_math",
                tags=["geometry"],
            )
        )

    return entries


def _build_audio_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    # MIDI frequencies and core waveform synth templates.
    waveform_templates = [
        ("sine", "T F MUL TWO_PI MUL SIN"),
        ("square", "T F MUL TWO_PI MUL SIN SIGN"),
        ("triangle", "T F MUL TWO_PI MUL SIN ASIN TWO_DIV_PI MUL"),
        ("saw", "T F MUL DUP FLOOR SUB 2 MUL 1 SUB"),
    ]
    for midi in range(24, 109):
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        for wave_id, tpl in waveform_templates:
            entries.append(
                _new_entry(
                    entry_id=f"audio_wave_{wave_id}_midi_{midi}",
                    name=f"{wave_id} waveform MIDI {midi}",
                    domain="audio",
                    category="waveform",
                    rpn_program=f"{freq:.6f} F_STORE {tpl}",
                    source="week18_generated_audio",
                    tags=["waveform", "midi"],
                )
            )

    # ADSR envelope templates.
    adsr_profiles = [
        ("soft_pad", "0.30 0.40 0.70 0.60 ADSR"),
        ("pluck", "0.01 0.20 0.30 0.15 ADSR"),
        ("piano", "0.02 0.18 0.55 0.30 ADSR"),
        ("organ", "0.01 0.10 0.90 0.25 ADSR"),
        ("strings", "0.18 0.30 0.75 0.40 ADSR"),
    ]
    for pid, program in adsr_profiles:
        entries.append(
            _new_entry(
                entry_id=f"audio_adsr_{pid}",
                name=f"ADSR profile {pid}",
                domain="audio",
                category="envelope",
                rpn_program=program,
                source="week18_generated_audio",
                tags=["adsr", "envelope"],
            )
        )

    # Filters.
    filters = [
        ("lowpass_rc", "X ALPHA MUL Y_PREV ONE ALPHA SUB MUL ADD"),
        ("highpass_rc", "Y_PREV X ADD X_PREV SUB ALPHA MUL"),
        ("bandpass_basic", "X HIGHPASS LOWPASS"),
        ("notch_basic", "X BANDPASS SUB"),
        ("compressor_soft", "X THRESH ABOVE RATIO COMPRESS"),
        ("limiter_hard", "X LIMIT_CLAMP"),
    ]
    for fid, program in filters:
        entries.append(
            _new_entry(
                entry_id=f"audio_filter_{fid}",
                name=f"Audio filter {fid}",
                domain="audio",
                category="filter",
                rpn_program=program,
                source="week18_generated_audio",
                tags=["dsp", "filter"],
            )
        )

    return entries


def _build_reality_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    physics_formulas = [
        ("newton_second_law", "M A MUL"),
        ("kinetic_energy", "M V 2 POW MUL 2 DIV"),
        ("potential_energy", "M G MUL H MUL"),
        ("momentum", "M V MUL"),
        ("impulse", "F DELTA_T MUL"),
        ("work", "F D MUL"),
        ("power", "WORK TIME DIV"),
        ("ohms_law_v", "I R MUL"),
        ("ohms_law_i", "V R DIV"),
        ("ohms_law_r", "V I DIV"),
        ("coulomb_force", "K Q1 MUL Q2 MUL R 2 POW DIV"),
        ("wave_speed", "FREQ LAMBDA MUL"),
        ("pressure", "FORCE AREA DIV"),
        ("density", "MASS VOLUME DIV"),
        ("ideal_gas", "N R_GAS MUL T MUL V DIV"),
    ]
    for rid, program in physics_formulas:
        entries.append(
            _new_entry(
                entry_id=f"reality_physics_{rid}",
                name=f"Physics formula {rid}",
                domain="reality",
                category="physics",
                rpn_program=program,
                source="week18_generated_reality",
                tags=["physics"],
            )
        )

    # Kinematic templates across acceleration regimes.
    for accel in range(-10, 21):
        entries.append(
            _new_entry(
                entry_id=f"reality_kinematics_velocity_a{accel}",
                name=f"Kinematics velocity update a={accel}",
                domain="reality",
                category="kinematics",
                rpn_program=f"V0 {accel} T MUL ADD",
                source="week18_generated_reality",
                tags=["kinematics", "motion"],
            )
        )
        entries.append(
            _new_entry(
                entry_id=f"reality_kinematics_position_a{accel}",
                name=f"Kinematics position update a={accel}",
                domain="reality",
                category="kinematics",
                rpn_program=f"X0 V0 T MUL ADD 0.5 {accel} MUL T 2 POW MUL ADD",
                source="week18_generated_reality",
                tags=["kinematics", "motion"],
            )
        )

    # Mechanics parameter grids.
    for mass in range(1, 26):
        for velocity in range(1, 11):
            entries.append(
                _new_entry(
                    entry_id=f"reality_energy_mass{mass}_vel{velocity}",
                    name=f"Kinetic energy template m={mass}, v={velocity}",
                    domain="reality",
                    category="mechanics_template",
                    rpn_program=f"{mass} {velocity} 2 POW MUL 2 DIV",
                    source="week18_generated_reality",
                    tags=["mechanics", "energy"],
                )
            )

    # Basic chemistry stoichiometry templates.
    for coeff_a in range(1, 11):
        for coeff_b in range(1, 6):
            entries.append(
                _new_entry(
                    entry_id=f"reality_chem_balance_a{coeff_a}_b{coeff_b}",
                    name=f"Stoichiometry ratio a={coeff_a}, b={coeff_b}",
                    domain="reality",
                    category="chemistry",
                    rpn_program=f"MOLES_A {coeff_a} DIV {coeff_b} MUL",
                    source="week18_generated_reality",
                    tags=["chemistry", "stoichiometry"],
                )
            )

    return entries


def _merge_by_id(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_ids = {str(row.get("id", "")) for row in existing if isinstance(row, dict)}
    out: list[dict[str, Any]] = []
    for row in incoming:
        rid = str(row.get("id", ""))
        if not rid or rid in existing_ids:
            continue
        existing_ids.add(rid)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local",
        help="Knowledgeverse storage root (contains galaxies/*.jsonl)",
    )
    args = parser.parse_args()

    storage_root = Path(args.storage_root)
    galaxies_root = storage_root / "galaxies"
    galaxies_root.mkdir(parents=True, exist_ok=True)

    targets = [
        ("Math", _build_math_entries),
        ("Audio", _build_audio_entries),
        ("Reality", _build_reality_entries),
    ]

    summary: dict[str, dict[str, int]] = {}
    for galaxy_name, builder in targets:
        path = galaxies_root / f"{galaxy_name}.jsonl"
        existing = _read_jsonl(path)
        incoming = builder()
        to_append = _merge_by_id(existing, incoming)
        if to_append:
            _append_jsonl(path, to_append)
        summary[galaxy_name] = {
            "before": len(existing),
            "generated": len(incoming),
            "appended": len(to_append),
            "after": len(existing) + len(to_append),
        }

    print(json.dumps({"storage_root": str(storage_root), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
