"""Multi-domain integration scenarios for the Reality Enabler."""

from __future__ import annotations

from typing import Dict, List, Optional

from knowledge3d.cranium.reality_nodes import RealitySystem
from knowledge3d.cranium.reality_physics_export import (
    export_acid_base_reaction,
    export_combustion,
    export_composite_material,
    export_crystal_lattice,
    export_heat_1d,
    export_heat_2d,
    export_metal_melting,
    export_population_dynamics,
    export_simple_cell,
    export_enzyme_kinetics,
    export_ideal_gas,
    export_phase_transition_water,
)


def export_cell_metabolism_scenario(params: Optional[Dict] = None) -> RealitySystem:
    """Cell metabolism: enzyme + diffusion + heat + pH buffer."""
    p = params or {}
    component_refs: List[str] = p.get("component_refs") or [
        "system:enzyme_kinetics",
        "system:simple_cell",
        "system:heat_1d",
        "system:acid_base",
    ]
    return RealitySystem(
        node_id="system:scenario_cell_metabolism",
        component_refs=component_refs,
        state={
            "enzyme_activity": float(p.get("enzyme_activity", 1.0)),
            "temp_K": float(p.get("temp_K", 310.0)),
            "pH": float(p.get("pH", 7.4)),
            "buffer_strength": float(p.get("buffer_strength", 0.1)),
        },
        behavior_rpn="""
            enzyme_activity RECALL 0.1 * temp_K RECALL + temp_K STORE
            pH RECALL 7.0 - abs buffer_strength RECALL * enzyme_activity RECALL - enzyme_activity STORE
            pH RECALL 7.4 0.1 tquant pH_state STORE
        """,
        law_rpn="temp_K RECALL 250 ge",
        rpn_tier=1,
        rpn_instance=None,
        matryoshka_dim=128,
    )


def export_material_synthesis_scenario(params: Optional[Dict] = None) -> RealitySystem:
    """Combustion-driven metal melting and lattice formation."""
    p = params or {}
    component_refs: List[str] = p.get("component_refs") or [
        "system:combustion_ch4",
        "system:heat_2d",
        "system:metal_melting",
        "system:crystal_lattice",
    ]
    return RealitySystem(
        node_id="system:scenario_material_synthesis",
        component_refs=component_refs,
        state={
            "core_temp": float(p.get("core_temp", 300.0)),
            "combustion_energy": float(p.get("combustion_energy", 0.0)),
            "phase": float(p.get("phase", -1.0)),
            "lattice_a": float(p.get("lattice_a", 3.61e-10)),
        },
        behavior_rpn="""
            combustion_energy RECALL 0.001 * core_temp RECALL + core_temp STORE
            core_temp RECALL 1356 tcmp phase STORE
            phase RECALL 1 eq lattice_a RECALL 1.001 * lattice_a STORE
        """,
        law_rpn="core_temp RECALL 0 gt",
        rpn_tier=1,
        rpn_instance=None,
        matryoshka_dim=128,
    )


def export_ecosystem_scenario(params: Optional[Dict] = None) -> RealitySystem:
    """Ecosystem: populations coupled with atmosphere and water state."""
    p = params or {}
    component_refs: List[str] = p.get("component_refs") or [
        "system:population_dynamics",
        "system:ideal_gas",
        "system:heat_1d",
        "system:water_phase",
    ]
    return RealitySystem(
        node_id="system:scenario_ecosystem",
        component_refs=component_refs,
        state={
            "resources": float(p.get("resources", 1.0)),
            "atmosphere_CO2": float(p.get("atmosphere_CO2", 400.0)),
            "water_phase": float(p.get("water_phase", 0.0)),  # -1 ice, 0 liquid, +1 vapor
        },
        behavior_rpn="""
            resources RECALL 0.01 + resources STORE
            atmosphere_CO2 RECALL 0.99 * atmosphere_CO2 STORE
            water_phase RECALL 0.1 + water_phase STORE
        """,
        law_rpn="resources RECALL 0 gt",
        rpn_tier=1,
        rpn_instance=None,
        matryoshka_dim=128,
    )


__all__ = [
    "export_cell_metabolism_scenario",
    "export_material_synthesis_scenario",
    "export_ecosystem_scenario",
]
