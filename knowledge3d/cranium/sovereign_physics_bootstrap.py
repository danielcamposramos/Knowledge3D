"""Foundational star scaffolds for the sovereign rigid-body physics path.

These helpers keep physics knowledge aligned with the vocabulary specs:

- constants/materials/bodies are Layer-2 reality substrate
- force laws are Layer-3 Grammar programs
- sleep policies are Layer-4 meta-rules
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_physical_constant_stars() -> List[Dict[str, Any]]:
    return [
        {
            "star_id": "physics_constant_gravitational",
            "facet": "physical_constant",
            "symbol": "G",
            "value_f64_hi": 6.67430e-11,
            "value_f64_lo": 0.0,
            "si_units": "m^3 kg^-1 s^-2",
            "uncertainty": 1.5e-15,
            "surface_forms": {"en": "gravitational constant", "pt": "constante gravitacional"},
        },
        {
            "star_id": "physics_constant_speed_of_light",
            "facet": "physical_constant",
            "symbol": "c",
            "value_f64_hi": 299792458.0,
            "value_f64_lo": 0.0,
            "si_units": "m s^-1",
            "uncertainty": 0.0,
            "surface_forms": {"en": "speed of light", "pt": "velocidade da luz"},
        },
        {
            "star_id": "physics_constant_reduced_planck",
            "facet": "physical_constant",
            "symbol": "ħ",
            "value_f64_hi": 1.054571817e-34,
            "value_f64_lo": 0.0,
            "si_units": "J s",
            "uncertainty": 0.0,
            "surface_forms": {"en": "reduced Planck constant", "pt": "constante reduzida de Planck"},
        },
        {
            "star_id": "physics_constant_boltzmann",
            "facet": "physical_constant",
            "symbol": "k_B",
            "value_f64_hi": 1.380649e-23,
            "value_f64_lo": 0.0,
            "si_units": "J K^-1",
            "uncertainty": 0.0,
            "surface_forms": {"en": "Boltzmann constant", "pt": "constante de Boltzmann"},
        },
        {
            "star_id": "physics_constant_vacuum_permittivity",
            "facet": "physical_constant",
            "symbol": "ε₀",
            "value_f64_hi": 8.8541878128e-12,
            "value_f64_lo": 0.0,
            "si_units": "F m^-1",
            "uncertainty": 1.3e-21,
            "surface_forms": {"en": "vacuum permittivity", "pt": "permissividade do vácuo"},
        },
        {
            "star_id": "physics_constant_vacuum_permeability",
            "facet": "physical_constant",
            "symbol": "μ₀",
            "value_f64_hi": 1.25663706212e-6,
            "value_f64_lo": 0.0,
            "si_units": "N A^-2",
            "uncertainty": 1.9e-16,
            "surface_forms": {"en": "vacuum permeability", "pt": "permeabilidade do vácuo"},
        },
        {
            "star_id": "physics_constant_stefan_boltzmann",
            "facet": "physical_constant",
            "symbol": "σ",
            "value_f64_hi": 5.670374419e-8,
            "value_f64_lo": 0.0,
            "si_units": "W m^-2 K^-4",
            "uncertainty": 0.0,
            "surface_forms": {"en": "Stefan-Boltzmann constant", "pt": "constante de Stefan-Boltzmann"},
        },
        {
            "star_id": "physics_constant_avogadro",
            "facet": "physical_constant",
            "symbol": "N_A",
            "value_f64_hi": 6.02214076e23,
            "value_f64_lo": 0.0,
            "si_units": "mol^-1",
            "uncertainty": 0.0,
            "surface_forms": {"en": "Avogadro constant", "pt": "constante de Avogadro"},
        },
        {
            "star_id": "physics_constant_elementary_charge",
            "facet": "physical_constant",
            "symbol": "e",
            "value_f64_hi": 1.602176634e-19,
            "value_f64_lo": 0.0,
            "si_units": "C",
            "uncertainty": 0.0,
            "surface_forms": {"en": "elementary charge", "pt": "carga elementar"},
        },
        {
            "star_id": "physics_constant_electron_mass",
            "facet": "physical_constant",
            "symbol": "m_e",
            "value_f64_hi": 9.1093837015e-31,
            "value_f64_lo": 0.0,
            "si_units": "kg",
            "uncertainty": 2.8e-40,
            "surface_forms": {"en": "electron mass", "pt": "massa do elétron"},
        },
        {
            "star_id": "physics_constant_proton_mass",
            "facet": "physical_constant",
            "symbol": "m_p",
            "value_f64_hi": 1.67262192369e-27,
            "value_f64_lo": 0.0,
            "si_units": "kg",
            "uncertainty": 5.1e-37,
            "surface_forms": {"en": "proton mass", "pt": "massa do próton"},
        },
    ]


def build_physics_material_stars() -> List[Dict[str, Any]]:
    return [
        {
            "star_id": "physics_material_steel",
            "facet": "physical_material",
            "density": 7850.0,
            "restitution": 0.25,
            "friction_static": 0.74,
            "friction_dynamic": 0.57,
            "young_modulus": 2.0e11,
            "poisson_ratio": 0.29,
            "visual_rpn_addr": "material_visual_steel_default",
            "base_material_star_id": None,
        },
        {
            "star_id": "physics_material_wood",
            "facet": "physical_material",
            "density": 700.0,
            "restitution": 0.35,
            "friction_static": 0.45,
            "friction_dynamic": 0.30,
            "young_modulus": 1.1e10,
            "poisson_ratio": 0.35,
            "visual_rpn_addr": "material_visual_wood_default",
            "base_material_star_id": None,
        },
        {
            "star_id": "physics_material_rubber",
            "facet": "physical_material",
            "density": 1100.0,
            "restitution": 0.8,
            "friction_static": 1.1,
            "friction_dynamic": 0.9,
            "young_modulus": 1.0e7,
            "poisson_ratio": 0.49,
            "visual_rpn_addr": "material_visual_rubber_default",
            "base_material_star_id": None,
        },
        {
            "star_id": "physics_material_ice",
            "facet": "physical_material",
            "density": 917.0,
            "restitution": 0.05,
            "friction_static": 0.1,
            "friction_dynamic": 0.03,
            "young_modulus": 9.0e9,
            "poisson_ratio": 0.33,
            "visual_rpn_addr": "material_visual_ice_default",
            "base_material_star_id": None,
        },
    ]


def build_default_gravity_force_law() -> Dict[str, Any]:
    return {
        "star_id": "physics_law_default_gravity",
        "facet": "force_law",
        "layer": 3,
        "physics_rpn_addr": "LOAD_STAR physics_constant_gravitational PH_GRAVITY_APPLY PH_INTEGRATE",
        "summary": "Default gravity-coupled rigid-body step using Reality Galaxy constant fetch.",
    }


def build_default_sleep_policy() -> Dict[str, Any]:
    return {
        "star_id": "physics_meta_sleep_policy_default",
        "facet": "meta_rule",
        "layer": 4,
        "energy_sleep_threshold": 0.001,
        "frames_before_sleep": 60,
        "wake_impulse_threshold": 0.01,
        "strategy_rpn_addr": "PH_SLEEP_CHECK PH_TERNARY_CLASSIFY",
    }


def serialize_material_table() -> List[Dict[str, float | int]]:
    """Convert physics material stars to the compact GPU lookup format.

    The numeric `star_id` values here must match the `galaxy_handles.x` values
    used by the sovereign runtime when bodies are spawned or imported.
    """
    materials = build_physics_material_stars()
    id_map = {
        "physics_material_steel": 1,
        "physics_material_wood": 2,
        "physics_material_rubber": 3,
        "physics_material_ice": 4,
    }
    result: List[Dict[str, float | int]] = []
    for material in materials:
        result.append(
            {
                "star_id": int(id_map.get(str(material["star_id"]), 0)),
                "friction": float(material["friction_dynamic"]),
                "restitution": float(material["restitution"]),
                "density": float(material["density"]),
                "texture_id": 0xFFFFFFFF,
            }
        )
    return result


__all__ = [
    "build_physical_constant_stars",
    "build_physics_material_stars",
    "build_default_gravity_force_law",
    "build_default_sleep_policy",
    "serialize_material_table",
]
