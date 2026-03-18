"""Physical constants from CODATA/NIST-style registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstant:
    key: str
    name: str
    symbol: str
    value: float
    unit: str
    exact: bool = False


PHYSICAL_CONSTANTS: dict[str, PhysicalConstant] = {
    "speed_of_light": PhysicalConstant("speed_of_light", "speed of light", "c", 2.99792458e8, "m/s", True),
    "gravitational_constant": PhysicalConstant("gravitational_constant", "gravitational constant", "G", 6.67430e-11, "m^3/(kg·s^2)", False),
    "planck_constant": PhysicalConstant("planck_constant", "Planck constant", "h", 6.62607015e-34, "J·s", True),
    "boltzmann_constant": PhysicalConstant("boltzmann_constant", "Boltzmann constant", "k_B", 1.380649e-23, "J/K", True),
    "avogadro_number": PhysicalConstant("avogadro_number", "Avogadro number", "N_A", 6.02214076e23, "1/mol", True),
    "elementary_charge": PhysicalConstant("elementary_charge", "elementary charge", "e", 1.602176634e-19, "C", True),
    "vacuum_permittivity": PhysicalConstant("vacuum_permittivity", "vacuum permittivity", "ε₀", 8.8541878128e-12, "F/m", False),
    "vacuum_permeability": PhysicalConstant("vacuum_permeability", "vacuum permeability", "μ₀", 1.25663706212e-6, "N/A^2", False),
    "electron_mass": PhysicalConstant("electron_mass", "electron mass", "m_e", 9.1093837015e-31, "kg", False),
    "proton_mass": PhysicalConstant("proton_mass", "proton mass", "m_p", 1.67262192369e-27, "kg", False),
    "neutron_mass": PhysicalConstant("neutron_mass", "neutron mass", "m_n", 1.67492749804e-27, "kg", False),
    "fine_structure_constant": PhysicalConstant("fine_structure_constant", "fine-structure constant", "α", 7.2973525693e-3, "dimensionless", False),
    "gas_constant": PhysicalConstant("gas_constant", "ideal gas constant", "R", 8.314462618, "J/(mol·K)", True),
    "stefan_boltzmann": PhysicalConstant("stefan_boltzmann", "Stefan-Boltzmann constant", "σ", 5.670374419e-8, "W/(m^2·K^4)", False),
    "standard_gravity": PhysicalConstant("standard_gravity", "standard gravity", "g", 9.80665, "m/s^2", True),
    "standard_atmosphere": PhysicalConstant("standard_atmosphere", "standard atmosphere", "atm", 101325.0, "Pa", True),
}


def iter_physical_constants() -> list[PhysicalConstant]:
    return [PHYSICAL_CONSTANTS[key] for key in sorted(PHYSICAL_CONSTANTS.keys())]


__all__ = ["PHYSICAL_CONSTANTS", "PhysicalConstant", "iter_physical_constants"]
