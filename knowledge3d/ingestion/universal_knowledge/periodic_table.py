"""Periodic table and subatomic particle registries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementEntry:
    atomic_number: int
    symbol: str
    name_en: str
    atomic_mass: float
    category: str
    group: int | None
    period: int
    block: str
    state_at_stp: str
    electron_config: str = ""
    electronegativity: float | None = None
    density: float | None = None
    melting_point: float | None = None
    boiling_point: float | None = None
    discovery_year: int | None = None
    oxidation_states: tuple[int, ...] = field(default_factory=tuple)

    @property
    def surface_forms(self) -> dict[str, str]:
        return {
            "en": self.name_en,
            "pt": self.name_en,
            "la": self.name_en,
        }


_ELEMENT_ROWS = [
    (1, "H", "Hydrogen", 1.008, "nonmetal", 1, 1, "s", "gas"),
    (2, "He", "Helium", 4.002602, "noble_gas", 18, 1, "s", "gas"),
    (3, "Li", "Lithium", 6.94, "alkali_metal", 1, 2, "s", "solid"),
    (4, "Be", "Beryllium", 9.0121831, "alkaline_earth_metal", 2, 2, "s", "solid"),
    (5, "B", "Boron", 10.81, "metalloid", 13, 2, "p", "solid"),
    (6, "C", "Carbon", 12.011, "nonmetal", 14, 2, "p", "solid"),
    (7, "N", "Nitrogen", 14.007, "nonmetal", 15, 2, "p", "gas"),
    (8, "O", "Oxygen", 15.999, "nonmetal", 16, 2, "p", "gas"),
    (9, "F", "Fluorine", 18.998403163, "halogen", 17, 2, "p", "gas"),
    (10, "Ne", "Neon", 20.1797, "noble_gas", 18, 2, "p", "gas"),
    (11, "Na", "Sodium", 22.98976928, "alkali_metal", 1, 3, "s", "solid"),
    (12, "Mg", "Magnesium", 24.305, "alkaline_earth_metal", 2, 3, "s", "solid"),
    (13, "Al", "Aluminum", 26.9815385, "post_transition_metal", 13, 3, "p", "solid"),
    (14, "Si", "Silicon", 28.085, "metalloid", 14, 3, "p", "solid"),
    (15, "P", "Phosphorus", 30.973761998, "nonmetal", 15, 3, "p", "solid"),
    (16, "S", "Sulfur", 32.06, "nonmetal", 16, 3, "p", "solid"),
    (17, "Cl", "Chlorine", 35.45, "halogen", 17, 3, "p", "gas"),
    (18, "Ar", "Argon", 39.948, "noble_gas", 18, 3, "p", "gas"),
    (19, "K", "Potassium", 39.0983, "alkali_metal", 1, 4, "s", "solid"),
    (20, "Ca", "Calcium", 40.078, "alkaline_earth_metal", 2, 4, "s", "solid"),
    (21, "Sc", "Scandium", 44.955908, "transition_metal", 3, 4, "d", "solid"),
    (22, "Ti", "Titanium", 47.867, "transition_metal", 4, 4, "d", "solid"),
    (23, "V", "Vanadium", 50.9415, "transition_metal", 5, 4, "d", "solid"),
    (24, "Cr", "Chromium", 51.9961, "transition_metal", 6, 4, "d", "solid"),
    (25, "Mn", "Manganese", 54.938044, "transition_metal", 7, 4, "d", "solid"),
    (26, "Fe", "Iron", 55.845, "transition_metal", 8, 4, "d", "solid"),
    (27, "Co", "Cobalt", 58.933194, "transition_metal", 9, 4, "d", "solid"),
    (28, "Ni", "Nickel", 58.6934, "transition_metal", 10, 4, "d", "solid"),
    (29, "Cu", "Copper", 63.546, "transition_metal", 11, 4, "d", "solid"),
    (30, "Zn", "Zinc", 65.38, "transition_metal", 12, 4, "d", "solid"),
    (31, "Ga", "Gallium", 69.723, "post_transition_metal", 13, 4, "p", "solid"),
    (32, "Ge", "Germanium", 72.63, "metalloid", 14, 4, "p", "solid"),
    (33, "As", "Arsenic", 74.921595, "metalloid", 15, 4, "p", "solid"),
    (34, "Se", "Selenium", 78.971, "nonmetal", 16, 4, "p", "solid"),
    (35, "Br", "Bromine", 79.904, "halogen", 17, 4, "p", "liquid"),
    (36, "Kr", "Krypton", 83.798, "noble_gas", 18, 4, "p", "gas"),
    (37, "Rb", "Rubidium", 85.4678, "alkali_metal", 1, 5, "s", "solid"),
    (38, "Sr", "Strontium", 87.62, "alkaline_earth_metal", 2, 5, "s", "solid"),
    (39, "Y", "Yttrium", 88.90584, "transition_metal", 3, 5, "d", "solid"),
    (40, "Zr", "Zirconium", 91.224, "transition_metal", 4, 5, "d", "solid"),
    (41, "Nb", "Niobium", 92.90637, "transition_metal", 5, 5, "d", "solid"),
    (42, "Mo", "Molybdenum", 95.95, "transition_metal", 6, 5, "d", "solid"),
    (43, "Tc", "Technetium", 98.0, "transition_metal", 7, 5, "d", "solid"),
    (44, "Ru", "Ruthenium", 101.07, "transition_metal", 8, 5, "d", "solid"),
    (45, "Rh", "Rhodium", 102.9055, "transition_metal", 9, 5, "d", "solid"),
    (46, "Pd", "Palladium", 106.42, "transition_metal", 10, 5, "d", "solid"),
    (47, "Ag", "Silver", 107.8682, "transition_metal", 11, 5, "d", "solid"),
    (48, "Cd", "Cadmium", 112.414, "transition_metal", 12, 5, "d", "solid"),
    (49, "In", "Indium", 114.818, "post_transition_metal", 13, 5, "p", "solid"),
    (50, "Sn", "Tin", 118.71, "post_transition_metal", 14, 5, "p", "solid"),
    (51, "Sb", "Antimony", 121.76, "metalloid", 15, 5, "p", "solid"),
    (52, "Te", "Tellurium", 127.6, "metalloid", 16, 5, "p", "solid"),
    (53, "I", "Iodine", 126.90447, "halogen", 17, 5, "p", "solid"),
    (54, "Xe", "Xenon", 131.293, "noble_gas", 18, 5, "p", "gas"),
    (55, "Cs", "Cesium", 132.90545196, "alkali_metal", 1, 6, "s", "solid"),
    (56, "Ba", "Barium", 137.327, "alkaline_earth_metal", 2, 6, "s", "solid"),
    (57, "La", "Lanthanum", 138.90547, "lanthanide", None, 6, "f", "solid"),
    (58, "Ce", "Cerium", 140.116, "lanthanide", None, 6, "f", "solid"),
    (59, "Pr", "Praseodymium", 140.90766, "lanthanide", None, 6, "f", "solid"),
    (60, "Nd", "Neodymium", 144.242, "lanthanide", None, 6, "f", "solid"),
    (61, "Pm", "Promethium", 145.0, "lanthanide", None, 6, "f", "solid"),
    (62, "Sm", "Samarium", 150.36, "lanthanide", None, 6, "f", "solid"),
    (63, "Eu", "Europium", 151.964, "lanthanide", None, 6, "f", "solid"),
    (64, "Gd", "Gadolinium", 157.25, "lanthanide", None, 6, "f", "solid"),
    (65, "Tb", "Terbium", 158.92535, "lanthanide", None, 6, "f", "solid"),
    (66, "Dy", "Dysprosium", 162.5, "lanthanide", None, 6, "f", "solid"),
    (67, "Ho", "Holmium", 164.93033, "lanthanide", None, 6, "f", "solid"),
    (68, "Er", "Erbium", 167.259, "lanthanide", None, 6, "f", "solid"),
    (69, "Tm", "Thulium", 168.93422, "lanthanide", None, 6, "f", "solid"),
    (70, "Yb", "Ytterbium", 173.045, "lanthanide", None, 6, "f", "solid"),
    (71, "Lu", "Lutetium", 174.9668, "lanthanide", None, 6, "f", "solid"),
    (72, "Hf", "Hafnium", 178.49, "transition_metal", 4, 6, "d", "solid"),
    (73, "Ta", "Tantalum", 180.94788, "transition_metal", 5, 6, "d", "solid"),
    (74, "W", "Tungsten", 183.84, "transition_metal", 6, 6, "d", "solid"),
    (75, "Re", "Rhenium", 186.207, "transition_metal", 7, 6, "d", "solid"),
    (76, "Os", "Osmium", 190.23, "transition_metal", 8, 6, "d", "solid"),
    (77, "Ir", "Iridium", 192.217, "transition_metal", 9, 6, "d", "solid"),
    (78, "Pt", "Platinum", 195.084, "transition_metal", 10, 6, "d", "solid"),
    (79, "Au", "Gold", 196.966569, "transition_metal", 11, 6, "d", "solid"),
    (80, "Hg", "Mercury", 200.592, "transition_metal", 12, 6, "d", "liquid"),
    (81, "Tl", "Thallium", 204.38, "post_transition_metal", 13, 6, "p", "solid"),
    (82, "Pb", "Lead", 207.2, "post_transition_metal", 14, 6, "p", "solid"),
    (83, "Bi", "Bismuth", 208.9804, "post_transition_metal", 15, 6, "p", "solid"),
    (84, "Po", "Polonium", 209.0, "post_transition_metal", 16, 6, "p", "solid"),
    (85, "At", "Astatine", 210.0, "halogen", 17, 6, "p", "solid"),
    (86, "Rn", "Radon", 222.0, "noble_gas", 18, 6, "p", "gas"),
    (87, "Fr", "Francium", 223.0, "alkali_metal", 1, 7, "s", "solid"),
    (88, "Ra", "Radium", 226.0, "alkaline_earth_metal", 2, 7, "s", "solid"),
    (89, "Ac", "Actinium", 227.0, "actinide", None, 7, "f", "solid"),
    (90, "Th", "Thorium", 232.0377, "actinide", None, 7, "f", "solid"),
    (91, "Pa", "Protactinium", 231.03588, "actinide", None, 7, "f", "solid"),
    (92, "U", "Uranium", 238.02891, "actinide", None, 7, "f", "solid"),
    (93, "Np", "Neptunium", 237.0, "actinide", None, 7, "f", "solid"),
    (94, "Pu", "Plutonium", 244.0, "actinide", None, 7, "f", "solid"),
    (95, "Am", "Americium", 243.0, "actinide", None, 7, "f", "solid"),
    (96, "Cm", "Curium", 247.0, "actinide", None, 7, "f", "solid"),
    (97, "Bk", "Berkelium", 247.0, "actinide", None, 7, "f", "solid"),
    (98, "Cf", "Californium", 251.0, "actinide", None, 7, "f", "solid"),
    (99, "Es", "Einsteinium", 252.0, "actinide", None, 7, "f", "solid"),
    (100, "Fm", "Fermium", 257.0, "actinide", None, 7, "f", "solid"),
    (101, "Md", "Mendelevium", 258.0, "actinide", None, 7, "f", "solid"),
    (102, "No", "Nobelium", 259.0, "actinide", None, 7, "f", "solid"),
    (103, "Lr", "Lawrencium", 266.0, "actinide", None, 7, "f", "solid"),
    (104, "Rf", "Rutherfordium", 267.0, "transition_metal", 4, 7, "d", "unknown"),
    (105, "Db", "Dubnium", 268.0, "transition_metal", 5, 7, "d", "unknown"),
    (106, "Sg", "Seaborgium", 269.0, "transition_metal", 6, 7, "d", "unknown"),
    (107, "Bh", "Bohrium", 270.0, "transition_metal", 7, 7, "d", "unknown"),
    (108, "Hs", "Hassium", 277.0, "transition_metal", 8, 7, "d", "unknown"),
    (109, "Mt", "Meitnerium", 278.0, "unknown", 9, 7, "d", "unknown"),
    (110, "Ds", "Darmstadtium", 281.0, "unknown", 10, 7, "d", "unknown"),
    (111, "Rg", "Roentgenium", 282.0, "unknown", 11, 7, "d", "unknown"),
    (112, "Cn", "Copernicium", 285.0, "post_transition_metal", 12, 7, "d", "unknown"),
    (113, "Nh", "Nihonium", 286.0, "post_transition_metal", 13, 7, "p", "unknown"),
    (114, "Fl", "Flerovium", 289.0, "post_transition_metal", 14, 7, "p", "unknown"),
    (115, "Mc", "Moscovium", 290.0, "post_transition_metal", 15, 7, "p", "unknown"),
    (116, "Lv", "Livermorium", 293.0, "post_transition_metal", 16, 7, "p", "unknown"),
    (117, "Ts", "Tennessine", 294.0, "halogen", 17, 7, "p", "unknown"),
    (118, "Og", "Oganesson", 294.0, "noble_gas", 18, 7, "p", "unknown"),
]


ELEMENTS: list[ElementEntry] = [
    ElementEntry(
        atomic_number=atomic_number,
        symbol=symbol,
        name_en=name,
        atomic_mass=atomic_mass,
        category=category,
        group=group,
        period=period,
        block=block,
        state_at_stp=state,
    )
    for atomic_number, symbol, name, atomic_mass, category, group, period, block, state in _ELEMENT_ROWS
]

ELEMENTS_BY_SYMBOL: dict[str, ElementEntry] = {element.symbol: element for element in ELEMENTS}
ELEMENTS_BY_ATOMIC_NUMBER: dict[int, ElementEntry] = {element.atomic_number: element for element in ELEMENTS}

SUBATOMIC_PARTICLES: dict[str, dict[str, object]] = {
    "proton": {"charge": 1, "mass_u": 1.007276, "quarks": "uud", "baryon": True},
    "neutron": {"charge": 0, "mass_u": 1.008665, "quarks": "udd", "baryon": True},
    "electron": {"charge": -1, "mass_u": 0.000549, "lepton": True},
    "photon": {"charge": 0, "mass_u": 0.0, "boson": True, "force": "electromagnetic"},
    "gluon": {"charge": 0, "mass_u": 0.0, "boson": True, "force": "strong"},
    "w_plus": {"charge": 1, "mass_u": 86.1, "boson": True, "force": "weak"},
    "w_minus": {"charge": -1, "mass_u": 86.1, "boson": True, "force": "weak"},
    "z_boson": {"charge": 0, "mass_u": 97.3, "boson": True, "force": "weak"},
    "higgs": {"charge": 0, "mass_u": 134.0, "boson": True, "force": "mass"},
    "up": {"charge": 2 / 3, "generation": 1},
    "down": {"charge": -1 / 3, "generation": 1},
    "charm": {"charge": 2 / 3, "generation": 2},
    "strange": {"charge": -1 / 3, "generation": 2},
    "top": {"charge": 2 / 3, "generation": 3},
    "bottom": {"charge": -1 / 3, "generation": 3},
}


def iter_elements() -> list[ElementEntry]:
    return list(ELEMENTS)


__all__ = [
    "ELEMENTS",
    "ELEMENTS_BY_ATOMIC_NUMBER",
    "ELEMENTS_BY_SYMBOL",
    "ElementEntry",
    "SUBATOMIC_PARTICLES",
    "iter_elements",
]
