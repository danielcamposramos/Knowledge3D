"""Measurement domains, SI conversions, and RPN conversion programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _eval_rpn(program: str, value: float) -> float:
    stack = [float(value)]
    for token in str(program or "").split():
        upper = token.upper()
        if upper in {"MUL", "*"}:
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif upper in {"DIV", "/"}:
            b = stack.pop()
            a = stack.pop()
            stack.append(0.0 if b == 0 else a / b)
        elif upper in {"ADD", "+"}:
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif upper in {"SUB", "-"}:
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        else:
            stack.append(float(token))
    return float(stack[-1])


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    to_si_rpn: str = "1.0 MUL"
    from_si_rpn: str | None = None
    era: str | None = None


@dataclass(frozen=True)
class MeasurementDomain:
    key: str
    si_base: str
    units: dict[str, UnitDefinition]


MEASUREMENT_DOMAINS: dict[str, MeasurementDomain] = {
    "length": MeasurementDomain("length", "metre", {
        "metre": UnitDefinition("m", "1.0 MUL", "1.0 MUL"),
        "kilometre": UnitDefinition("km", "1000.0 MUL", "1000.0 DIV"),
        "centimetre": UnitDefinition("cm", "0.01 MUL", "0.01 DIV"),
        "millimetre": UnitDefinition("mm", "0.001 MUL", "0.001 DIV"),
        "micrometre": UnitDefinition("μm", "1e-6 MUL", "1e-6 DIV"),
        "nanometre": UnitDefinition("nm", "1e-9 MUL", "1e-9 DIV"),
        "inch": UnitDefinition("in", "0.0254 MUL", "0.0254 DIV"),
        "foot": UnitDefinition("ft", "0.3048 MUL", "0.3048 DIV"),
        "yard": UnitDefinition("yd", "0.9144 MUL", "0.9144 DIV"),
        "mile": UnitDefinition("mi", "1609.344 MUL", "1609.344 DIV"),
        "nautical_mile": UnitDefinition("nmi", "1852.0 MUL", "1852.0 DIV"),
        "astronomical_unit": UnitDefinition("AU", "1.496e11 MUL", "1.496e11 DIV"),
        "light_year": UnitDefinition("ly", "9.461e15 MUL", "9.461e15 DIV"),
        "parsec": UnitDefinition("pc", "3.086e16 MUL", "3.086e16 DIV"),
        "angstrom": UnitDefinition("Å", "1e-10 MUL", "1e-10 DIV"),
        "cubit": UnitDefinition("cubit", "0.4572 MUL", "0.4572 DIV", era="ancient"),
        "fathom": UnitDefinition("ftm", "1.8288 MUL", "1.8288 DIV"),
        "furlong": UnitDefinition("fur", "201.168 MUL", "201.168 DIV"),
        "league": UnitDefinition("lea", "4828.032 MUL", "4828.032 DIV"),
    }),
    "mass": MeasurementDomain("mass", "kilogram", {
        "kilogram": UnitDefinition("kg", "1.0 MUL", "1.0 MUL"),
        "gram": UnitDefinition("g", "0.001 MUL", "0.001 DIV"),
        "milligram": UnitDefinition("mg", "1e-6 MUL", "1e-6 DIV"),
        "tonne": UnitDefinition("t", "1000.0 MUL", "1000.0 DIV"),
        "pound": UnitDefinition("lb", "0.453592 MUL", "0.453592 DIV"),
        "ounce": UnitDefinition("oz", "0.0283495 MUL", "0.0283495 DIV"),
        "stone": UnitDefinition("st", "6.35029 MUL", "6.35029 DIV"),
        "atomic_mass_unit": UnitDefinition("u", "1.66054e-27 MUL", "1.66054e-27 DIV"),
        "solar_mass": UnitDefinition("M☉", "1.989e30 MUL", "1.989e30 DIV"),
        "carat": UnitDefinition("ct", "0.0002 MUL", "0.0002 DIV"),
        "grain": UnitDefinition("gr", "6.47989e-5 MUL", "6.47989e-5 DIV"),
    }),
    "temperature": MeasurementDomain("temperature", "kelvin", {
        "kelvin": UnitDefinition("K", "1.0 MUL", "1.0 MUL"),
        "celsius": UnitDefinition("°C", "273.15 ADD", "273.15 SUB"),
        "fahrenheit": UnitDefinition("°F", "32 SUB 5 MUL 9 DIV 273.15 ADD", "273.15 SUB 9 MUL 5 DIV 32 ADD"),
        "rankine": UnitDefinition("°R", "5 MUL 9 DIV", "9 MUL 5 DIV"),
    }),
    "pressure": MeasurementDomain("pressure", "pascal", {
        "pascal": UnitDefinition("Pa", "1.0 MUL", "1.0 MUL"),
        "kilopascal": UnitDefinition("kPa", "1000.0 MUL", "1000.0 DIV"),
        "megapascal": UnitDefinition("MPa", "1e6 MUL", "1e6 DIV"),
        "bar": UnitDefinition("bar", "1e5 MUL", "1e5 DIV"),
        "atmosphere": UnitDefinition("atm", "101325.0 MUL", "101325.0 DIV"),
        "torr": UnitDefinition("Torr", "133.322 MUL", "133.322 DIV"),
        "psi": UnitDefinition("psi", "6894.76 MUL", "6894.76 DIV"),
        "mmHg": UnitDefinition("mmHg", "133.322 MUL", "133.322 DIV"),
    }),
    "time": MeasurementDomain("time", "second", {
        "second": UnitDefinition("s", "1.0 MUL", "1.0 MUL"),
        "millisecond": UnitDefinition("ms", "0.001 MUL", "0.001 DIV"),
        "microsecond": UnitDefinition("μs", "1e-6 MUL", "1e-6 DIV"),
        "nanosecond": UnitDefinition("ns", "1e-9 MUL", "1e-9 DIV"),
        "minute": UnitDefinition("min", "60.0 MUL", "60.0 DIV"),
        "hour": UnitDefinition("h", "3600.0 MUL", "3600.0 DIV"),
        "day": UnitDefinition("d", "86400.0 MUL", "86400.0 DIV"),
        "week": UnitDefinition("wk", "604800.0 MUL", "604800.0 DIV"),
        "year_julian": UnitDefinition("a", "31557600.0 MUL", "31557600.0 DIV"),
        "planck_time": UnitDefinition("tₚ", "5.391e-44 MUL", "5.391e-44 DIV"),
    }),
    "electric_current": MeasurementDomain("electric_current", "ampere", {
        "ampere": UnitDefinition("A", "1.0 MUL", "1.0 MUL"),
        "milliampere": UnitDefinition("mA", "0.001 MUL", "0.001 DIV"),
    }),
    "voltage": MeasurementDomain("voltage", "volt", {
        "volt": UnitDefinition("V", "1.0 MUL", "1.0 MUL"),
        "millivolt": UnitDefinition("mV", "0.001 MUL", "0.001 DIV"),
        "kilovolt": UnitDefinition("kV", "1000.0 MUL", "1000.0 DIV"),
    }),
    "energy": MeasurementDomain("energy", "joule", {
        "joule": UnitDefinition("J", "1.0 MUL", "1.0 MUL"),
        "kilojoule": UnitDefinition("kJ", "1000.0 MUL", "1000.0 DIV"),
        "calorie": UnitDefinition("cal", "4.184 MUL", "4.184 DIV"),
        "kilocalorie": UnitDefinition("kcal", "4184.0 MUL", "4184.0 DIV"),
        "electronvolt": UnitDefinition("eV", "1.602e-19 MUL", "1.602e-19 DIV"),
        "kilowatt_hour": UnitDefinition("kWh", "3.6e6 MUL", "3.6e6 DIV"),
        "btu": UnitDefinition("BTU", "1055.06 MUL", "1055.06 DIV"),
        "erg": UnitDefinition("erg", "1e-7 MUL", "1e-7 DIV"),
    }),
    "force": MeasurementDomain("force", "newton", {
        "newton": UnitDefinition("N", "1.0 MUL", "1.0 MUL"),
        "kilonewton": UnitDefinition("kN", "1000.0 MUL", "1000.0 DIV"),
        "dyne": UnitDefinition("dyn", "1e-5 MUL", "1e-5 DIV"),
        "pound_force": UnitDefinition("lbf", "4.44822 MUL", "4.44822 DIV"),
    }),
    "frequency": MeasurementDomain("frequency", "hertz", {
        "hertz": UnitDefinition("Hz", "1.0 MUL", "1.0 MUL"),
        "kilohertz": UnitDefinition("kHz", "1000.0 MUL", "1000.0 DIV"),
        "megahertz": UnitDefinition("MHz", "1e6 MUL", "1e6 DIV"),
        "gigahertz": UnitDefinition("GHz", "1e9 MUL", "1e9 DIV"),
    }),
    "area": MeasurementDomain("area", "square_metre", {
        "square_metre": UnitDefinition("m²", "1.0 MUL", "1.0 MUL"),
        "hectare": UnitDefinition("ha", "10000.0 MUL", "10000.0 DIV"),
        "acre": UnitDefinition("ac", "4046.86 MUL", "4046.86 DIV"),
    }),
    "volume": MeasurementDomain("volume", "cubic_metre", {
        "cubic_metre": UnitDefinition("m³", "1.0 MUL", "1.0 MUL"),
        "litre": UnitDefinition("L", "0.001 MUL", "0.001 DIV"),
        "gallon_us": UnitDefinition("gal", "0.003785 MUL", "0.003785 DIV"),
        "gallon_uk": UnitDefinition("gal(UK)", "0.004546 MUL", "0.004546 DIV"),
    }),
    "speed": MeasurementDomain("speed", "metre_per_second", {
        "metre_per_second": UnitDefinition("m/s", "1.0 MUL", "1.0 MUL"),
        "kilometre_per_hour": UnitDefinition("km/h", "0.27778 MUL", "0.27778 DIV"),
        "mile_per_hour": UnitDefinition("mph", "0.44704 MUL", "0.44704 DIV"),
        "knot": UnitDefinition("kn", "0.51444 MUL", "0.51444 DIV"),
        "speed_of_light": UnitDefinition("c", "2.998e8 MUL", "2.998e8 DIV"),
    }),
    "angle": MeasurementDomain("angle", "radian", {
        "radian": UnitDefinition("rad", "1.0 MUL", "1.0 MUL"),
        "degree": UnitDefinition("°", "0.017453 MUL", "0.017453 DIV"),
        "arcminute": UnitDefinition("′", "0.000290888 MUL", "0.000290888 DIV"),
        "arcsecond": UnitDefinition("″", "4.84814e-6 MUL", "4.84814e-6 DIV"),
        "gradian": UnitDefinition("gon", "0.015708 MUL", "0.015708 DIV"),
    }),
    "luminous_intensity": MeasurementDomain("luminous_intensity", "candela", {
        "candela": UnitDefinition("cd", "1.0 MUL", "1.0 MUL"),
        "lumen": UnitDefinition("lm", "1.0 MUL", "1.0 MUL"),
        "lux": UnitDefinition("lx", "1.0 MUL", "1.0 MUL"),
    }),
    "amount_of_substance": MeasurementDomain("amount_of_substance", "mole", {
        "mole": UnitDefinition("mol", "1.0 MUL", "1.0 MUL"),
    }),
    "data_storage": MeasurementDomain("data_storage", "byte", {
        "bit": UnitDefinition("b", "0.125 MUL", "8 MUL"),
        "byte": UnitDefinition("B", "1.0 MUL", "1.0 MUL"),
        "kilobyte": UnitDefinition("KB", "1000.0 MUL", "1000.0 DIV"),
        "megabyte": UnitDefinition("MB", "1e6 MUL", "1e6 DIV"),
        "gigabyte": UnitDefinition("GB", "1e9 MUL", "1e9 DIV"),
        "terabyte": UnitDefinition("TB", "1e12 MUL", "1e12 DIV"),
        "petabyte": UnitDefinition("PB", "1e15 MUL", "1e15 DIV"),
    }),
}


def iter_domains() -> list[MeasurementDomain]:
    return [MEASUREMENT_DOMAINS[key] for key in sorted(MEASUREMENT_DOMAINS.keys())]


def to_si_value(domain_name: str, unit_name: str, value: float) -> float:
    domain = MEASUREMENT_DOMAINS[str(domain_name).strip().lower()]
    unit = domain.units[str(unit_name).strip().lower()]
    return _eval_rpn(unit.to_si_rpn, value)


def from_si_value(domain_name: str, unit_name: str, value: float) -> float:
    domain = MEASUREMENT_DOMAINS[str(domain_name).strip().lower()]
    unit = domain.units[str(unit_name).strip().lower()]
    program = unit.from_si_rpn
    if program is None:
        if unit.to_si_rpn.endswith(" MUL"):
            factor = float(unit.to_si_rpn.split()[0])
            return value / factor if factor else 0.0
        return value
    return _eval_rpn(program, value)


def convert(domain_name: str, value: float, from_unit: str, to_unit: str) -> float:
    si_value = to_si_value(domain_name, from_unit, value)
    return from_si_value(domain_name, to_unit, si_value)


__all__ = ["MEASUREMENT_DOMAINS", "MeasurementDomain", "UnitDefinition", "convert", "from_si_value", "iter_domains", "to_si_value"]
