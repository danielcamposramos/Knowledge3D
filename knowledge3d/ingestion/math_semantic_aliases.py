"""Batch 8 math semantic alias tables.

Pure constants only. This module is ingestion-path infrastructure.
"""

from __future__ import annotations

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id
from knowledge3d.ingestion.math_symbol_builder import math_symbol_star_id


LETTER_ALIASES: dict[str, str] = {
    "alpha": canonical_char_star_id("\u03b1"),
    "beta": canonical_char_star_id("\u03b2"),
    "gamma": canonical_char_star_id("\u03b3"),
    "theta": canonical_char_star_id("\u03b8"),
    "mu": canonical_char_star_id("\u03bc"),
    "sigma": canonical_char_star_id("\u03c3"),
    "lambda": canonical_char_star_id("\u03bb"),
    "rho": canonical_char_star_id("\u03c1"),
}

CHAR_ALIASES: dict[str, str] = {
    "greater_glyph": canonical_char_star_id(">"),
}


SYMBOL_ALIASES: dict[str, str] = {
    "plus": math_symbol_star_id("+"),
    "minus": math_symbol_star_id("-"),
    "times": math_symbol_star_id("\u00d7"),
    "divide": math_symbol_star_id("\u00f7"),
    "equal": math_symbol_star_id("="),
    "sqrt": math_symbol_star_id("\u221a"),
    "sum": math_symbol_star_id("\u2211"),
    "product": math_symbol_star_id("\u220f"),
    "integral": math_symbol_star_id("\u222b"),
    "pi": math_symbol_star_id("\u03c0"),
    "infinity": math_symbol_star_id("\u221e"),
    "leq": math_symbol_star_id("\u2264"),
    "geq": math_symbol_star_id("\u2265"),
    "neq": math_symbol_star_id("\u2260"),
    "in": math_symbol_star_id("\u2208"),
    "notin": math_symbol_star_id("\u2209"),
    "subset": math_symbol_star_id("\u2282"),
    "subseteq": math_symbol_star_id("\u2286"),
    "plus_minus": math_symbol_star_id("\u00b1"),
    "power": canonical_char_star_id("^"),
    "parenthesis": canonical_char_star_id("("),
    "bracket": canonical_char_star_id("["),
    "absolute": canonical_char_star_id("|"),
    "floor": canonical_char_star_id("\u230a"),
    "multiply": math_symbol_star_id("\u00d7"),
    "add": math_symbol_star_id("+"),
    "subtract": math_symbol_star_id("-"),
    "div": math_symbol_star_id("\u00f7"),
    "exponent": canonical_char_star_id("^"),
    "fraction": canonical_char_star_id("/"),
    "zero": canonical_char_star_id("0"),
    "alpha": math_symbol_star_id("\u03b1"),
    "beta": math_symbol_star_id("\u03b2"),
    "gamma": math_symbol_star_id("\u03b3"),
    "theta": math_symbol_star_id("\u03b8"),
    "mu": math_symbol_star_id("\u03bc"),
    "sigma": math_symbol_star_id("\u03c3"),
    "lambda": math_symbol_star_id("\u03bb"),
    "rho": math_symbol_star_id("\u03c1"),
    "sin": "math_symbol_sin",
    "cos": "math_symbol_cos",
    "tan": "math_symbol_tan",
    "greater": math_symbol_star_id(">"),
}

CONSTANT_ALIASES: dict[str, str] = {
    "pi": math_symbol_star_id("\u03c0"),
    "e": canonical_char_star_id("e"),
    "reciprocal": "concept_reciprocal",
    "zero": canonical_char_star_id("0"),
    "one": canonical_char_star_id("1"),
    "ten": canonical_char_star_id("1"),
    "half": canonical_char_star_id("2"),
    "quarter": canonical_char_star_id("4"),
}

CONCEPT_MEANING_STARS: dict[str, str] = {
    "base_ten": "concept_base_ten",
    "prime": "concept_prime",
    "alpha": "concept_alpha",
    "beta": "concept_beta",
    "gamma": "concept_gamma",
    "theta": "concept_theta",
    "mu": "concept_mu",
    "sigma": "concept_sigma",
    "lambda": "concept_lambda",
    "rho": "concept_rho",
    "sin": "concept_sin",
    "cos": "concept_cos",
    "tan": "concept_tan",
    "greater": "concept_greater",
}

UNICODE_TO_NAME: dict[str, str] = {
    "\u221a": "sqrt",
    "\u03c0": "pi",
    "\u221e": "infinity",
    "\u2211": "sum",
    "\u220f": "product",
    "\u222b": "integral",
    "\u00d7": "times",
    "\u00f7": "divide",
    "\u00b1": "plus_minus",
    "\u2260": "neq",
    "\u2264": "leq",
    "\u2265": "geq",
    "\u2208": "in",
    "\u2209": "notin",
    "\u2282": "subset",
    "\u2286": "subseteq",
}


__all__ = [
    "CHAR_ALIASES",
    "CONCEPT_MEANING_STARS",
    "CONSTANT_ALIASES",
    "LETTER_ALIASES",
    "SYMBOL_ALIASES",
    "UNICODE_TO_NAME",
]
