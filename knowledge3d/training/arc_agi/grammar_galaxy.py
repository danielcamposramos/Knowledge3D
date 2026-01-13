"""Compatibility wrapper for arc_agi.grammar_galaxy."""

from __future__ import annotations

import importlib


_module = importlib.import_module("grammar_galaxy")

GrammarRule = _module.GrammarRule
GrammarGalaxy = _module.GrammarGalaxy
default_grammar_rules = _module.default_grammar_rules
get_grammar_galaxy = getattr(_module, "get_grammar_galaxy", None)

__all__ = [
    "GrammarRule",
    "GrammarGalaxy",
    "default_grammar_rules",
    "get_grammar_galaxy",
]
