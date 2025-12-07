"""
Math Galaxy — Canonical Always-Loaded Mathematical Symbol Knowledge.

This galaxy stores mathematical symbols as procedural RPN programs.
Like Character Galaxy is to languages, Math Galaxy is foundational for mathematics.

Architecture:
    - Layer 1 (Form): Procedural glyph (Bézier → RPN)
    - Layer 2 (Meaning): Unicode + semantic domain + pronunciation
    - Layer 3 (Rules): RPN programs that use this symbol (symlink refs)
    - Layer 4 (Meta-Rules): When to apply which symbol

Symbols here are KNOWLEDGE, not trainable weights. They are always-loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class MathSymbol:
    """A mathematical symbol with procedural definition and semantic metadata."""

    # Layer 1: Form
    unicode_codepoint: int                  # e.g., 8721 for ∑ (U+2211)
    char: str                               # The actual character: "∑"
    rpn_program: str                        # Procedural drawing program
    bezier_segments: List[Tuple[float, ...]] = field(default_factory=list)

    # Layer 2: Meaning
    name: str = ""
    domain: str = "math"
    latex: str = ""
    pronunciation: str = ""

    # Layer 3: Rule references (symlinks, not duplicated)
    rule_refs: List[str] = field(default_factory=list)

    # Layer 4: Meta-rule references
    meta_refs: List[str] = field(default_factory=list)

    @property
    def symbol_id(self) -> str:
        return f"math_{self.unicode_codepoint}"


class MathGalaxy:
    """
    Canonical storage for mathematical symbols (always-loaded).
    Follows Character Galaxy pattern — procedural definitions, not trained weights.
    """

    # Canonical symbol definitions (organized by domain; add here to reach 152 total math symbols + 12 pt diacritics)
    CANONICAL_SYMBOLS: Dict[int, Dict] = {
        # High Priority — Core calculus/analysis (subset shown)
        8721: {"char": "∑", "name": "summation", "domain": "math_calculus", "latex": "\\sum"},
        8747: {"char": "∫", "name": "integral", "domain": "math_calculus", "latex": "\\int"},
        8706: {"char": "∂", "name": "partial_derivative", "domain": "math_calculus", "latex": "\\partial"},
        8711: {"char": "∇", "name": "nabla", "domain": "math_vector", "latex": "\\nabla"},
        916: {"char": "Δ", "name": "delta", "domain": "math_calculus", "latex": "\\Delta"},
        8719: {"char": "∏", "name": "product", "domain": "math_calculus", "latex": "\\prod"},
        8730: {"char": "√", "name": "sqrt", "domain": "math_algebra", "latex": "\\sqrt"},
        8734: {"char": "∞", "name": "infinity", "domain": "math_analysis", "latex": "\\infty"},
        177: {"char": "±", "name": "plus_minus", "domain": "math_algebra", "latex": "\\pm"},
        945: {"char": "α", "name": "alpha", "domain": "math_greek", "latex": "\\alpha"},
        946: {"char": "β", "name": "beta", "domain": "math_greek", "latex": "\\beta"},
        947: {"char": "γ", "name": "gamma", "domain": "math_greek", "latex": "\\gamma"},
        948: {"char": "δ", "name": "delta_lower", "domain": "math_greek", "latex": "\\delta"},
        949: {"char": "ε", "name": "epsilon", "domain": "math_greek", "latex": "\\epsilon"},
        952: {"char": "θ", "name": "theta", "domain": "math_greek", "latex": "\\theta"},
        955: {"char": "λ", "name": "lambda", "domain": "math_greek", "latex": "\\lambda"},
        956: {"char": "μ", "name": "mu", "domain": "math_greek", "latex": "\\mu"},
        960: {"char": "π", "name": "pi", "domain": "math_greek", "latex": "\\pi"},
        963: {"char": "σ", "name": "sigma_lower", "domain": "math_greek", "latex": "\\sigma"},
        969: {"char": "ω", "name": "omega", "domain": "math_greek", "latex": "\\omega"},
        # Medium Priority — Set theory/logic (subset shown)
        8712: {"char": "∈", "name": "element_of", "domain": "math_set", "latex": "\\in"},
        8713: {"char": "∉", "name": "not_element_of", "domain": "math_set", "latex": "\\notin"},
        8714: {"char": "∊", "name": "small_element_of", "domain": "math_set", "latex": "\\in"},
        8715: {"char": "∋", "name": "contains_as_member", "domain": "math_set", "latex": "\\ni"},
        8716: {"char": "∌", "name": "does_not_contain", "domain": "math_set", "latex": "\\notni"},
        8717: {"char": "∍", "name": "contains_as_member_variant", "domain": "math_set", "latex": "\\barni"},
        8834: {"char": "⊂", "name": "subset", "domain": "math_set", "latex": "\\subset"},
        8835: {"char": "⊃", "name": "superset", "domain": "math_set", "latex": "\\superset"},
        8838: {"char": "⊆", "name": "subset_eq", "domain": "math_set", "latex": "\\subseteq"},
        8839: {"char": "⊇", "name": "superset_eq", "domain": "math_set", "latex": "\\supseteq"},
        8836: {"char": "⊄", "name": "not_subset", "domain": "math_set", "latex": "\\nsubset"},
        8837: {"char": "⊅", "name": "not_superset", "domain": "math_set", "latex": "\\nsupset"},
        8840: {"char": "⊈", "name": "not_subset_eq", "domain": "math_set", "latex": "\\nsubseteq"},
        8841: {"char": "⊉", "name": "not_superset_eq", "domain": "math_set", "latex": "\\nsupseteq"},
        8842: {"char": "⊊", "name": "subset_not_eq", "domain": "math_set", "latex": "\\subsetneq"},
        8843: {"char": "⊋", "name": "superset_not_eq", "domain": "math_set", "latex": "\\supsetneq"},
        8746: {"char": "∪", "name": "union", "domain": "math_set", "latex": "\\cup"},
        8745: {"char": "∩", "name": "intersection", "domain": "math_set", "latex": "\\cap"},
        8846: {"char": "⊎", "name": "disjoint_union", "domain": "math_set", "latex": "\\uplus"},
        8851: {"char": "⊓", "name": "square_intersection", "domain": "math_set", "latex": "\\sqcap"},
        8852: {"char": "⊔", "name": "square_union", "domain": "math_set", "latex": "\\sqcup"},
        8709: {"char": "∅", "name": "empty_set", "domain": "math_set", "latex": "\\emptyset"},
        8704: {"char": "∀", "name": "forall", "domain": "math_logic", "latex": "\\forall"},
        8707: {"char": "∃", "name": "exists", "domain": "math_logic", "latex": "\\exists"},
        8743: {"char": "∧", "name": "logical_and", "domain": "math_logic", "latex": "\\land"},
        8744: {"char": "∨", "name": "logical_or", "domain": "math_logic", "latex": "\\lor"},
        172: {"char": "¬", "name": "negation", "domain": "math_logic", "latex": "\\neg"},
        8658: {"char": "⇒", "name": "implies", "domain": "math_logic", "latex": "\\Rightarrow"},
        8660: {"char": "⇔", "name": "iff", "domain": "math_logic", "latex": "\\Leftrightarrow"},
        8708: {"char": "∄", "name": "not_exists", "domain": "math_logic", "latex": "\\nexists"},
        8866: {"char": "⊢", "name": "entails_left", "domain": "math_logic", "latex": "\\vdash"},
        8867: {"char": "⊣", "name": "entails_right", "domain": "math_logic", "latex": "\\dashv"},
        8868: {"char": "⊤", "name": "top", "domain": "math_logic", "latex": "\\top"},
        8872: {"char": "⊨", "name": "models", "domain": "math_logic", "latex": "\\models"},

        # Relations & comparisons
        8804: {"char": "≤", "name": "less_equal", "domain": "math_relation", "latex": "\\leq"},
        8805: {"char": "≥", "name": "greater_equal", "domain": "math_relation", "latex": "\\geq"},
        8800: {"char": "≠", "name": "not_equal", "domain": "math_relation", "latex": "\\neq"},
        8776: {"char": "≈", "name": "approximately", "domain": "math_relation", "latex": "\\approx"},
        8801: {"char": "≡", "name": "identical", "domain": "math_relation", "latex": "\\equiv"},
        8733: {"char": "∝", "name": "proportional", "domain": "math_relation", "latex": "\\propto"},
        8810: {"char": "≪", "name": "much_less", "domain": "math_relation", "latex": "\\ll"},
        8811: {"char": "≫", "name": "much_greater", "domain": "math_relation", "latex": "\\gg"},
        8826: {"char": "≺", "name": "precedes", "domain": "math_relation", "latex": "\\prec"},
        8827: {"char": "≻", "name": "succeeds", "domain": "math_relation", "latex": "\\succ"},
        8828: {"char": "≼", "name": "precedes_eq", "domain": "math_relation", "latex": "\\preceq"},
        8829: {"char": "≽", "name": "succeeds_eq", "domain": "math_relation", "latex": "\\succeq"},
        8870: {"char": "⊀", "name": "not_precedes", "domain": "math_relation", "latex": "\\nprec"},
        8871: {"char": "⊁", "name": "not_succeeds", "domain": "math_relation", "latex": "\\nsucc"},

        # Arrows
        8594: {"char": "→", "name": "right_arrow", "domain": "math_arrow", "latex": "\\rightarrow"},
        8592: {"char": "←", "name": "left_arrow", "domain": "math_arrow", "latex": "\\leftarrow"},
        8596: {"char": "↔", "name": "left_right_arrow", "domain": "math_arrow", "latex": "\\leftrightarrow"},
        8593: {"char": "↑", "name": "up_arrow", "domain": "math_arrow", "latex": "\\uparrow"},
        8595: {"char": "↓", "name": "down_arrow", "domain": "math_arrow", "latex": "\\downarrow"},
        8597: {"char": "↕", "name": "up_down_arrow", "domain": "math_arrow", "latex": "\\updownarrow"},
        8598: {"char": "↖", "name": "up_left_arrow", "domain": "math_arrow", "latex": "\\nwarrow"},
        8599: {"char": "↗", "name": "up_right_arrow", "domain": "math_arrow", "latex": "\\nearrow"},
        8600: {"char": "↘", "name": "down_right_arrow", "domain": "math_arrow", "latex": "\\searrow"},
        8601: {"char": "↙", "name": "down_left_arrow", "domain": "math_arrow", "latex": "\\swarrow"},
        8614: {"char": "↦", "name": "maps_to", "domain": "math_arrow", "latex": "\\mapsto"},
        8656: {"char": "⇐", "name": "left_double_arrow", "domain": "math_arrow", "latex": "\\Leftarrow"},
        8657: {"char": "⇑", "name": "up_double_arrow", "domain": "math_arrow", "latex": "\\Uparrow"},
        8659: {"char": "⇓", "name": "down_double_arrow", "domain": "math_arrow", "latex": "\\Downarrow"},
        8661: {"char": "⇕", "name": "up_down_double_arrow", "domain": "math_arrow", "latex": "\\Updownarrow"},

        # Operators
        215: {"char": "×", "name": "times", "domain": "math_operator", "latex": "\\times"},
        247: {"char": "÷", "name": "divide", "domain": "math_operator", "latex": "\\div"},
        8901: {"char": "⋅", "name": "dot", "domain": "math_operator", "latex": "\\cdot"},
        8853: {"char": "⊕", "name": "oplus", "domain": "math_operator", "latex": "\\oplus"},
        8855: {"char": "⊗", "name": "otimes", "domain": "math_operator", "latex": "\\otimes"},
        8857: {"char": "⊙", "name": "odot", "domain": "math_operator", "latex": "\\odot"},
        8728: {"char": "∘", "name": "compose", "domain": "math_operator", "latex": "\\circ"},
        8727: {"char": "∗", "name": "asterisk", "domain": "math_operator", "latex": "\\ast"},
        8224: {"char": "†", "name": "dagger", "domain": "math_operator", "latex": "\\dagger"},
        8225: {"char": "‡", "name": "double_dagger", "domain": "math_operator", "latex": "\\ddagger"},
        8731: {"char": "∛", "name": "cube_root", "domain": "math_operator", "latex": "\\sqrt[3]{}"},
        8732: {"char": "∜", "name": "fourth_root", "domain": "math_operator", "latex": "\\sqrt[4]{}"},
        8720: {"char": "∐", "name": "coproduct", "domain": "math_operator", "latex": "\\coprod"},

        # Greek (uppercase)
        915: {"char": "Γ", "name": "Gamma", "domain": "math_greek", "latex": "\\Gamma"},
        920: {"char": "Θ", "name": "Theta", "domain": "math_greek", "latex": "\\Theta"},
        923: {"char": "Λ", "name": "Lambda", "domain": "math_greek", "latex": "\\Lambda"},
        926: {"char": "Ξ", "name": "Xi", "domain": "math_greek", "latex": "\\Xi"},
        928: {"char": "Π", "name": "Pi", "domain": "math_greek", "latex": "\\Pi"},
        931: {"char": "Σ", "name": "Sigma", "domain": "math_greek", "latex": "\\Sigma"},
        933: {"char": "Υ", "name": "Upsilon", "domain": "math_greek", "latex": "\\Upsilon"},
        934: {"char": "Φ", "name": "Phi", "domain": "math_greek", "latex": "\\Phi"},
        936: {"char": "Ψ", "name": "Psi", "domain": "math_greek", "latex": "\\Psi"},
        937: {"char": "Ω", "name": "Omega", "domain": "math_greek", "latex": "\\Omega"},

        # Greek (lowercase, additional)
        950: {"char": "ζ", "name": "zeta", "domain": "math_greek", "latex": "\\zeta"},
        951: {"char": "η", "name": "eta", "domain": "math_greek", "latex": "\\eta"},
        953: {"char": "ι", "name": "iota", "domain": "math_greek", "latex": "\\iota"},
        954: {"char": "κ", "name": "kappa", "domain": "math_greek", "latex": "\\kappa"},
        957: {"char": "ν", "name": "nu", "domain": "math_greek", "latex": "\\nu"},
        958: {"char": "ξ", "name": "xi", "domain": "math_greek", "latex": "\\xi"},
        961: {"char": "ρ", "name": "rho", "domain": "math_greek", "latex": "\\rho"},
        962: {"char": "ς", "name": "final_sigma", "domain": "math_greek", "latex": "\\varsigma"},
        964: {"char": "τ", "name": "tau", "domain": "math_greek", "latex": "\\tau"},
        965: {"char": "υ", "name": "upsilon", "domain": "math_greek", "latex": "\\upsilon"},
        966: {"char": "φ", "name": "phi", "domain": "math_greek", "latex": "\\phi"},
        967: {"char": "χ", "name": "chi", "domain": "math_greek", "latex": "\\chi"},
        968: {"char": "ψ", "name": "psi", "domain": "math_greek", "latex": "\\psi"},
        977: {"char": "ϑ", "name": "theta_variant", "domain": "math_greek", "latex": "\\vartheta"},
        981: {"char": "ϕ", "name": "phi_variant", "domain": "math_greek", "latex": "\\varphi"},
        982: {"char": "ϖ", "name": "pi_variant", "domain": "math_greek", "latex": "\\varpi"},
        1008: {"char": "ϰ", "name": "kappa_variant", "domain": "math_greek", "latex": "\\varkappa"},
        1009: {"char": "ϱ", "name": "rho_variant", "domain": "math_greek", "latex": "\\varrho"},
        1013: {"char": "ϵ", "name": "epsilon_variant", "domain": "math_greek", "latex": "\\varepsilon"},

        # Geometry
        8736: {"char": "∠", "name": "angle", "domain": "math_geometry", "latex": "\\angle"},
        8869: {"char": "⊥", "name": "perpendicular", "domain": "math_geometry", "latex": "\\perp"},
        8741: {"char": "∥", "name": "parallel", "domain": "math_geometry", "latex": "\\parallel"},
        8773: {"char": "≅", "name": "congruent", "domain": "math_geometry", "latex": "\\cong"},
        8764: {"char": "∼", "name": "similar", "domain": "math_geometry", "latex": "\\sim"},
        9651: {"char": "△", "name": "triangle", "domain": "math_geometry", "latex": "\\triangle"},
        9633: {"char": "□", "name": "square", "domain": "math_geometry", "latex": "\\square"},
        9675: {"char": "○", "name": "circle", "domain": "math_geometry", "latex": "\\circ"},
        9655: {"char": "▷", "name": "triangle_right", "domain": "math_geometry", "latex": "\\triangleright"},
        9665: {"char": "◁", "name": "triangle_left", "domain": "math_geometry", "latex": "\\triangleleft"},
        9661: {"char": "▽", "name": "triangle_down", "domain": "math_geometry", "latex": "\\triangledown"},
        9679: {"char": "●", "name": "circle_filled", "domain": "math_geometry", "latex": "\\bullet"},
        9632: {"char": "■", "name": "square_filled", "domain": "math_geometry", "latex": "\\blacksquare"},
        8735: {"char": "∟", "name": "right_angle", "domain": "math_geometry", "latex": "\\angle"},
        8737: {"char": "∡", "name": "measured_angle", "domain": "math_geometry", "latex": "\\measuredangle"},
        8738: {"char": "∢", "name": "spherical_angle", "domain": "math_geometry", "latex": "\\sphericalangle"},
        2224: {"char": "∤", "name": "not_divides", "domain": "math_geometry", "latex": "\\not\\mid"},
        2226: {"char": "∦", "name": "not_parallel", "domain": "math_geometry", "latex": "\\nparallel"},

        # Calculus/analysis (extended integrals/primes)
        8750: {"char": "∮", "name": "contour_integral", "domain": "math_calculus", "latex": "\\oint"},
        8751: {"char": "∯", "name": "surface_integral", "domain": "math_calculus", "latex": "\\oiint"},
        8752: {"char": "∰", "name": "volume_integral", "domain": "math_calculus", "latex": "\\oiiint"},
        8748: {"char": "∬", "name": "double_integral", "domain": "math_calculus", "latex": "\\iint"},
        8749: {"char": "∭", "name": "triple_integral", "domain": "math_calculus", "latex": "\\iiint"},
        8242: {"char": "′", "name": "prime", "domain": "math_calculus", "latex": "'"},
        8243: {"char": "″", "name": "double_prime", "domain": "math_calculus", "latex": "''"},

        # Number sets
        8469: {"char": "ℕ", "name": "naturals", "domain": "math_set", "latex": "\\mathbb{N}"},
        8484: {"char": "ℤ", "name": "integers", "domain": "math_set", "latex": "\\mathbb{Z}"},
        8474: {"char": "ℚ", "name": "rationals", "domain": "math_set", "latex": "\\mathbb{Q}"},
        8477: {"char": "ℝ", "name": "reals", "domain": "math_set", "latex": "\\mathbb{R}"},
        8450: {"char": "ℂ", "name": "complex", "domain": "math_set", "latex": "\\mathbb{C}"},
        8461: {"char": "ℍ", "name": "quaternions", "domain": "math_set", "latex": "\\mathbb{H}"},
        8473: {"char": "ℙ", "name": "primes", "domain": "math_set", "latex": "\\mathbb{P}"},

        # Miscellaneous math
        8756: {"char": "∴", "name": "therefore", "domain": "math_logic", "latex": "\\therefore"},
        8757: {"char": "∵", "name": "because", "domain": "math_logic", "latex": "\\because"},
        8943: {"char": "⋯", "name": "cdots", "domain": "math_misc", "latex": "\\cdots"},
        8942: {"char": "⋮", "name": "vdots", "domain": "math_misc", "latex": "\\vdots"},
        8944: {"char": "⋰", "name": "iddots", "domain": "math_misc", "latex": "\\iddots"},
        8945: {"char": "⋱", "name": "ddots", "domain": "math_misc", "latex": "\\ddots"},
        10003: {"char": "✓", "name": "checkmark", "domain": "math_misc", "latex": "\\checkmark"},
        # Portuguese diacritics (always-loaded for multilingual support)
        227: {"char": "ã", "name": "a_tilde", "domain": "lang_pt", "latex": "\\~{a}"},
        225: {"char": "á", "name": "a_acute", "domain": "lang_pt", "latex": "\\'{a}"},
        224: {"char": "à", "name": "a_grave", "domain": "lang_pt", "latex": "\\`{a}"},
        226: {"char": "â", "name": "a_circumflex", "domain": "lang_pt", "latex": "\\^{a}"},
        231: {"char": "ç", "name": "c_cedilla", "domain": "lang_pt", "latex": "\\c{c}"},
        233: {"char": "é", "name": "e_acute", "domain": "lang_pt", "latex": "\\'{e}"},
        234: {"char": "ê", "name": "e_circumflex", "domain": "lang_pt", "latex": "\\^{e}"},
        237: {"char": "í", "name": "i_acute", "domain": "lang_pt", "latex": "\\'{i}"},
        243: {"char": "ó", "name": "o_acute", "domain": "lang_pt", "latex": "\\'{o}"},
        245: {"char": "õ", "name": "o_tilde", "domain": "lang_pt", "latex": "\\~{o}"},
        244: {"char": "ô", "name": "o_circumflex", "domain": "lang_pt", "latex": "\\^{o}"},
        250: {"char": "ú", "name": "u_acute", "domain": "lang_pt", "latex": "\\'{u}"},
        # TODO: Add remaining canonical math symbols (~120) following this pattern.
    }

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("/K3D/Knowledge3D.local/galaxies/math")
        self.symbols: Dict[int, MathSymbol] = {}
        self._load_canonical_symbols()

    def _load_canonical_symbols(self) -> None:
        """Load all canonical symbols at startup (always-loaded)."""
        for codepoint, meta in self.CANONICAL_SYMBOLS.items():
            symbol = MathSymbol(
                unicode_codepoint=codepoint,
                char=meta["char"],
                name=meta["name"],
                domain=meta["domain"],
                latex=meta.get("latex", ""),
                pronunciation=meta.get("pronunciation", ""),
                rpn_program=self._generate_rpn_program(meta["char"]),
            )
            self.symbols[codepoint] = symbol

    def _generate_rpn_program(self, char: str) -> str:
        """
        Generate procedural RPN program for drawing a symbol.

        Placeholder until ProceduralCompiler extraction fills real programs.
        Uses real engine opcodes (MOVE, LINE, CLOSE, STROKE).
        """
        return "32 32 MOVE 32 48 LINE 48 48 LINE 48 32 LINE CLOSE STROKE"

    def get(self, codepoint: int) -> Optional[MathSymbol]:
        return self.symbols.get(codepoint)

    def get_by_char(self, char: str) -> Optional[MathSymbol]:
        return self.symbols.get(ord(char))

    def get_by_domain(self, domain: str) -> List[MathSymbol]:
        return [s for s in self.symbols.values() if s.domain == domain]

    def symbol_ref(self, codepoint: int) -> int:
        """Symlink reference to a symbol (canonical codepoint)."""
        if codepoint not in self.symbols:
            raise KeyError(f"Symbol U+{codepoint:04X} not in Math Galaxy")
        return codepoint

    def add_rule_ref(self, codepoint: int, rule_id: str) -> None:
        if symbol := self.symbols.get(codepoint):
            if rule_id not in symbol.rule_refs:
                symbol.rule_refs.append(rule_id)

    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Persist galaxy symbols to storage (JSON).

        Args:
            output_path: Optional override path. Defaults to storage_path/math_galaxy.json
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)
        target = output_path or (self.storage_path / "math_galaxy.json")
        payload = {cp: asdict(sym) for cp, sym in self.symbols.items()}
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# Global singleton — always-loaded
_math_galaxy: Optional[MathGalaxy] = None


def get_math_galaxy() -> MathGalaxy:
    global _math_galaxy
    if _math_galaxy is None:
        _math_galaxy = MathGalaxy()
    return _math_galaxy
