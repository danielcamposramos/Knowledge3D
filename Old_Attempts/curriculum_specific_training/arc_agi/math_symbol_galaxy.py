"""
Math Symbol Galaxy - LaTeX symbols with RPN meanings.

Each symbol is a Galaxy entry that the model can "see" and compose into RPN.
This is sovereign: the symbols ARE the model's math knowledge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MathSymbol:
    """A math symbol with its RPN meaning."""

    symbol: str  # The symbol/command (e.g., "\\frac", "!", "^")
    category: str  # operator, function, constant, delimiter, relation
    arity: int  # Number of arguments (0=constant, 1=unary, 2=binary)
    rpn_template: str  # RPN program template with {0}, {1} placeholders
    precedence: int  # For infix operators (higher = binds tighter)
    associativity: str  # "left", "right", or "none"
    description: str  # Human-readable description
    # Symlink/variant forms that refer to the same meaning (Dual Client Contract).
    # Examples: "\\cos" <-> "cos" <-> "cosine" <-> "⁡cos"
    variants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] | None = None


# =============================================================================
# MATH SYMBOL GALAXY - Core entries
# =============================================================================

DEFAULT_SYMBOL_VARIANTS: Dict[str, List[str]] = {
    # Trig functions
    "\\sin": ["sin", "sine"],
    "\\cos": ["cos", "cosine"],
    "\\tan": ["tan", "tangent"],
    # Common functions
    "\\sqrt": ["sqrt", "square root", "√"],
    "\\log": ["log", "ln"],
    "\\ln": ["ln", "log"],
    "\\exp": ["exp", "exponential"],
    "\\abs": ["abs", "absolute value", "| |"],
    "\\floor": ["floor"],
    "\\ceil": ["ceil", "ceiling"],
    "\\gcd": ["gcd"],
    "\\lcm": ["lcm"],
    # Constants / greek letters (be conservative: don't merge uppercase forms)
    "\\pi": ["pi", "π"],
    # Operators / relations (keep minimal to avoid ambiguity)
    "\\sum": ["sum", "∑", "Σ"],
}

MATH_SYMBOLS: List[MathSymbol] = [
    # ===== ARITHMETIC OPERATORS =====
    MathSymbol(
        symbol="+",
        category="operator",
        arity=2,
        rpn_template="{0} {1} +",
        precedence=1,
        associativity="left",
        description="Addition",
    ),
    MathSymbol(
        symbol="-",
        category="operator",
        arity=2,
        rpn_template="{0} {1} -",
        precedence=1,
        associativity="left",
        description="Subtraction",
    ),
    MathSymbol(
        symbol="*",
        category="operator",
        arity=2,
        rpn_template="{0} {1} *",
        precedence=2,
        associativity="left",
        description="Multiplication",
    ),
    MathSymbol(
        symbol="/",
        category="operator",
        arity=2,
        rpn_template="{0} {1} /",
        precedence=2,
        associativity="left",
        description="Division",
    ),
    MathSymbol(
        symbol="^",
        category="operator",
        arity=2,
        rpn_template="{0} {1} pow",
        precedence=3,
        associativity="right",
        description="Exponentiation",
    ),
    MathSymbol(
        symbol="!",
        category="operator",
        arity=1,
        rpn_template="{0} factorial",
        precedence=4,
        associativity="left",
        description="Factorial (postfix)",
    ),
    MathSymbol(
        symbol="%",
        category="operator",
        arity=2,
        rpn_template="{0} {1} mod",
        precedence=2,
        associativity="left",
        description="Modulo",
    ),
    # ===== LATEX COMMANDS =====
    MathSymbol(
        symbol="\\frac",
        category="function",
        arity=2,
        rpn_template="{0} {1} /",
        precedence=0,
        associativity="none",
        description="Fraction a/b",
    ),
    MathSymbol(
        symbol="\\binom",
        category="function",
        arity=2,
        rpn_template="{0} {1} binomial",
        precedence=0,
        associativity="none",
        description="Binomial coefficient C(n,k)",
    ),
    MathSymbol(
        symbol="\\sqrt",
        category="function",
        arity=1,
        rpn_template="{0} sqrt",
        precedence=0,
        associativity="none",
        description="Square root",
    ),
    MathSymbol(
        symbol="\\sin",
        category="function",
        arity=1,
        rpn_template="{0} sin",
        precedence=0,
        associativity="none",
        description="Sine",
    ),
    MathSymbol(
        symbol="\\cos",
        category="function",
        arity=1,
        rpn_template="{0} cos",
        precedence=0,
        associativity="none",
        description="Cosine",
    ),
    MathSymbol(
        symbol="\\tan",
        category="function",
        arity=1,
        rpn_template="{0} tan",
        precedence=0,
        associativity="none",
        description="Tangent",
    ),
    MathSymbol(
        symbol="\\log",
        category="function",
        arity=1,
        rpn_template="{0} log",
        precedence=0,
        associativity="none",
        description="Natural logarithm",
    ),
    MathSymbol(
        symbol="\\ln",
        category="function",
        arity=1,
        rpn_template="{0} log",
        precedence=0,
        associativity="none",
        description="Natural logarithm",
    ),
    MathSymbol(
        symbol="\\exp",
        category="function",
        arity=1,
        rpn_template="{0} exp",
        precedence=0,
        associativity="none",
        description="Exponential e^x",
    ),
    MathSymbol(
        symbol="\\abs",
        category="function",
        arity=1,
        rpn_template="{0} abs",
        precedence=0,
        associativity="none",
        description="Absolute value",
    ),
    MathSymbol(
        symbol="\\floor",
        category="function",
        arity=1,
        rpn_template="{0} floor",
        precedence=0,
        associativity="none",
        description="Floor function",
    ),
    MathSymbol(
        symbol="\\ceil",
        category="function",
        arity=1,
        rpn_template="{0} ceil",
        precedence=0,
        associativity="none",
        description="Ceiling function",
    ),
    MathSymbol(
        symbol="\\gcd",
        category="function",
        arity=2,
        rpn_template="{0} {1} gcd",
        precedence=0,
        associativity="none",
        description="Greatest common divisor",
    ),
    MathSymbol(
        symbol="\\lcm",
        category="function",
        arity=2,
        rpn_template="{0} {1} * {0} {1} gcd /",
        precedence=0,
        associativity="none",
        description="Least common multiple",
    ),
    # ===== CONSTANTS =====
    MathSymbol(
        symbol="\\pi",
        category="constant",
        arity=0,
        rpn_template="3.14159265358979",
        precedence=0,
        associativity="none",
        description="Pi",
    ),
    MathSymbol(
        symbol="e",
        category="constant",
        arity=0,
        rpn_template="2.71828182845905",
        precedence=0,
        associativity="none",
        description="Euler's number",
    ),
    # ===== COMPARISON =====
    MathSymbol(
        symbol="=",
        category="relation",
        arity=2,
        rpn_template="{0} {1} eq",
        precedence=0,
        associativity="none",
        description="Equality",
    ),
    MathSymbol(
        symbol=">",
        category="relation",
        arity=2,
        rpn_template="{0} {1} gt",
        precedence=0,
        associativity="none",
        description="Greater than",
    ),
    MathSymbol(
        symbol="<",
        category="relation",
        arity=2,
        rpn_template="{0} {1} lt",
        precedence=0,
        associativity="none",
        description="Less than",
    ),
    MathSymbol(
        symbol="\\geq",
        category="relation",
        arity=2,
        rpn_template="{0} {1} gte",
        precedence=0,
        associativity="none",
        description="Greater than or equal",
    ),
    MathSymbol(
        symbol="\\leq",
        category="relation",
        arity=2,
        rpn_template="{0} {1} lt",
        precedence=0,
        associativity="none",
        description="Less than or equal",
    ),
]


class MathSymbolGalaxy:
    """
    Galaxy of math symbols with RPN meanings.

    The model "sees" symbols and looks them up here to compose RPN programs.
    This IS the model's math knowledge - sovereign, no external preprocessing.
    """

    def __init__(self, symbols: Optional[List[MathSymbol]] = None):
        self._symbols: Dict[str, MathSymbol] = {}
        self._by_category: Dict[str, List[MathSymbol]] = {}
        # Variant symlink registry: many surface forms map to one meaning.
        self._variant_to_canonical: Dict[str, str] = {}
        self._canonical_to_variants: Dict[str, set[str]] = {}
        for symbol in (symbols or MATH_SYMBOLS):
            self.add_symbol(symbol)

    def __len__(self) -> int:
        return len(self._symbols)

    def lookup(self, symbol: str) -> Optional[MathSymbol]:
        """Look up a symbol's RPN meaning."""
        canonical = self._variant_to_canonical.get(symbol, symbol)
        return self._symbols.get(canonical)

    def canonical_symbol(self, symbol: str) -> Optional[str]:
        """Return the canonical key for `symbol` if registered, else None."""
        if symbol in self._symbols:
            return symbol
        return self._variant_to_canonical.get(symbol)

    def variants_for(self, symbol: str) -> List[str]:
        """
        Return known variants for the meaning referred to by `symbol`.

        This enables retrieval-time symlink expansion without doing ad-hoc
        string normalization: the mapping lives in the Galaxy.
        """
        canonical = self.canonical_symbol(symbol)
        if canonical is None:
            return []
        variants = self._canonical_to_variants.get(canonical)
        if not variants:
            return [canonical]
        # Stable ordering for deterministic tests.
        return sorted(variants)

    def add_symbol(self, symbol: MathSymbol) -> None:
        """
        Add or update a symbol in the galaxy.

        This is intentionally mutable: TRM/ingestion can expand the galaxy over time.
        """
        existing = self._symbols.get(symbol.symbol)
        if existing is not None:
            # Remove from old category bucket to prevent duplicates.
            old_bucket = self._by_category.get(existing.category, [])
            self._by_category[existing.category] = [s for s in old_bucket if s.symbol != existing.symbol]

        self._symbols[symbol.symbol] = symbol
        self._by_category.setdefault(symbol.category, []).append(symbol)
        self._register_symbol_variants(symbol)

    def _register_variant(self, *, canonical: str, variant: str) -> None:
        v = str(variant or "").strip()
        if not v:
            return
        prev = self._variant_to_canonical.get(v)
        if prev is not None and prev != canonical:
            # Ambiguous surface forms MUST NOT collapse meanings.
            return
        self._variant_to_canonical[v] = canonical
        self._canonical_to_variants.setdefault(canonical, set()).add(v)

    def _register_symbol_variants(self, symbol: MathSymbol) -> None:
        canonical = symbol.symbol
        self._register_variant(canonical=canonical, variant=canonical)

        # Explicit variants from definition.
        for v in (symbol.variants or []):
            self._register_variant(canonical=canonical, variant=v)

        # Standard variants for core symbols (kept small + curated).
        for v in DEFAULT_SYMBOL_VARIANTS.get(canonical, []):
            self._register_variant(canonical=canonical, variant=v)

        # Auto-variant: "\cos" -> "cos"
        if canonical.startswith("\\") and len(canonical) > 1 and canonical[1:].isalpha():
            self._register_variant(canonical=canonical, variant=canonical[1:])

        # If we have a canonical unicode glyph in metadata, register it.
        if symbol.metadata:
            char = symbol.metadata.get("char")
            if isinstance(char, str) and char:
                self._register_variant(canonical=canonical, variant=char)
            latex = symbol.metadata.get("latex")
            if isinstance(latex, str) and latex and latex != canonical:
                self._register_variant(canonical=canonical, variant=latex)

    def get_rpn_template(self, symbol: str) -> Optional[str]:
        """Get RPN template for a symbol."""
        entry = self.lookup(symbol)
        return entry.rpn_template if entry else None

    def compose_rpn(self, symbol: str, *args: object) -> str:
        """Compose RPN program from symbol and arguments."""
        entry = self.lookup(symbol)
        if not entry:
            return ""
        template = entry.rpn_template
        for i, arg in enumerate(args):
            template = template.replace(f"{{{i}}}", str(arg))
        return template

    def all_symbols(self) -> List[MathSymbol]:
        """Return all registered symbols."""
        return list(self._symbols.values())

    def symbols_by_category(self, category: str) -> List[MathSymbol]:
        """Return symbols in a category."""
        return self._by_category.get(category, [])

    def query_semantic(self, query: str, k: int = 5) -> List[MathSymbol]:
        """
        Lightweight semantic lookup over symbols without external dependencies.

        For now this is a deterministic heuristic (substring + token overlap) that
        keeps the hot path sovereign. When/if a VRAM spatial index is available,
        this method can be upgraded to use it.
        """
        q = (query or "").strip().lower()
        if not q:
            return self.all_symbols()[:k]

        q_tokens = {t for t in q.replace("\\", " ").replace("_", " ").split() if t}
        scored: List[tuple[float, MathSymbol]] = []
        for sym in self._symbols.values():
            hay = f"{sym.symbol} {sym.category} {sym.description}".lower()
            score = 0.0
            if q in hay:
                score += 3.0
            for tok in q_tokens:
                if tok in hay:
                    score += 1.0
            # Prefer higher-arity functions when query mentions function-ish cues.
            if any(tok in q_tokens for tok in ("function", "formula", "rule")):
                score += 0.1 * float(sym.arity)
            if score > 0.0:
                scored.append((score, sym))

        scored.sort(key=lambda x: (-x[0], x[1].symbol))
        return [s for _, s in scored[: max(1, int(k))]]


# =============================================================================
# LOAD EXTENDED SYMBOLS FROM CRANIUM MATH GALAXY
# =============================================================================


def _infer_arity(template: str) -> int:
    """Infer arity from placeholder usage."""
    if "{1}" in template:
        return 2
    if "{0}" in template:
        return 1
    return 0


def _load_extended_symbols() -> None:
    """
    Load additional symbols from cranium/math_galaxy.py and add RPN templates.

    The MathGalaxy has 120+ Unicode symbols with LaTeX mappings.
    We add lightweight RPN templates where possible to expose computation.
    """
    try:
        from knowledge3d.cranium.math_galaxy import get_math_galaxy
    except ImportError:
        return

    name_rpn_map = {
        # Basic arithmetic/relations
        "times": "{0} {1} *",
        "divide": "{0} {1} /",
        "dot": "{0} {1} dot",
        "proportional": "{0} {1} /",
        "less_equal": "{0} {1} lt",
        "greater_equal": "{0} {1} gte",
        "not_equal": "{0} {1} eq",
        "approximately": "{0} {1} eq",
        "identical": "{0} {1} eq",
        "precedes": "{0} {1} lt",
        "succeeds": "{0} {1} gt",
        "precedes_eq": "{0} {1} lt",
        "succeeds_eq": "{0} {1} gt",
        # Roots/exponentials
        "sqrt": "{0} sqrt",
        "cube_root": "{0} sqrt",
        "fourth_root": "{0} sqrt",
        # Constants
        "pi": "3.14159265358979",
        # Logic
        "forall": "forall",
        "exists": "exists",
        "negation": "{0} not",
        "logical_and": "{0} {1} and",
        "logical_or": "{0} {1} or",
        "implies": "{0} {1} implies",
        # Set operators (placeholders; not executable yet)
        "union": "",
        "intersection": "",
        "subset": "",
        "superset": "",
    }

    cranium_galaxy = get_math_galaxy()
    existing_symbols = {s.symbol for s in MATH_SYMBOLS}

    for symbol in cranium_galaxy.symbols.values():
        latex = symbol.latex or symbol.char
        if not latex:
            continue

        if latex in existing_symbols:
            continue

        template = name_rpn_map.get(symbol.name, "")
        arity = _infer_arity(template)
        category = symbol.domain.replace("math_", "") if symbol.domain.startswith("math_") else symbol.domain

        new_symbol = MathSymbol(
            symbol=latex,
            category=category,
            arity=arity,
            rpn_template=template,
            precedence=0,
            associativity="none",
            description=symbol.name.replace("_", " ").title(),
            metadata={
                "unicode_codepoint": int(getattr(symbol, "unicode_codepoint", 0) or 0),
                "char": str(getattr(symbol, "char", "") or ""),
                "latex": str(getattr(symbol, "latex", "") or ""),
                "name": str(getattr(symbol, "name", "") or ""),
                "domain": str(getattr(symbol, "domain", "") or ""),
            },
        )
        MATH_SYMBOLS.append(new_symbol)
        existing_symbols.add(latex)


# Load extended symbols on module import and recreate global instance
_load_extended_symbols()
MATH_GALAXY = MathSymbolGalaxy()
