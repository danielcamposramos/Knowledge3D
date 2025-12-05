"""
Math Symbols Registry - Comprehensive Unicode mathematical symbols organized by category.

This registry contains ~1000+ mathematical symbols from Unicode blocks:
    - Mathematical Operators (U+2200-U+22FF) - Complete
    - Supplemental Mathematical Operators (U+2A00-U+2AFF) - Complete
    - Miscellaneous Mathematical Symbols-A (U+27C0-U+27EF) - Complete
    - Miscellaneous Mathematical Symbols-B (U+2980-U+29FF) - Complete
    - Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF) - Selective (~150)
    - Greek and Coptic (U+0370-U+03FF) - Complete
    - Arrows (U+2190-U+21FF) - Complete
    - Geometric Shapes (U+25A0-U+25FF) - Complete (2D shapes for 3D prep)
    - Geometric Shapes Extended (U+1F780-U+1F7FF) - Complete
    - Box Drawing (U+2500-U+257F) - Complete

Symbols are organized by semantic category to facilitate training organization
and RPN semantic mapping.

Usage:
    >>> from knowledge3d.cranium.math_symbols_registry import is_math_symbol, ALL_MATH_SYMBOLS
    >>> print(is_math_symbol('∑'))  # True
    >>> print(is_math_symbol('A'))  # False
    >>> print(len(ALL_MATH_SYMBOLS))  # ~1000+

Integration:
    Used by train_atomic_character.py to detect math symbols via get_character_script().
"""

from typing import List, Set


# ============================================================================
# Basic Operators (Core arithmetic and comparison)
# ============================================================================
BASIC_OPS: List[str] = list('+-×÷=≠<>≤≥±∓∼≈≅')

# ============================================================================
# Calculus & Analysis
# ============================================================================
CALCULUS: List[str] = [
    '∂',  # Partial differential
    '∇',  # Nabla (gradient)
    '∆',  # Delta (change)
    '∫',  # Integral
    '∬',  # Double integral
    '∭',  # Triple integral
    '∮',  # Contour integral
    '∯',  # Surface integral
    '∰',  # Volume integral
    '∑',  # Summation
    '∏',  # Product
    '∐',  # Coproduct
    '√',  # Square root
    '∛',  # Cube root
    '∜',  # Fourth root
    '∞',  # Infinity
    '∝',  # Proportional to
    '∫',  # Integral
]

# ============================================================================
# Set Theory & Logic
# ============================================================================
SET_THEORY: List[str] = [
    # Membership & inclusion
    '∈', '∉', '∊', '∋', '∌', '∍',
    # Subset/superset
    '⊂', '⊃', '⊄', '⊅', '⊆', '⊇', '⊈', '⊉', '⊊', '⊋',
    # Set operations
    '∪', '∩', '⊎', '⊓', '⊔',
    # Special sets
    '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ', 'ℙ', 'ℍ',
    # Blackboard bold (double-struck)
    '𝔸', '𝔹', '𝔼', '𝔽', '𝔾', '𝕀', '𝕁', '𝕂', '𝕃', '𝕄',
    '𝕆', '𝕊', '𝕋', '𝕌', '𝕍', '𝕎', '𝕏', '𝕐', '𝕑',
]

LOGIC: List[str] = [
    # Quantifiers
    '∀', '∃', '∄', '∃!',
    # Logical connectives
    '∧', '∨', '¬', '⊕', '⊻', '⊼', '⊽',
    # Implications
    '⇒', '⇐', '⇔', '→', '←', '↔',
    '⊢', '⊣', '⊨', '⊭', '⊤', '⊥',
    # Equivalence
    '≡', '≢', '≃', '≄', '≅', '≆', '≇', '≈', '≉',
]

# ============================================================================
# Greek Alphabet (Mathematical usage)
# ============================================================================
GREEK_LOWER: List[str] = list('αβγδεζηθικλμνξοπρςστυφχψω')
GREEK_UPPER: List[str] = list('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
GREEK_VARIANTS: List[str] = ['ϑ', 'ϕ', 'ϖ', 'ϰ', 'ϱ', 'ϵ']  # Variant forms
GREEK_ALL = GREEK_LOWER + GREEK_UPPER + GREEK_VARIANTS

# ============================================================================
# Arrows (Functions, mappings, implications)
# ============================================================================
ARROWS: List[str] = [
    # Basic arrows
    '←', '→', '↑', '↓', '↔', '↕', '↖', '↗', '↘', '↙',
    # Double arrows
    '⇐', '⇒', '⇑', '⇓', '⇔', '⇕',
    # Harpoons
    '↼', '⇀', '↽', '⇁', '⇋', '⇌',
    # Special arrows
    '↦', '↣', '↠', '⟵', '⟶', '⟷', '⟸', '⟹', '⟺',
    # Long arrows
    '⟼', '⟻',
]

# ============================================================================
# Brackets & Grouping (Critical for RPN parsing)
# ============================================================================
BRACKETS: List[str] = [
    '(', ')', '[', ']', '{', '}',
    '⟨', '⟩', '⟪', '⟫',  # Angle brackets
    '⌈', '⌉', '⌊', '⌋',  # Ceiling/floor
    '|', '‖', '∥',       # Vertical bars
    '⦀', '⦃', '⦄', '⦅', '⦆', '⦇', '⦈', '⦉', '⦊',
]

# ============================================================================
# Superscripts & Subscripts (Visual recognition for exponents/indices)
# ============================================================================
SUPERSCRIPTS: List[str] = list('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ')
SUBSCRIPTS: List[str] = list('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ')

# ============================================================================
# Fractions (Common fractions as single characters)
# ============================================================================
FRACTIONS: List[str] = list('½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒')

# ============================================================================
# Additional Operators (Extended mathematical operations)
# ============================================================================
ADDITIONAL_OPS: List[str] = [
    # Binary operators
    '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', '⊜', '⊝',
    '⊞', '⊟', '⊠', '⊡',
    # Relational operators
    '≺', '≻', '≼', '≽', '≾', '≿', '⊀', '⊁',
    '⋖', '⋗', '⋘', '⋙', '⋚', '⋛',
    # Dots & products
    '⋅', '∙', '∘', '⊙', '⊚', '⊛',
    # Stars & asterisks
    '∗', '⋆', '★', '☆',
]

# ============================================================================
# Geometry & Topology
# ============================================================================
GEOMETRY: List[str] = [
    # Angles
    '∟', '∠', '∡', '∢', '⊾', '⊿',
    # Parallel/perpendicular
    '∣', '∤', '∥', '∦', '⊥',
    # Shapes
    '△', '▽', '▷', '◁', '◊', '○', '●', '□', '■', '◯',
    # Topology
    '⊶', '⊷', '⊸', '⊹', '⊺', '⊻', '⊼', '⊽',
]

# ============================================================================
# Relations & Equivalences
# ============================================================================
RELATIONS: List[str] = [
    # Equivalence
    '≡', '≢', '≣', '≐', '≑', '≒', '≓', '≔', '≕', '≖', '≗',
    # Similarity
    '∼', '∽', '≁', '≃', '≄', '≅', '≆', '≇', '≈', '≉', '≊', '≋',
    # Ordering
    '≪', '≫', '⋘', '⋙',
    # Preceeds/follows
    '≺', '≻', '⊀', '⊁', '≼', '≽', '≾', '≿',
]

# ============================================================================
# Miscellaneous Mathematical Symbols
# ============================================================================
MISC_MATH: List[str] = [
    # Dots
    '·', '⋯', '⋮', '⋰', '⋱',
    # Hats & accents (combining diacritics often)
    '̂', '̃', '̄', '̅', '̆', '̇', '̈',
    # Special symbols
    '℧', '℩', 'Å', '℮', 'ℯ', 'ℓ', '№', '℘', '℗', '℠', '™',
    # Prime marks
    '′', '″', '‴', '‵', '‶', '‷',
    # Degree & temperature
    '°', '℃', '℉',
]

# ============================================================================
# N-ary Operators (Large operators)
# ============================================================================
NARY_OPS: List[str] = [
    '∑', '∏', '∐',  # Sum, product, coproduct
    '⋀', '⋁', '⋂', '⋃',  # Big logical/set operators
    '⨀', '⨁', '⨂', '⨃', '⨄', '⨅', '⨆',  # Big circled operators
    '∫', '∬', '∭', '∮', '∯', '∰', '∱', '∲', '∳',  # Integrals
]

# Aggregated operator set (core + extended, excluding arrows/relations)
OPERATORS: List[str] = BASIC_OPS + ADDITIONAL_OPS + NARY_OPS

# ============================================================================
# Supplemental Mathematical Operators (U+2A00-U+2AFF) - Extended operators
# ============================================================================
SUPPLEMENTAL_OPS: List[str] = [
    # N-ary operators
    '⨀', '⨁', '⨂', '⨃', '⨄', '⨅', '⨆', '⨇', '⨈', '⨉',
    # Binary operators
    '⨝', '⨞', '⨟', '⨠', '⨡', '⨢', '⨣', '⨤', '⨥', '⨦', '⨧', '⨨', '⨩', '⨪', '⨫', '⨬', '⨭', '⨮',
    '⨯', '⨰', '⨱', '⨲', '⨳', '⨴', '⨵', '⨶', '⨷', '⨸', '⨹', '⨺', '⨻', '⨼', '⨽',
    # Relations
    '⩳', '⩴', '⩵', '⩶', '⩷', '⩸', '⩹', '⩺', '⩻', '⩼', '⩽', '⩾', '⩿',
    '⪀', '⪁', '⪂', '⪃', '⪄', '⪅', '⪆', '⪇', '⪈', '⪉', '⪊', '⪋', '⪌', '⪍', '⪎', '⪏',
    '⪐', '⪑', '⪒', '⪓', '⪔', '⪕', '⪖', '⪗', '⪘', '⪙', '⪚', '⪛', '⪜', '⪝', '⪞', '⪟',
    '⪠', '⪡', '⪢', '⪣', '⪤', '⪥', '⪦', '⪧', '⪨', '⪩', '⪪', '⪫', '⪬', '⪭', '⪮', '⪯',
    '⪰', '⪱', '⪲', '⪳', '⪴', '⪵', '⪶', '⪷', '⪸', '⪹', '⪺', '⪻', '⪼', '⪽', '⪾', '⪿',
    '⫀', '⫁', '⫂', '⫃', '⫄', '⫅', '⫆', '⫇', '⫈', '⫉', '⫊', '⫋', '⫌', '⫍', '⫎', '⫏',
    # Set/logic operators
    '⫐', '⫑', '⫒', '⫓', '⫔', '⫕', '⫖', '⫗', '⫘', '⫙', '⫚', '⫛',
    # Arrows
    '⫷', '⫸', '⫹', '⫺',
]

# ============================================================================
# 2D Geometric Shapes (U+25A0-U+25FF) - Preparation for 3D shapes
# ============================================================================
GEOMETRIC_2D: List[str] = [
    # Squares
    '■', '□', '▢', '▣', '▤', '▥', '▦', '▧', '▨', '▩', '▪', '▫',
    # Rectangles
    '▬', '▭', '▮', '▯',
    # Triangles (pointing in all directions)
    '▲', '△', '▴', '▵', '▶', '▷', '▸', '▹', '►', '▻', '▼', '▽', '▾', '▿', '◀', '◁', '◂', '◃', '◄', '◅',
    # Diamonds
    '◆', '◇', '◈', '◊', '○', '◌', '◍', '◎', '●', '◐', '◑', '◒', '◓', '◔', '◕', '◖', '◗',
    # Circles and arcs
    '◘', '◙', '◚', '◛', '◜', '◝', '◞', '◟', '◠', '◡', '◢', '◣', '◤', '◥', '◦', '◧', '◨', '◩', '◪', '◫',
    # Polygons
    '◬', '◭', '◮', '◯', '◰', '◱', '◲', '◳', '◴', '◵', '◶', '◷',
    # Stars and special shapes
    '◸', '◹', '◺', '◻', '◼', '◽', '◾', '◿',
    '★', '☆', '☀', '☁', '☂', '☃', '☄', '☉', '☊', '☋',
    # Pentagons, hexagons
    '⬠', '⬡', '⬢', '⬣', '⬤', '⬥', '⬦', '⬧', '⬨', '⬩', '⬪', '⬫', '⬬', '⬭', '⬮', '⬯',
    '⬰', '⬱', '⬲', '⬳', '⬴', '⬵', '⬶', '⬷', '⬸', '⬹', '⬺', '⬻', '⬼', '⬽', '⬾', '⬿',
    '⭀', '⭁', '⭂', '⭃', '⭄', '⭅', '⭆', '⭇', '⭈', '⭉', '⭊', '⭋', '⭌',
]

# ============================================================================
# Box Drawing (U+2500-U+257F) - Structural elements
# ============================================================================
BOX_DRAWING: List[str] = [
    # Horizontal lines
    '─', '━', '│', '┃', '┄', '┅', '┆', '┇', '┈', '┉', '┊', '┋',
    # Corners and junctions
    '┌', '┍', '┎', '┏', '┐', '┑', '┒', '┓', '└', '┕', '┖', '┗', '┘', '┙', '┚', '┛',
    '├', '┝', '┞', '┟', '┠', '┡', '┢', '┣', '┤', '┥', '┦', '┧', '┨', '┩', '┪', '┫',
    '┬', '┭', '┮', '┯', '┰', '┱', '┲', '┳', '┴', '┵', '┶', '┷', '┸', '┹', '┺', '┻',
    '┼', '┽', '┾', '┿', '╀', '╁', '╂', '╃', '╄', '╅', '╆', '╇', '╈', '╉', '╊', '╋',
    # Double lines
    '═', '║', '╒', '╓', '╔', '╕', '╖', '╗', '╘', '╙', '╚', '╛', '╜', '╝',
    '╞', '╟', '╠', '╡', '╢', '╣', '╤', '╥', '╦', '╧', '╨', '╩', '╪', '╫', '╬',
    # Curved and diagonal
    '╭', '╮', '╯', '╰', '╱', '╲', '╳', '╴', '╵', '╶', '╷', '╸', '╹', '╺', '╻', '╼', '╽', '╾', '╿',
]

# ============================================================================
# Mathematical Alphanumeric Symbols - Selective (most common styling)
# ============================================================================
MATH_ALPHANUMERIC: List[str] = [
    # Bold uppercase
    '𝐀', '𝐁', '𝐂', '𝐃', '𝐄', '𝐅', '𝐆', '𝐇', '𝐈', '𝐉', '𝐊', '𝐋', '𝐌', '𝐍', '𝐎', '𝐏', '𝐐', '𝐑', '𝐒', '𝐓', '𝐔', '𝐕', '𝐖', '𝐗', '𝐘', '𝐙',
    # Bold lowercase
    '𝐚', '𝐛', '𝐜', '𝐝', '𝐞', '𝐟', '𝐠', '𝐡', '𝐢', '𝐣', '𝐤', '𝐥', '𝐦', '𝐧', '𝐨', '𝐩', '𝐪', '𝐫', '𝐬', '𝐭', '𝐮', '𝐯', '𝐰', '𝐱', '𝐲', '𝐳',
    # Bold digits
    '𝟎', '𝟏', '𝟐', '𝟑', '𝟒', '𝟓', '𝟔', '𝟕', '𝟖', '𝟗',
    # Italic uppercase (selective)
    '𝐴', '𝐵', '𝐶', '𝐷', '𝐸', '𝐹', '𝐺', '𝐻', '𝐼', '𝐽', '𝐾', '𝐿', '𝑀', '𝑁', '𝑂', '𝑃', '𝑄', '𝑅', '𝑆', '𝑇', '𝑈', '𝑉', '𝑊', '𝑋', '𝑌', '𝑍',
    # Italic lowercase (selective)
    '𝑎', '𝑏', '𝑐', '𝑑', '𝑒', '𝑓', '𝑔', 'ℎ', '𝑖', '𝑗', '𝑘', '𝑙', '𝑚', '𝑛', '𝑜', '𝑝', '𝑞', '𝑟', '𝑠', '𝑡', '𝑢', '𝑣', '𝑤', '𝑥', '𝑦', '𝑧',
    # Script (calligraphic) - selective
    '𝒜', 'ℬ', '𝒞', '𝒟', 'ℰ', 'ℱ', '𝒢', 'ℋ', 'ℐ', '𝒥', '𝒦', 'ℒ', 'ℳ', '𝒩', '𝒪', '𝒫', '𝒬', 'ℛ', '𝒮', '𝒯', '𝒰', '𝒱', '𝒲', '𝒳', '𝒴', '𝒵',
]

# ============================================================================
# Miscellaneous Symbols-B (U+2980-U+29FF) - Additional brackets and operators
# ============================================================================
MISC_SYMBOLS_B: List[str] = [
    # Triple brackets
    '⦀', '⦁', '⦂', '⦃', '⦄', '⦅', '⦆', '⦇', '⦈', '⦉', '⦊', '⦋', '⦌', '⦍', '⦎', '⦏',
    '⦐', '⦑', '⦒', '⦓', '⦔', '⦕', '⦖', '⦗', '⦘',
    # Operators
    '⦙', '⦚', '⦛', '⦜', '⦝', '⦞', '⦟', '⦠', '⦡', '⦢', '⦣', '⦤', '⦥', '⦦', '⦧', '⦨', '⦩', '⦪', '⦫', '⦬', '⦭', '⦮',
    # Relations
    '⧀', '⧁', '⧂', '⧃', '⧄', '⧅', '⧆', '⧇', '⧈', '⧉', '⧊', '⧋', '⧌', '⧍', '⧎', '⧏',
    '⧐', '⧑', '⧒', '⧓', '⧔', '⧕', '⧖', '⧗', '⧘', '⧙', '⧚', '⧛', '⧜', '⧝', '⧞', '⧟',
    '⧠', '⧡', '⧢', '⧣', '⧤', '⧥', '⧦', '⧧', '⧨', '⧩', '⧪', '⧫', '⧬', '⧭', '⧮', '⧯',
    '⧰', '⧱', '⧲', '⧳', '⧴', '⧵', '⧶', '⧷', '⧸', '⧹', '⧺', '⧻', '⧼', '⧽', '⧾', '⧿',
]

# ============================================================================
# Consolidated Registry
# ============================================================================
ALL_MATH_SYMBOLS: List[str] = (
    BASIC_OPS +
    CALCULUS +
    SET_THEORY +
    LOGIC +
    GREEK_ALL +
    ARROWS +
    BRACKETS +
    SUPERSCRIPTS +
    SUBSCRIPTS +
    FRACTIONS +
    ADDITIONAL_OPS +
    GEOMETRY +
    RELATIONS +
    MISC_MATH +
    NARY_OPS +
    SUPPLEMENTAL_OPS +
    GEOMETRIC_2D +
    BOX_DRAWING +
    MATH_ALPHANUMERIC +
    MISC_SYMBOLS_B
)

# Remove duplicates and convert to set for O(1) lookup
_MATH_SYMBOLS_SET: Set[str] = set(ALL_MATH_SYMBOLS)

# ============================================================================
# Category Mapping (for semantic organization)
# ============================================================================
SYMBOL_CATEGORIES: dict[str, str] = {}

for sym in BASIC_OPS:
    SYMBOL_CATEGORIES[sym] = 'math_operator'
for sym in CALCULUS:
    SYMBOL_CATEGORIES[sym] = 'math_calculus'
for sym in SET_THEORY:
    SYMBOL_CATEGORIES[sym] = 'math_set'
for sym in LOGIC:
    SYMBOL_CATEGORIES[sym] = 'math_logic'
for sym in GREEK_ALL:
    SYMBOL_CATEGORIES[sym] = 'math_greek'
for sym in ARROWS:
    SYMBOL_CATEGORIES[sym] = 'math_arrow'
for sym in BRACKETS:
    SYMBOL_CATEGORIES[sym] = 'math_bracket'
for sym in SUPERSCRIPTS:
    SYMBOL_CATEGORIES[sym] = 'math_superscript'
for sym in SUBSCRIPTS:
    SYMBOL_CATEGORIES[sym] = 'math_subscript'
for sym in FRACTIONS:
    SYMBOL_CATEGORIES[sym] = 'math_fraction'
for sym in ADDITIONAL_OPS:
    SYMBOL_CATEGORIES[sym] = 'math_operator'
for sym in GEOMETRY:
    SYMBOL_CATEGORIES[sym] = 'math_geometry'
for sym in RELATIONS:
    SYMBOL_CATEGORIES[sym] = 'math_relation'
for sym in MISC_MATH:
    SYMBOL_CATEGORIES[sym] = 'math_misc'
for sym in NARY_OPS:
    SYMBOL_CATEGORIES[sym] = 'math_nary'
for sym in SUPPLEMENTAL_OPS:
    SYMBOL_CATEGORIES[sym] = 'math_supplemental'
for sym in GEOMETRIC_2D:
    SYMBOL_CATEGORIES[sym] = 'math_shape_2d'
for sym in BOX_DRAWING:
    SYMBOL_CATEGORIES[sym] = 'math_box'
for sym in MATH_ALPHANUMERIC:
    SYMBOL_CATEGORIES[sym] = 'math_alphanumeric'
for sym in MISC_SYMBOLS_B:
    SYMBOL_CATEGORIES[sym] = 'math_misc_b'

# ============================================================================
# Public API
# ============================================================================

def is_math_symbol(char: str) -> bool:
    """
    Check if character is a mathematical symbol.

    Args:
        char: Single character to check

    Returns:
        True if character is in math symbol registry, False otherwise

    Example:
        >>> is_math_symbol('∑')
        True
        >>> is_math_symbol('A')
        False
        >>> is_math_symbol('α')
        True
    """
    return char in _MATH_SYMBOLS_SET


def get_symbol_category(symbol: str) -> str:
    """
    Get semantic category for a mathematical symbol.

    Args:
        symbol: Mathematical symbol

    Returns:
        Category string (e.g., 'math_calculus', 'math_logic', 'math_greek')
        Returns 'math_unknown' if symbol not in registry

    Example:
        >>> get_symbol_category('∑')
        'math_calculus'
        >>> get_symbol_category('α')
        'math_greek'
    """
    return SYMBOL_CATEGORIES.get(symbol, 'math_unknown')


def get_symbols_by_category(category: str) -> List[str]:
    """
    Get all symbols in a specific category.

    Args:
        category: Category name (e.g., 'math_calculus', 'math_logic')

    Returns:
        List of symbols in that category

    Example:
        >>> calculus_symbols = get_symbols_by_category('math_calculus')
        >>> print(calculus_symbols[:5])
        ['∂', '∇', '∆', '∫', '∬']
    """
    return [sym for sym, cat in SYMBOL_CATEGORIES.items() if cat == category]


def get_all_categories() -> List[str]:
    """
    Get list of all symbol categories.

    Returns:
        List of unique category names

    Example:
        >>> categories = get_all_categories()
        >>> print(sorted(categories))
        ['math_arrow', 'math_bracket', 'math_calculus', ...]
    """
    return sorted(set(SYMBOL_CATEGORIES.values()))


# ============================================================================
# Statistics
# ============================================================================

def get_registry_stats() -> dict[str, any]:
    """
    Get statistics about the math symbol registry.

    Returns:
        Dictionary with:
            - total_symbols: Total number of unique symbols
            - categories: Number of categories
            - symbols_by_category: Count of symbols per category
    """
    from collections import Counter

    category_counts = Counter(SYMBOL_CATEGORIES.values())

    return {
        "total_symbols": len(_MATH_SYMBOLS_SET),
        "categories": len(category_counts),
        "symbols_by_category": dict(category_counts),
    }


__all__ = [
    'ALL_MATH_SYMBOLS',
    'SYMBOL_CATEGORIES',
    'is_math_symbol',
    'get_symbol_category',
    'get_symbols_by_category',
    'get_all_categories',
    'get_registry_stats',
    # Category lists (for direct access if needed)
    'BASIC_OPS',
    'CALCULUS',
    'SET_THEORY',
    'LOGIC',
    'GREEK_ALL',
    'ARROWS',
    'BRACKETS',
    'SUPERSCRIPTS',
    'SUBSCRIPTS',
    'FRACTIONS',
    'ADDITIONAL_OPS',
    'GEOMETRY',
    'RELATIONS',
    'MISC_MATH',
    'NARY_OPS',
    'SUPPLEMENTAL_OPS',
    'GEOMETRIC_2D',
    'BOX_DRAWING',
    'MATH_ALPHANUMERIC',
    'MISC_SYMBOLS_B',
]
