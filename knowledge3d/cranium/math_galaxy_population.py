"""
Math Galaxy population helpers for multi-domain role extraction (Phase 2).

This module exposes validated domain signatures, role patterns, and
single-step theorem patterns. Multi-step chaining remains a Grammar
Galaxy concern and is not derived here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Domain signatures (extracted from sovereign_knowledge_articulator.py).
# Note: equation keywords are lowercase to match normalized equation text.
DOMAIN_SIGNATURES: List[Dict[str, Any]] = [
    {
        "domain": "geometry",
        "context_keywords": [
            "circle",
            "triangle",
            "rectangle",
            "sphere",
            "cylinder",
            "cone",
            "polygon",
            "angle",
            "perpendicular",
            "parallel",
            "tangent",
            "area",
            "volume",
            "perimeter",
            "circumference",
        ],
        "equation_keywords": ["pi", "\u03c0"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
    {
        "domain": "linear_algebra",
        "context_keywords": [
            "matrix",
            "vector",
            "determinant",
            "eigenvalue",
            "eigenvector",
            "linear",
            "subspace",
            "basis",
            "span",
            "dimension",
            "rank",
            "orthogonal",
            "projection",
        ],
        "equation_keywords": ["det", "trace", "rank", "||", "\u2225"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
    {
        "domain": "calculus",
        "context_keywords": [
            "derivative",
            "integral",
            "limit",
            "differential",
            "gradient",
            "rate of change",
            "tangent line",
            "area under",
            "accumulation",
            "converge",
        ],
        "equation_keywords": ["d/d", "\u2202", "\u222b", "lim"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
    {
        "domain": "physics",
        "context_keywords": [
            "velocity",
            "acceleration",
            "force",
            "energy",
            "momentum",
            "electric",
            "magnetic",
            "wave",
            "frequency",
            "mass",
            "pressure",
            "temperature",
        ],
        "equation_keywords": ["f=ma"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
    {
        "domain": "number_theory",
        "context_keywords": [
            "prime",
            "divisor",
            "factor",
            "gcd",
            "lcm",
            "modulo",
            "congruence",
            "integer",
            "rational",
            "irrational",
        ],
        "equation_keywords": ["mod", "\u2261", "gcd", "lcm"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
    {
        "domain": "statistics",
        "context_keywords": [
            "probability",
            "random",
            "distribution",
            "mean",
            "variance",
            "standard deviation",
            "expected value",
            "sample",
            "population",
        ],
        "equation_keywords": ["p(", "e[", "e(", "var(", "var[", "\u03c3", "\u03bc"],
        "weight_context": 1.0,
        "weight_equation": 2.0,
        "weight_book_metadata": 10.0,
    },
]


def populate_domain_signatures(math_galaxy: Optional[object] = None) -> List[Dict[str, Any]]:
    """
    Populate Math Galaxy with 6 domain signatures.

    This is a placeholder until RPN storage is implemented. If the math_galaxy
    object exposes a `domain_signatures` attribute or `add_domain_signature`
    method, this function will use it; otherwise it returns the signatures.
    """
    if math_galaxy is None:
        return list(DOMAIN_SIGNATURES)

    if hasattr(math_galaxy, "add_domain_signature"):
        for sig in DOMAIN_SIGNATURES:
            math_galaxy.add_domain_signature(sig)
        return list(DOMAIN_SIGNATURES)

    if hasattr(math_galaxy, "domain_signatures"):
        try:
            math_galaxy.domain_signatures = list(DOMAIN_SIGNATURES)
        except Exception:
            pass
    return list(DOMAIN_SIGNATURES)


_GENERIC_ROLES = {"unknown"}
_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "for", "from", "by", "with", "as", "at",
    "is", "are", "be", "let", "suppose", "assume", "if", "then", "where", "given",
    "that", "this", "these", "those", "we", "our", "any", "all", "such", "there",
    "it", "its", "into", "also", "an", "a", "on", "over", "under", "between",
}
_THEOREM_CONTEXT_STOPWORDS = _STOPWORDS | {"x", "y", "z", "f", "g", "h"}


def _iter_artifacts(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for base in paths:
        for artifacts_file in base.rglob("artifacts.jsonl"):
            try:
                with artifacts_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
            except FileNotFoundError:
                continue


def _tokenize_context(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tokenize_theorem_context(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _THEOREM_CONTEXT_STOPWORDS]


def _equation_patterns(lhs: Optional[str], rhs: Optional[str], conclusion: Optional[str]) -> List[str]:
    patterns: List[str] = []
    if lhs:
        lhs_clean = str(lhs).strip()
        if lhs_clean:
            patterns.append(f"{lhs_clean} =")
    eq_text = str(conclusion or "")
    if not eq_text and lhs and rhs:
        eq_text = f"{lhs} = {rhs}"
    if eq_text:
        for token in re.findall(r"[A-Za-z]{1,3}|π|∂|∫|≡|\\|\\||∥", eq_text):
            patterns.append(token.lower())
    return patterns


def _variable_conventions(var_mapping: Dict[str, str], placeholder: str) -> List[str]:
    names = {placeholder}
    for orig, ph in (var_mapping or {}).items():
        if ph == placeholder:
            names.add(str(orig))
    cleaned = []
    for name in names:
        token = str(name).strip()
        if token:
            cleaned.append(token)
    return cleaned


def _role_tier(role: str) -> int:
    tier2 = {
        "exponent",
        "base",
        "coefficient",
        "constant_factor",
        "numerator",
        "denominator",
        "radicand",
        "index",
        "argument",
        "parameter",
    }
    tier3 = {"constant", "variable", "placeholder", "unknown"}
    if role in tier2:
        return 2
    if role in tier3:
        return 3
    return 1


_THEOREM_PATTERN_DEFS: List[Dict[str, Any]] = [
    {
        "pattern_id": "power_rule_polynomial",
        "domain": "calculus",
        "context_keywords": ["power rule", "derivative", "differentiate", "d/dx"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"x\s*\^",
        "semantic_tags": ["derivative", "polynomial", "power_rule"],
        "grammar_rule": "apply_power_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_N",
            "PUSH_X",
            "PUSH_N",
            "PUSH_1",
            "SUB",
            "POW",
            "MULT",
        ],
    },
    {
        "pattern_id": "product_rule",
        "domain": "calculus",
        "context_keywords": ["product rule", "derivative", "differentiate"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"\+|plus",
        "semantic_tags": ["derivative", "product_rule"],
        "grammar_rule": "apply_product_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_F",
            "DERIVATIVE",
            "PUSH_G",
            "MULT",
            "PUSH_F",
            "PUSH_G",
            "DERIVATIVE",
            "MULT",
            "ADD",
        ],
    },
    {
        "pattern_id": "quotient_rule",
        "domain": "calculus",
        "context_keywords": ["quotient rule", "derivative", "differentiate"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"/|over",
        "semantic_tags": ["derivative", "quotient_rule"],
        "grammar_rule": "apply_quotient_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_F",
            "DERIVATIVE",
            "PUSH_G",
            "MULT",
            "PUSH_F",
            "PUSH_G",
            "DERIVATIVE",
            "MULT",
            "SUB",
            "PUSH_G",
            "PUSH_2",
            "POW",
            "DIV",
        ],
    },
    {
        "pattern_id": "chain_rule",
        "domain": "calculus",
        "context_keywords": ["chain rule", "derivative", "differentiate"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"\*|times",
        "semantic_tags": ["derivative", "chain_rule"],
        "grammar_rule": "apply_chain_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_G",
            "DERIVATIVE",
            "PUSH_F",
            "PUSH_G",
            "COMPOSE",
            "DERIVATIVE",
            "MULT",
        ],
    },
    {
        "pattern_id": "sum_rule",
        "domain": "calculus",
        "context_keywords": ["sum rule", "linearity", "derivative"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"\+|plus",
        "semantic_tags": ["derivative", "sum_rule"],
        "grammar_rule": "apply_sum_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_F",
            "DERIVATIVE",
            "PUSH_G",
            "DERIVATIVE",
            "ADD",
        ],
    },
    {
        "pattern_id": "constant_multiple_rule",
        "domain": "calculus",
        "context_keywords": ["constant multiple", "linearity", "derivative"],
        "lhs_regex": r"d\s*/\s*d\w|d/dx|derivative",
        "rhs_regex": r"\*|times",
        "semantic_tags": ["derivative", "constant_multiple"],
        "grammar_rule": "apply_constant_multiple_rule",
        "tier": 3,
        "rpn_program": [
            "PUSH_C",
            "PUSH_F",
            "DERIVATIVE",
            "MULT",
        ],
    },
    {
        "pattern_id": "integration_by_parts",
        "domain": "calculus",
        "context_keywords": ["integration by parts", "integral", "by parts"],
        "lhs_regex": r"\\int|integral",
        "rhs_regex": r"-|minus",
        "semantic_tags": ["integral", "integration_by_parts"],
        "grammar_rule": "apply_integration_by_parts",
        "tier": 3,
        "rpn_program": [
            "PUSH_U",
            "PUSH_V",
            "MULT",
            "PUSH_V",
            "PUSH_U",
            "DERIVATIVE",
            "INTEGRAL",
            "SUB",
        ],
    },
    {
        "pattern_id": "fundamental_theorem_calculus",
        "domain": "calculus",
        "context_keywords": ["fundamental theorem", "integral"],
        "lhs_regex": r"\\int|integral",
        "rhs_regex": r"-|minus",
        "semantic_tags": ["integral", "fundamental_theorem"],
        "grammar_rule": "apply_fundamental_theorem_calculus",
        "tier": 3,
        "rpn_program": [
            "PUSH_F",
            "EVAL_B",
            "PUSH_F",
            "EVAL_A",
            "SUB",
        ],
    },
    {
        "pattern_id": "pythagorean_identity",
        "domain": "geometry",
        "context_keywords": ["pythagorean", "identity", "trig"],
        "lhs_regex": r"sin|cos",
        "rhs_regex": r"=|1",
        "semantic_tags": ["angle", "trigonometry", "identity"],
        "grammar_rule": "apply_pythagorean_identity",
        "tier": 1,
        "rpn_program": [
            "PUSH_SIN_THETA",
            "POW2",
            "PUSH_COS_THETA",
            "POW2",
            "ADD",
            "PUSH_1",
            "EQ",
        ],
    },
]


def _is_transform_candidate(art: Dict[str, Any]) -> bool:
    lhs = art.get("lhs")
    rhs = art.get("rhs")
    conclusion = art.get("conclusion")
    if not lhs or not rhs or not conclusion:
        return False
    conditions = art.get("conditions") or []
    if conditions:
        return True
    artifact_type = str(art.get("artifact_type") or "").lower()
    return artifact_type in {"theorem", "lemma", "definition", "formula", "example"}


def _pattern_matches(art: Dict[str, Any], pattern_def: Dict[str, Any]) -> bool:
    lhs = str(art.get("lhs") or "")
    rhs = str(art.get("rhs") or "")
    context_text = " ".join(
        [
            *[str(c) for c in (art.get("conditions") or [])],
            str(art.get("raw_block") or ""),
            str(art.get("conclusion") or ""),
        ]
    ).lower()
    keywords = [k.lower() for k in pattern_def.get("context_keywords") or []]
    if keywords and not any(k in context_text for k in keywords):
        return False
    lhs_regex = pattern_def.get("lhs_regex")
    if lhs_regex and not re.search(lhs_regex, lhs, flags=re.IGNORECASE):
        return False
    rhs_regex = pattern_def.get("rhs_regex")
    if rhs_regex and not re.search(rhs_regex, rhs, flags=re.IGNORECASE):
        return False
    return True


def _collect_semantic_tags(artifacts: Sequence[Dict[str, Any]], base_tags: Sequence[str]) -> List[str]:
    tags = set(base_tags)
    for art in artifacts:
        domain = str(art.get("domain") or "").strip().lower()
        if domain:
            tags.add(domain)
        for info in (art.get("symbol_bindings") or {}).values():
            role = str(info.get("meaning") or "").strip().lower()
            if role and role not in _GENERIC_ROLES:
                tags.add(role)
    return sorted(t for t in tags if t)


def _summarize_precondition(artifacts: Sequence[Dict[str, Any]], pattern_def: Dict[str, Any]) -> Dict[str, Any]:
    type_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    lhs_samples: Counter[str] = Counter()
    for art in artifacts:
        artifact_type = str(art.get("artifact_type") or "unknown").lower()
        type_counts[artifact_type] += 1
        context_text = " ".join([*art.get("conditions", []), art.get("raw_block") or ""])
        context_counts.update(_tokenize_theorem_context(context_text))
        lhs = str(art.get("lhs") or "").strip()
        if lhs:
            lhs_samples[lhs] += 1
    return {
        "artifact_types": [t for t, _ in type_counts.most_common()],
        "context_cues": [t for t, _ in context_counts.most_common(8)],
        "lhs_pattern": pattern_def.get("lhs_regex") or "",
        "rhs_pattern": pattern_def.get("rhs_regex") or "",
    }


def _summarize_postcondition(pattern_def: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result_type": pattern_def.get("result_type", "expression"),
        "validation": "equivalence",
    }


def _build_theorem_pattern(
    pattern_def: Dict[str, Any], artifacts: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    rpn_program = pattern_def.get("rpn_program")
    if not rpn_program:
        return None
    lhs = str(artifacts[0].get("lhs") or "")
    rhs = str(artifacts[0].get("rhs") or "")
    semantic_tags = _collect_semantic_tags(artifacts, pattern_def.get("semantic_tags") or [])
    book_counts: Counter[str] = Counter()
    for art in artifacts:
        book_id = str(art.get("book_id") or "")
        if book_id:
            book_counts[book_id] += 1
    return {
        "pattern_id": pattern_def["pattern_id"],
        "domain": pattern_def.get("domain") or str(artifacts[0].get("domain") or "unknown"),
        "semantic_tags": semantic_tags,
        "grammar_rule": pattern_def.get("grammar_rule"),
        "precondition": _summarize_precondition(artifacts, pattern_def),
        "transformation": {
            "lhs": lhs,
            "rhs": rhs,
            "rpn_program": list(rpn_program),
            "tier": int(pattern_def.get("tier", 1)),
        },
        "postcondition": _summarize_postcondition(pattern_def),
        "source": {
            "book_ids": [b for b, _ in book_counts.most_common(5)],
            "artifact_ids": [a.get("artifact_id") for a in artifacts[:10]],
            "example_count": len(artifacts),
        },
    }


def extract_theorem_patterns(
    artifact_dirs: Sequence[str],
    *,
    min_examples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extract single-step theorem patterns from artifacts.

    This targets atomic transformations (lhs -> rhs) with optional conditions.
    Multi-step derivations are intentionally excluded and remain the Grammar
    Galaxy's responsibility.
    """
    paths = [Path(p) for p in artifact_dirs]
    pattern_defs = {p["pattern_id"]: p for p in _THEOREM_PATTERN_DEFS}
    pattern_hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for art in _iter_artifacts(paths):
        if not _is_transform_candidate(art):
            continue
        for pattern_def in _THEOREM_PATTERN_DEFS:
            if _pattern_matches(art, pattern_def):
                pattern_hits[pattern_def["pattern_id"]].append(art)
                break

    patterns: List[Dict[str, Any]] = []
    for pattern_id, artifacts in pattern_hits.items():
        if len(artifacts) < min_examples:
            continue
        pattern = _build_theorem_pattern(pattern_defs[pattern_id], artifacts)
        if pattern:
            patterns.append(pattern)

    patterns.sort(key=lambda item: (-item["source"]["example_count"], item["pattern_id"]))
    return patterns


THEOREM_PATTERNS: List[Dict[str, Any]] = []


def populate_theorem_patterns(
    artifact_dirs: Sequence[str],
    math_galaxy: Optional[object] = None,
    *,
    min_examples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Populate Math Galaxy with single-step theorem patterns.

    If the math_galaxy exposes `add_theorem_pattern` or `theorem_patterns`,
    those are updated; otherwise the patterns are returned.
    """
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=min_examples)
    global THEOREM_PATTERNS
    THEOREM_PATTERNS = list(patterns)

    if math_galaxy is None:
        return list(patterns)

    if hasattr(math_galaxy, "add_theorem_pattern"):
        for pat in patterns:
            math_galaxy.add_theorem_pattern(pat)
        return list(patterns)

    if hasattr(math_galaxy, "theorem_patterns"):
        try:
            math_galaxy.theorem_patterns = list(patterns)
        except Exception:
            pass
    return list(patterns)


def extract_role_patterns(
    artifact_dirs: Sequence[str],
    *,
    min_examples: int = 3,
    top_k_context: int = 8,
    top_k_equation: int = 6,
    top_k_vars: int = 6,
) -> List[Dict[str, Any]]:
    """
    Extract role patterns from validated artifacts.

    This is intentionally lightweight: it collects frequency stats for
    context cues, equation tokens, and variable conventions. RPN programs
    will be synthesized in later tasks.
    """
    paths = [Path(p) for p in artifact_dirs]
    role_counts: Counter[str] = Counter()
    role_domains: Counter[Tuple[str, str]] = Counter()
    role_context: Dict[str, Counter[str]] = defaultdict(Counter)
    role_equations: Dict[str, Counter[str]] = defaultdict(Counter)
    role_vars: Dict[str, Counter[str]] = defaultdict(Counter)

    for art in _iter_artifacts(paths):
        var_mapping = art.get("var_mapping") or {}
        context_text = " ".join([*art.get("conditions", []), art.get("raw_block") or "", art.get("conclusion") or ""])
        context_tokens = _tokenize_context(context_text)
        equation_tokens = _equation_patterns(art.get("lhs"), art.get("rhs"), art.get("conclusion"))
        bindings = art.get("symbol_bindings") or {}
        for placeholder, info in bindings.items():
            role = str(info.get("meaning") or "unknown").strip()
            if role in _GENERIC_ROLES:
                continue
            role_counts[role] += 1
            domain = str(info.get("domain_hint") or art.get("domain") or "").strip().lower()
            if domain:
                role_domains[(role, domain)] += 1
            role_context[role].update(context_tokens)
            role_equations[role].update(equation_tokens)
            role_vars[role].update(_variable_conventions(var_mapping, str(placeholder)))

    patterns: List[Dict[str, Any]] = []
    for role, count in role_counts.items():
        if count < min_examples:
            continue
        domain = None
        if role_domains:
            domain_counts: Counter[str] = Counter()
            for (r, d), c in role_domains.items():
                if r == role and d:
                    domain_counts[d] += c
            if domain_counts:
                domain = domain_counts.most_common(1)[0][0]
        patterns.append(
            {
                "role": role,
                "domain": domain or "unknown",
                "tier": _role_tier(role),
                "example_count": count,
                "context_cues": [t for t, _ in role_context[role].most_common(top_k_context)],
                "equation_patterns": [t for t, _ in role_equations[role].most_common(top_k_equation)],
                "variable_conventions": [t for t, _ in role_vars[role].most_common(top_k_vars)],
            }
        )

    patterns.sort(key=lambda item: (-item["example_count"], item["role"]))
    return patterns


ROLE_PATTERNS: List[Dict[str, Any]] = []


def populate_role_patterns(
    artifact_dirs: Sequence[str],
    math_galaxy: Optional[object] = None,
    *,
    min_examples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Populate Math Galaxy with role patterns derived from validated artifacts.

    This returns the patterns and optionally attaches them to a Math Galaxy
    object if it exposes `add_role_pattern` or `role_patterns`.
    """
    patterns = extract_role_patterns(artifact_dirs, min_examples=min_examples)
    global ROLE_PATTERNS
    ROLE_PATTERNS = list(patterns)

    if math_galaxy is None:
        return list(patterns)

    if hasattr(math_galaxy, "add_role_pattern"):
        for pat in patterns:
            math_galaxy.add_role_pattern(pat)
        return list(patterns)

    if hasattr(math_galaxy, "role_patterns"):
        try:
            math_galaxy.role_patterns = list(patterns)
        except Exception:
            pass
    return list(patterns)


__all__ = [
    "DOMAIN_SIGNATURES",
    "ROLE_PATTERNS",
    "THEOREM_PATTERNS",
    "populate_domain_signatures",
    "extract_role_patterns",
    "populate_role_patterns",
    "extract_theorem_patterns",
    "populate_theorem_patterns",
]
