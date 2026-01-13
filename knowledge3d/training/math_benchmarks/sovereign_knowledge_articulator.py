"""
Sovereign Knowledge Articulator (Phase 1).

This module upgrades Book Galaxy ingestion beyond "lhs = rhs" template scraping
by producing *articulated* knowledge artifacts:
  - conditions (when the artifact applies),
  - conclusions (the equation / rule),
  - symbol bindings (variable semantics when recoverable).

Important:
- This is ingestion-side tooling (NOT the inference hot path).
- It must stay dependency-light (stdlib + existing RPNParser only).
- It is designed for `pdftotext` output (plain text), not full LaTeX ASTs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knowledge3d.training.math_benchmarks.rpn_parser import RPNParser


@dataclass(frozen=True)
class SymbolBinding:
    symbol: str
    meaning: str = "unknown"
    domain: str = "unknown"
    domain_hint: Optional[str] = None
    constraints: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeArtifact:
    artifact_id: str
    artifact_type: str  # theorem | definition | lemma | corollary | proposition | formula | example
    name: str
    domain: Optional[str]
    book_id: str
    page_number: int

    # Applicability
    conditions: List[str] = field(default_factory=list)
    conditions_rpn: List[str] = field(default_factory=list)

    # Equation (when present)
    lhs: Optional[str] = None
    rhs: Optional[str] = None
    lhs_rpn: Optional[str] = None  # normalized RPN for LHS (placeholders are single-letter vars)
    rhs_rpn: Optional[str] = None  # normalized RPN for RHS (placeholders are single-letter vars)
    rpn: Optional[str] = None  # chosen executable RPN candidate (best-effort)
    conclusion: Optional[str] = None
    conclusion_rpn: Optional[str] = None
    derived_rpns: List[Dict[str, Any]] = field(default_factory=list)  # additional executable candidates
    var_mapping: Dict[str, str] = field(default_factory=dict)  # original identifier -> placeholder

    # Semantics
    symbol_bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # placeholder -> SymbolBinding(asdict)

    # Raw extraction provenance
    source: str = "plain_text"
    raw_block: Optional[str] = None


@dataclass
class RoleExtractionConfig:
    """
    Ingestion-time semantic role extraction settings.

    This is *not* used in the inference hot path.
    """

    enabled: bool = False
    model: str = "granite4:tiny-h"
    fallback_model: Optional[str] = "qwen2.5:14b"
    timeout_s: float = 30.0
    max_context_chars: int = 1200
    cache_path: Optional[str] = None  # JSONL cache
    restart_between_calls: bool = False
    # Prefer HTTP API when available; fall back to `ollama run`.
    prefer_http: bool = True
    http_url: str = "http://127.0.0.1:11434/api/generate"
    fallback_on_unknown: bool = True


# Tier 1A: Geometry & Spatial Measurement
GEOMETRY_ROLES: List[str] = [
    "radius",
    "diameter",
    "chord",
    "arc_length",
    "height",
    "width",
    "length",
    "depth",
    "thickness",
    "distance",
    "perimeter",
    "circumference",
    "leg",
    "hypotenuse",
    "base",
    "altitude",
    "median",
    "side",
    "angle",
    "central_angle",
    "inscribed_angle",
    "radian",
    "area",
    "surface_area",
    "volume",
    "cross_section",
    "slope",
    "intercept",
    "coordinate",
]

# Tier 1B: Linear Algebra & Vector Spaces
LINEAR_ALGEBRA_ROLES: List[str] = [
    "vector",
    "component",
    "magnitude",
    "direction",
    "dot_product",
    "cross_product",
    "projection",
    "matrix",
    "element",
    "row",
    "column",
    "determinant",
    "trace",
    "rank",
    "eigenvalue",
    "eigenvector",
    "characteristic_polynomial",
    "dimension",
    "basis",
    "span",
    "kernel",
    "image",
    "linear_combination",
    "transformation",
]

# Tier 1C: Calculus & Analysis
CALCULUS_ROLES: List[str] = [
    "derivative",
    "differential",
    "rate_of_change",
    "gradient",
    "partial_derivative",
    "directional_derivative",
    "slope",
    "tangent",
    "secant",
    "integral",
    "antiderivative",
    "accumulation",
    "area_under_curve",
    "definite_integral",
    "indefinite_integral",
    "limit",
    "epsilon",
    "delta",
    "bound",
    "supremum",
    "infimum",
    "sequence",
    "series",
    "term",
    "sum",
    "convergence",
]

# Tier 1D: Physics & Applied Math
PHYSICS_ROLES: List[str] = [
    "position",
    "velocity",
    "acceleration",
    "force",
    "mass",
    "momentum",
    "energy",
    "work",
    "power",
    "torque",
    "angular_velocity",
    "angular_acceleration",
    "frequency",
    "wavelength",
    "amplitude",
    "period",
    "phase",
    "temperature",
    "pressure",
    "volume",
    "entropy",
    "heat_capacity",
    "internal_energy",
    "charge",
    "current",
    "voltage",
    "resistance",
    "electric_field",
    "magnetic_field",
    "flux",
]

# Tier 1E: Number Theory & Algebra
NUMBER_THEORY_ROLES: List[str] = [
    "prime",
    "composite",
    "factor",
    "divisor",
    "multiple",
    "greatest_common_divisor",
    "least_common_multiple",
    "modulus",
    "remainder",
    "quotient",
    "congruence",
    "group_element",
    "ring_element",
    "field_element",
    "order",
    "generator",
    "identity",
]

# Tier 1F: Probability & Statistics
STATISTICS_ROLES: List[str] = [
    "mean",
    "median",
    "mode",
    "variance",
    "standard_deviation",
    "percentile",
    "quartile",
    "range",
    "probability",
    "event",
    "sample_space",
    "outcome",
    "expected_value",
    "distribution",
    "density",
    "parameter",
    "statistic",
    "estimate",
    "confidence_interval",
    "p_value",
    "significance_level",
]

# Tier 2: Formula components
FORMULA_ROLES: List[str] = [
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
]

# Tier 3: Generic fallbacks
GENERIC_ROLES: List[str] = [
    "constant",
    "variable",
    "placeholder",
    "unknown",
]

DOMAIN_ROLE_MAP: Dict[str, List[str]] = {
    "geometry": GEOMETRY_ROLES,
    "linear_algebra": LINEAR_ALGEBRA_ROLES,
    "calculus": CALCULUS_ROLES,
    "physics": PHYSICS_ROLES,
    "number_theory": NUMBER_THEORY_ROLES,
    "statistics": STATISTICS_ROLES,
}

DOMAIN_ORDER: List[str] = [
    "geometry",
    "linear_algebra",
    "calculus",
    "physics",
    "number_theory",
    "statistics",
]

ROLE_ALIASES: Dict[str, str] = {
    "norm": "magnitude",
    "component_1": "component",
    "component_2": "component",
    "component_3": "component",
    "component_i": "component",
    "component_j": "component",
    "component_k": "component",
    "avg": "mean",
    "average": "mean",
    "stddev": "standard_deviation",
    "stdev": "standard_deviation",
    "prob": "probability",
    "mod": "modulus",
    "speed": "velocity",
}

_DOMAIN_BY_ROLE: Dict[str, str] = {}
for _domain, _roles in DOMAIN_ROLE_MAP.items():
    for _role in _roles:
        if _role not in _DOMAIN_BY_ROLE:
            _DOMAIN_BY_ROLE[_role] = _domain
for _role in FORMULA_ROLES:
    _DOMAIN_BY_ROLE[_role] = "formula"
for _role in GENERIC_ROLES:
    _DOMAIN_BY_ROLE[_role] = "generic"

_DOMAIN_CONTEXT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "geometry": (
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
    ),
    "linear_algebra": (
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
    ),
    "calculus": (
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
    ),
    "physics": (
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
    ),
    "number_theory": (
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
    ),
    "statistics": (
        "probability",
        "random",
        "distribution",
        "mean",
        "variance",
        "standard deviation",
        "expected value",
        "sample",
        "population",
    ),
}

_DOMAIN_EQUATION_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "geometry": (
        re.compile(r"\bpi\b"),
        re.compile("π"),
    ),
    "linear_algebra": (
        re.compile(r"\bdet\b"),
        re.compile(r"\btrace\b"),
        re.compile(r"\brank\b"),
        re.compile(r"\|\|"),
        re.compile("∥"),
    ),
    "calculus": (
        re.compile(r"\bd/d\b"),
        re.compile("∂"),
        re.compile("∫"),
        re.compile(r"\blim\b"),
    ),
    "physics": (
        re.compile(r"\bF\s*=\s*m\s*a\b"),
    ),
    "number_theory": (
        re.compile(r"\bmod\b"),
        re.compile("≡"),
        re.compile(r"\bgcd\b"),
        re.compile(r"\blcm\b"),
    ),
    "statistics": (
        re.compile(r"\bp\s*\("),
        re.compile(r"\be\s*\["),
        re.compile(r"\be\s*\("),
        re.compile(r"\bvar\s*\("),
        re.compile(r"\bvar\s*\["),
        re.compile("σ"),
        re.compile("μ"),
    ),
}


class OllamaRoleExtractor:
    """
    Ingestion-time helper to assign `SymbolBinding.meaning` via a local Ollama model.

    - Sequential and blocking: safe for single-GPU environments.
    - Cached: JSONL append-only cache for pause/resume.
    - Optional: can be disabled entirely (default).
    """

    def __init__(self, *, config: RoleExtractionConfig) -> None:
        self._cfg = config
        self._cache: Dict[str, Tuple[str, str, Optional[str]]] = {}
        if self._cfg.cache_path:
            self._load_cache(self._cfg.cache_path)

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def _load_cache(self, path: str) -> None:
        try:
            p = str(path)
            if not p or not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = str(obj.get("key") or "")
                    role = str(obj.get("role") or "")
                    domain = str(obj.get("domain") or "")
                    domain_hint = str(obj.get("domain_hint") or "") or None
                    if key and role:
                        self._cache[key] = (role, domain or "real", domain_hint)
        except Exception:
            # Cache is best-effort; never fail ingestion because of it.
            return

    def _append_cache(self, path: str, *, key: str, role: str, domain: str, domain_hint: Optional[str]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"key": key, "role": role, "domain": domain, "domain_hint": domain_hint or ""},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            return

    def _cache_key(
        self,
        *,
        var: str,
        context: str,
        equation: str,
        model: str,
        book_domain_hint: Optional[str] = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(str(model).encode("utf-8", errors="replace"))
        h.update(b"\n")
        h.update(str(var).encode("utf-8", errors="replace"))
        h.update(b"\n")
        h.update(str(context).encode("utf-8", errors="replace"))
        h.update(b"\n")
        h.update(str(equation).encode("utf-8", errors="replace"))
        if book_domain_hint:
            h.update(b"\n")
            h.update(str(book_domain_hint).encode("utf-8", errors="replace"))
        return h.hexdigest()

    def _restart_ollama(self) -> None:
        if not self._cfg.restart_between_calls:
            return
        try:
            subprocess.run(["pkill", "-f", "ollama"], check=False, capture_output=True)
            time.sleep(1.0)
        except Exception:
            return

    def _count_context_hits(self, text: str, keywords: Sequence[str]) -> int:
        count = 0
        for kw in keywords:
            token = str(kw or "").strip().lower()
            if not token:
                continue
            if " " in token:
                if token in text:
                    count += 1
            else:
                if re.search(rf"\b{re.escape(token)}\b", text):
                    count += 1
        return count

    def _normalize_domain_hint(self, hint: Optional[str]) -> Optional[str]:
        if not hint:
            return None
        token = str(hint).strip().lower()
        token = token.replace("-", "_").replace(" ", "_")
        alias_map = {
            "linalg": "linear_algebra",
            "lin_alg": "linear_algebra",
            "linear": "linear_algebra",
            "stats": "statistics",
            "probability": "statistics",
            "stat": "statistics",
            "calc": "calculus",
            "geom": "geometry",
            "num_theory": "number_theory",
            "numbertheory": "number_theory",
        }
        token = alias_map.get(token, token)
        return token if token in DOMAIN_ROLE_MAP else None

    def _detect_domains(
        self,
        context: str,
        equation: str,
        *,
        book_domain_hint: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int]]:
        ctx = str(context or "").lower()
        eq = str(equation or "").lower()
        ctx_scores: Dict[str, int] = {}
        eq_scores: Dict[str, int] = {}
        book_scores: Dict[str, int] = {}
        total_scores: Dict[str, int] = {}

        for domain in DOMAIN_ORDER:
            ctx_hits = self._count_context_hits(ctx, _DOMAIN_CONTEXT_KEYWORDS.get(domain, ()))
            eq_hits = 0
            for pat in _DOMAIN_EQUATION_PATTERNS.get(domain, ()):
                if pat.search(eq):
                    eq_hits += 1
            ctx_scores[domain] = ctx_hits
            eq_scores[domain] = eq_hits

        normalized_hint = self._normalize_domain_hint(book_domain_hint)
        if normalized_hint:
            book_scores[normalized_hint] = 1

        for domain in DOMAIN_ORDER:
            hint_weight = 10 if book_scores.get(domain) else 0
            total_scores[domain] = ctx_scores.get(domain, 0) + (2 * eq_scores.get(domain, 0)) + hint_weight

        sorted_domains = [
            domain
            for domain in DOMAIN_ORDER
            if total_scores.get(domain, 0) > 0
        ]
        sorted_domains.sort(
            key=lambda d: (-total_scores.get(d, 0), -eq_scores.get(d, 0), DOMAIN_ORDER.index(d))
        )
        return sorted_domains, total_scores, eq_scores, book_scores

    def _get_role_choices_multidomain(self, detected_domains: Sequence[str]) -> List[str]:
        role_choices: List[str] = []
        for domain in detected_domains:
            role_choices.extend(DOMAIN_ROLE_MAP.get(domain, []))

        all_tier1: List[str] = []
        for domain in DOMAIN_ORDER:
            all_tier1.extend(DOMAIN_ROLE_MAP.get(domain, []))
        for role in all_tier1:
            if role not in role_choices:
                role_choices.append(role)

        role_choices.extend(FORMULA_ROLES)
        role_choices.extend(GENERIC_ROLES)

        seen: set[str] = set()
        deduped: List[str] = []
        for role in role_choices:
            if role in seen:
                continue
            seen.add(role)
            deduped.append(role)
        return deduped

    def infer_role(
        self,
        *,
        var: str,
        context: str,
        equation: str,
        book_domain_hint: Optional[str] = None,
        role_choices: Optional[Sequence[str]] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Returns (meaning, domain, domain_hint).

        Meaning is one of the role choices or "unknown". Domain is the numeric
        domain (best-effort), and domain_hint is the detected semantic domain.
        """
        v = str(var or "").strip()
        if not v:
            return ("unknown", "real", None)

        ctx = str(context or "")
        if len(ctx) > int(self._cfg.max_context_chars):
            ctx = ctx[: int(self._cfg.max_context_chars)]

        eq = str(equation or "").strip()
        normalized_hint = self._normalize_domain_hint(book_domain_hint)
        detected_domains, domain_scores, eq_scores, book_scores = self._detect_domains(
            ctx, eq, book_domain_hint=normalized_hint
        )
        domain_hint = detected_domains[0] if detected_domains else None
        if role_choices is None:
            role_choices = self._get_role_choices_multidomain(detected_domains)

        key = self._cache_key(
            var=v,
            context=ctx,
            equation=eq,
            model=self._cfg.model,
            book_domain_hint=normalized_hint,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._restart_ollama()

        prompt = self._build_prompt(
            var=v,
            context=ctx,
            equation=eq,
            role_choices=role_choices,
            detected_domains=detected_domains,
            domain_scores=domain_scores,
            eq_scores=eq_scores,
            book_domain_hint=normalized_hint,
            book_scores=book_scores,
        )
        role = self._run_model(prompt, model=self._cfg.model, role_choices=role_choices)

        if (
            role == "unknown"
            and self._cfg.fallback_model
            and self._cfg.fallback_on_unknown
        ):
            self._restart_ollama()
            role = self._run_model(prompt, model=self._cfg.fallback_model, role_choices=role_choices)
            if role != "unknown":
                key = self._cache_key(
                    var=v,
                    context=ctx,
                    equation=eq,
                    model=self._cfg.fallback_model,
                    book_domain_hint=normalized_hint,
                )

        role = self._sanitize_role(role, role_choices=role_choices)
        domain = self._domain_for_role(role)

        self._cache[key] = (role, domain, domain_hint)
        if self._cfg.cache_path:
            self._append_cache(self._cfg.cache_path, key=key, role=role, domain=domain, domain_hint=domain_hint)
        return (role, domain, domain_hint)

    def _build_prompt(
        self,
        *,
        var: str,
        context: str,
        equation: str,
        role_choices: Sequence[str],
        detected_domains: Sequence[str],
        domain_scores: Dict[str, int],
        eq_scores: Dict[str, int],
        book_domain_hint: Optional[str],
        book_scores: Dict[str, int],
    ) -> str:
        ordered: List[str] = []
        seen: set[str] = set()
        for item in role_choices:
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        if detected_domains:
            domain_lines = []
            for domain in detected_domains[:3]:
                total = domain_scores.get(domain, 0)
                eq = eq_scores.get(domain, 0)
                book = book_scores.get(domain, 0)
                book_label = "hint" if book else "nohint"
                domain_lines.append(f"{domain} (score={total}, eq={eq}, book={book_label})")
            domain_hint = "\n  - ".join(domain_lines)
        else:
            domain_hint = "none (use general math roles)"

        avoid_examples: List[str] = []
        if detected_domains:
            avoid_map = {
                "geometry": ["radius", "area"],
                "linear_algebra": ["determinant", "eigenvalue"],
                "calculus": ["derivative", "integral"],
                "physics": ["force", "energy"],
                "number_theory": ["prime", "modulus"],
                "statistics": ["probability", "variance"],
            }
            for domain in DOMAIN_ORDER:
                if domain in detected_domains:
                    continue
                avoid_examples.extend(avoid_map.get(domain, []))
        avoid_line = ", ".join(avoid_examples[:8]) if avoid_examples else "n/a"

        top_roles = ordered[:15]
        remaining = max(0, len(ordered) - len(top_roles))
        return (
            "You extract semantic roles of mathematical variables from textbook context.\n"
            "Return ONLY one role token from the allowed list.\n\n"
            f"BOOK DOMAIN HINT: {book_domain_hint or 'none'}\n\n"
            "DETECTED DOMAINS (ranked):\n"
            f"  - {domain_hint}\n\n"
            f"ALLOWED ROLES (top priority {len(top_roles)}/{len(ordered)}):\n"
            f"  {', '.join(top_roles)}\n"
            f"... plus {remaining} more roles.\n\n"
            "PRIORITY:\n"
            "1) Use a domain-specific role that matches the context.\n"
            "2) If no domain-specific role fits, use a formula component.\n"
            "3) Use 'constant' only for known constants (pi, e, g, c).\n"
            "4) Use 'variable' or 'unknown' only as a last resort.\n"
            f"5) Avoid unrelated domains unless explicitly mentioned (e.g., {avoid_line}).\n\n"
            "Think step-by-step internally but output ONLY the final role token.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Context: right triangle with legs a and b; a^2 + b^2 = c^2\n"
            "Equation: a^2 + b^2 = c^2\n"
            "Variable: c\n"
            "Role: hypotenuse\n\n"
            "Context: velocity magnitude v = sqrt(v_x^2 + v_y^2)\n"
            "Equation: v = sqrt(v_x^2 + v_y^2)\n"
            "Variable: v_x\n"
            "Role: component\n\n"
            "Context: Euclidean norm of vector x is ||x|| = sqrt(x1^2 + x2^2)\n"
            "Equation: ||x|| = sqrt(x1^2 + x2^2)\n"
            "Variable: x1\n"
            "Role: component\n\n"
            "Context: slope is m = dy/dx in differential calculus\n"
            "Equation: m = dy/dx\n"
            "Variable: m\n"
            "Role: derivative\n\n"
            "Context: modular arithmetic a ≡ b (mod n)\n"
            "Equation: a ≡ b (mod n)\n"
            "Variable: n\n"
            "Role: modulus\n\n"
            "Context: probability of event A is p = P(A)\n"
            "Equation: p = P(A)\n"
            "Variable: p\n"
            "Role: probability\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"EQUATION/FORMULA:\n{equation}\n\n"
            f"VARIABLE: {var}\n\n"
            "ROLE:"
        )

    def _ollama_http(self, prompt: str, *, model: str) -> str:
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(self._cfg.http_url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise RuntimeError(str(e))
        obj = json.loads(raw)
        return str(obj.get("response") or "").strip()

    def _ollama_cli(self, prompt: str, *, model: str) -> str:
        # Avoid huge argv: pass prompt via stdin.
        proc = subprocess.run(
            ["ollama", "run", model],
            input=str(prompt),
            text=True,
            capture_output=True,
            timeout=float(self._cfg.timeout_s),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip()[:200])
        return str(proc.stdout or "").strip()

    def _run_model(self, prompt: str, *, model: str, role_choices: Sequence[str]) -> str:
        try:
            if self._cfg.prefer_http:
                role = self._ollama_http(prompt, model=model)
            else:
                role = self._ollama_cli(prompt, model=model)
        except Exception:
            return "unknown"
        return self._sanitize_role(role, role_choices=role_choices)

    def _sanitize_role(self, role: str, *, role_choices: Sequence[str]) -> str:
        low = str(role or "").strip().lower()
        low = re.sub(r"[^a-z0-9_]+", "", low)
        low = ROLE_ALIASES.get(low, low)
        allowed = {str(x).strip().lower() for x in role_choices}
        if low in allowed:
            return low
        # Sometimes models answer with extra text; pick first token-like substring.
        for t in re.findall(r"[a-z0-9_]{3,}", str(role or "").lower()):
            t = ROLE_ALIASES.get(t, t)
            if t in allowed:
                return t
        return "unknown"

    def _domain_for_role(self, role: str) -> str:
        if role in {
            "radius",
            "diameter",
            "chord",
            "arc_length",
            "height",
            "width",
            "length",
            "depth",
            "thickness",
            "distance",
            "perimeter",
            "circumference",
            "leg",
            "hypotenuse",
            "base",
            "altitude",
            "median",
            "side",
        }:
            return "positive_real"
        if role in {"area", "surface_area", "volume", "cross_section", "variance", "standard_deviation"}:
            return "nonnegative_real"
        if role in {"angle", "central_angle", "inscribed_angle", "radian"}:
            return "angle"
        if role in {"probability", "p_value", "significance_level"}:
            return "probability"
        return "real"


_BLOCK_HEAD_RE = re.compile(
    r"^\s*(?P<kind>Theorem|Definition|Lemma|Corollary|Proposition|Example|Exercise)\b(?P<rest>.*)$",
    re.IGNORECASE,
)

# Match a single "=" but avoid comparison operators like "!=" "<=" ">=" and "==".
_EQ_LINE_RE = re.compile(r"(?P<lhs>[^=\n]{1,120})\s*(?<![!<>])=(?![=])\s*(?P<rhs>[^\n]{1,240})")

_LATEX_ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<kind>theorem|definition|lemma|corollary|proposition|example|exercise)\}"
    r"(?P<opt>\[[^\]]+\])?"
    r"(?P<body>.*?)"
    r"\\end\{\1\}",
    re.IGNORECASE | re.DOTALL,
)

# Common unicode normalization for PDF text exports.
_UNICODE_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("\x03", "="),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
    ("×", "*"),
    ("∙", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("∕", "/"),
    ("∗", "*"),
    ("∶", ":"),
    ("＝", "="),
    ("≈", "="),
    ("≃", "="),
    ("≅", "="),
    ("≤", "<="),
    ("≥", ">="),
)

_SUPERSCRIPT_DIGITS: Dict[str, str] = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
}


def _normalize_text(text: str) -> str:
    out = str(text or "")
    for a, b in _UNICODE_REPLACEMENTS:
        out = out.replace(a, b)
    # Convert common superscripts into caret notation when attached to a symbol/number:
    # "a²" -> "a^2"
    for sup, digit in _SUPERSCRIPT_DIGITS.items():
        out = re.sub(rf"(\w){re.escape(sup)}", rf"\\1^{digit}", out)
    # Normalize whitespace
    out = re.sub(r"[ \t]+", " ", out)
    return out


def _strip_latex_noise(text: str) -> str:
    """
    Best-effort LaTeX → plain-text normalization (ingestion-side only).

    This intentionally does NOT attempt a full LaTeX AST parse; it exists to
    improve condition/symbol extraction when sources contain lightweight LaTeX.
    """
    s = str(text or "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "")
    s = re.sub(r"\\(begin|end)\{[^}]+\}", " ", s)
    # Common math commands that are semantically meaningful for conditions.
    s = s.replace("\\triangle", "triangle")
    s = s.replace("\\angle", "angle")
    # Drop remaining commands, keep their identifiers (e.g., \mathbb{R} -> R).
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _slug_ident(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower())
    value = value.strip("_")
    return value or "artifact"


def _normalize_and_validate_rpn(program: str) -> Tuple[str, Dict[str, str]]:
    """
    Normalize an RPN program so identifiers become single-letter placeholders.
    Returns (normalized_program, mapping_original_to_placeholder).
    """
    tokens = [t for t in str(program or "").strip().split() if t]
    if not tokens:
        return ("", {})

    # Keep aligned with `book_galaxy_ingestion._normalize_template_rpn`, but we also
    # return the mapping for symbol bindings.
    known_ops = {
        "+",
        "-",
        "*",
        "/",
        "pow",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "log",
        "ln",
        "exp",
        "floor",
        "ceil",
        "gcd",
        "factorial",
        "binomial",
        "^",  # may appear from tokenization; keep but convert to pow below if needed
    }

    mapping: Dict[str, str] = {}
    pool = list("xyzabcdefghijklmnopqrstuvw")
    out: List[str] = []
    known_consts = {"pi", "π", "tau", "phi", "φ", "e"}

    for tok in tokens:
        lower = tok.lower()
        try:
            float(tok)
            out.append(tok)
            continue
        except Exception:
            pass
        if lower in known_consts:
            # Preserve well-known constants as-is so downstream heuristics and
            # runtime execution (ModularRPNEngine.CONSTANTS) can recognize them.
            out.append("π" if tok == "π" else lower)
            continue
        if lower in known_ops:
            out.append("pow" if lower == "^" else lower)
            continue
        if len(tok) == 1 and tok.isalpha():
            out.append(tok.lower())
            continue
        key = tok.strip()
        if key not in mapping:
            if not pool:
                return ("", {})
            mapping[key] = pool.pop(0)
        out.append(mapping[key])

    normalized = " ".join(out)

    # Strict validation: only numbers, known ops, or single-letter vars.
    for tok in normalized.split():
        tl = tok.lower()
        try:
            float(tok)
            continue
        except Exception:
            pass
        if tl in known_consts:
            continue
        if tl in known_ops and tl != "^":
            continue
        if len(tok) == 1 and tok.isalpha():
            continue
        return ("", {})

    return (normalized, mapping)


class SovereignKnowledgeArticulator:
    """
    Extract articulated artifacts from plain-text pages.

    This is a conservative, heuristic parser intended as a first iteration:
    it prioritizes avoiding incorrect structured output over completeness.
    """

    def __init__(
        self,
        *,
        parser: Optional[RPNParser] = None,
        role_extractor: Optional[OllamaRoleExtractor] = None,
    ) -> None:
        self._parser = parser or RPNParser()
        self._role_extractor = role_extractor

    def articulate_pages(
        self,
        *,
        pages: Sequence[Tuple[int, str]],
        book_id: str,
        domain: Optional[str] = None,
        max_artifacts_per_page: int = 32,
    ) -> List[KnowledgeArtifact]:
        artifacts: List[KnowledgeArtifact] = []
        for page_number, text in pages:
            artifacts.extend(
                self.articulate_page(
                    page_number=int(page_number),
                    text=text,
                    book_id=book_id,
                    domain=domain,
                    max_artifacts=max_artifacts_per_page,
                )
            )
        return artifacts

    def articulate_page(
        self,
        *,
        page_number: int,
        text: str,
        book_id: str,
        domain: Optional[str],
        max_artifacts: int = 32,
    ) -> List[KnowledgeArtifact]:
        text = _normalize_text(text)
        lines = [ln.rstrip() for ln in text.splitlines()]

        blocks: List[Dict[str, Any]] = []
        if "\\begin{" in text:
            blocks.extend(self._identify_latex_env_blocks(text))
        blocks.extend(self._identify_blocks(lines))
        artifacts: List[KnowledgeArtifact] = []
        used_pairs: set[tuple[str, str]] = set()

        for block in blocks:
            kind = block["kind"]
            block_lines = block["lines"]
            name = block.get("name") or f"{kind.title()} p{page_number}"
            raw_block = "\n".join(block_lines).strip()

            eq = self._extract_equation(block_lines)
            artifact_id = f"{book_id}_p{int(page_number)}_{_slug_ident(kind)}_{_slug_ident(name)}_{len(artifacts)}"
            if eq is None:
                if kind != "definition":
                    continue
                conditions = self._extract_conditions(block_lines, lhs="", rhs="")
                conditions_rpn = [r for r in (self._condition_to_rpn(c) for c in conditions) if r]
                artifacts.append(
                    KnowledgeArtifact(
                        artifact_id=artifact_id,
                        artifact_type=kind,
                        name=name,
                        domain=domain,
                        book_id=book_id,
                        page_number=int(page_number),
                        conditions=conditions,
                        conditions_rpn=conditions_rpn,
                        lhs=None,
                        rhs=None,
                        lhs_rpn=None,
                        rhs_rpn=None,
                        rpn=None,
                        conclusion=_strip_latex_noise(raw_block)[:500] if raw_block else None,
                        conclusion_rpn=None,
                        derived_rpns=[],
                        var_mapping={},
                        symbol_bindings={},
                        source=str(block.get("source") or "plain_text_block"),
                        raw_block=raw_block[:8000] if raw_block else None,
                    )
                )
            else:
                lhs, rhs = eq
                used_pairs.add((lhs, rhs))

                lhs_rpn_raw = self._rpn_from_infix(lhs)
                rhs_rpn_raw = self._rpn_from_infix(rhs)
                if not lhs_rpn_raw or not rhs_rpn_raw:
                    continue

                lhs_rpn, lhs_map = _normalize_and_validate_rpn(lhs_rpn_raw)
                rhs_rpn, rhs_map = _normalize_and_validate_rpn(rhs_rpn_raw)
                if not lhs_rpn or not rhs_rpn:
                    continue

                # Merge mappings (best-effort). When collisions happen, keep rhs mapping.
                var_map: Dict[str, str] = dict(lhs_map)
                var_map.update(rhs_map)

                chosen_rpn, derived = self._choose_executable_rpn(lhs, rhs, lhs_rpn, rhs_rpn)
                if not chosen_rpn and not derived:
                    continue

                conditions = self._extract_conditions(block_lines, lhs=lhs, rhs=rhs)
                conditions_rpn = [r for r in (self._condition_to_rpn(c) for c in conditions) if r]
                bindings = self._infer_symbol_bindings(
                    block_lines,
                    lhs=lhs,
                    rhs=rhs,
                    rpn=(chosen_rpn or rhs_rpn),
                    lhs_rpn=lhs_rpn,
                    rhs_rpn=rhs_rpn,
                    var_mapping=var_map,
                    book_domain_hint=domain,
                )

                artifacts.append(
                    KnowledgeArtifact(
                        artifact_id=artifact_id,
                        artifact_type=kind,
                        name=name,
                        domain=domain,
                        book_id=book_id,
                        page_number=int(page_number),
                        conditions=conditions,
                        conditions_rpn=conditions_rpn,
                        lhs=lhs,
                        rhs=rhs,
                        lhs_rpn=lhs_rpn,
                        rhs_rpn=rhs_rpn,
                        rpn=chosen_rpn or rhs_rpn,
                        conclusion=f"{lhs} = {rhs}",
                        conclusion_rpn=chosen_rpn or rhs_rpn,
                        derived_rpns=derived,
                        var_mapping=var_map,
                        symbol_bindings={k: asdict(v) for k, v in bindings.items()},
                        source=str(block.get("source") or "plain_text_block"),
                        raw_block=raw_block[:8000] if raw_block else None,
                    )
                )
            if len(artifacts) >= int(max_artifacts):
                break

        # If there's remaining budget, also emit a small number of "loose formulas"
        # (plain equation lines) that look high-signal (π/trig/sqrt/det/etc). This
        # helps books that don't preserve explicit "Theorem/Definition" markers in
        # pdftotext output, while still avoiding low-quality equation spam.
        remaining = int(max_artifacts) - int(len(artifacts))
        if remaining > 0:
            try:
                artifacts.extend(
                    self._extract_loose_formulas(
                        lines,
                        book_id=book_id,
                        domain=domain,
                        page_number=int(page_number),
                        max_items=min(12, remaining),
                        exclude_pairs=used_pairs,
                    )
                )
            except Exception:
                pass

        return artifacts

    def _identify_blocks(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            m = _BLOCK_HEAD_RE.match(ln)
            if not m:
                i += 1
                continue
            kind = m.group("kind").strip().lower()
            rest = (m.group("rest") or "").strip()
            name = self._extract_block_name(rest) if rest else None

            start = i
            i += 1
            buf: List[str] = [lines[start]]
            # Capture until next block header or a large gap.
            blank_run = 0
            while i < len(lines):
                cur = lines[i]
                cur_strip = cur.strip()
                if _BLOCK_HEAD_RE.match(cur_strip):
                    break
                buf.append(cur)
                if not cur_strip:
                    blank_run += 1
                    if blank_run >= 2:
                        i += 1
                        break
                else:
                    blank_run = 0
                i += 1

            blocks.append({"kind": kind, "name": name, "lines": buf})
        return blocks

    def _identify_latex_env_blocks(self, text: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for m in _LATEX_ENV_BLOCK_RE.finditer(text):
            kind = str(m.group("kind") or "").strip().lower()
            opt = str(m.group("opt") or "")
            name = opt.strip()[1:-1].strip() if opt.startswith("[") and opt.endswith("]") else None
            body = str(m.group("body") or "")
            raw = m.group(0)
            lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
            # Reuse block name extraction for cases like "\begin{theorem} (Name)".
            if not name:
                first = _strip_latex_noise(body).splitlines()[0] if body.strip() else ""
                name = self._extract_block_name(first) if first else None
            blocks.append({"kind": kind, "name": name, "lines": lines, "source": "latex_environment"})
        return blocks

    def _extract_block_name(self, rest: str) -> Optional[str]:
        # Examples: "1.2 (Pythagorean Theorem)" or "[Pythagorean Theorem]" or ": Pythagorean Theorem"
        rest = rest.strip()
        if not rest:
            return None
        m = re.search(r"\(([^)]+)\)", rest)
        if m:
            return m.group(1).strip()
        m = re.search(r"\[([^\]]+)\]", rest)
        if m:
            return m.group(1).strip()
        m = re.search(r":\s*(.+)$", rest)
        if m:
            return m.group(1).strip()
        # Otherwise, drop leading numbering.
        rest = re.sub(r"^[0-9.\-]+\s*", "", rest).strip()
        return rest or None

    def _extract_equation(self, block_lines: Sequence[str]) -> Optional[Tuple[str, str]]:
        for ln in block_lines:
            m = _EQ_LINE_RE.search(ln)
            if not m:
                continue
            lhs = m.group("lhs").strip()
            rhs = m.group("rhs").strip()
            # Filter trivial/too long.
            if not lhs or not rhs:
                continue
            if len(lhs) > 140 or len(rhs) > 260:
                continue
            # Avoid examples like "1*4 - 2*3 = -2" where RHS is just a number; still ok though.
            return (lhs, rhs)
        return None

    def _rpn_from_rhs(self, rhs: str) -> str:
        return self._rpn_from_infix(rhs)

    def _rpn_from_infix(self, expr: str) -> str:
        expr = _normalize_text(expr)
        expr = expr.strip().strip(".;")
        expr = re.split(r"\s{2,}|\s+#|\s+//", expr)[0].strip()
        try:
            return self._parser.infix_to_rpn(expr)
        except Exception:
            return ""

    def _choose_executable_rpn(
        self,
        lhs: str,
        rhs: str,
        lhs_rpn: str,
        rhs_rpn: str,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Pick an executable side (best-effort) and derive simple solved forms.

        Heuristics:
        - If either side looks like a single symbol/name, treat the other side as executable.
        - If equation matches `x^2 = expr` or `expr = x^2`, derive `sqrt(expr)` as `x`.
        """
        derived: List[Dict[str, Any]] = []

        def _is_single_symbol(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            # Accept single-letter vars and common function names like det, area, volume.
            if re.fullmatch(r"[a-zA-Z]", s):
                return True
            if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,15}", s):
                return True
            return False

        lhs_is_sym = _is_single_symbol(lhs)
        rhs_is_sym = _is_single_symbol(rhs)

        op_tokens = {
            "+",
            "-",
            "*",
            "/",
            "pow",
            "sqrt",
            "abs",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "log",
            "ln",
            "exp",
            "floor",
            "ceil",
            "gcd",
            "factorial",
            "binomial",
        }

        def _has_ops(rpn_expr: str) -> bool:
            return any(tok in op_tokens for tok in rpn_expr.split())

        chosen: Optional[str] = None
        if lhs_is_sym and _has_ops(rhs_rpn):
            chosen = rhs_rpn
        elif rhs_is_sym and _has_ops(lhs_rpn):
            chosen = lhs_rpn

        # Detect squared-variable patterns and derive sqrt form.
        # We detect on *normalized RPN* to avoid fragile infix parsing.
        # Pattern: "<v> 2 pow" by itself, and the other side contains ops.
        def _is_square_of_single_var(rpn_expr: str) -> Optional[str]:
            toks = rpn_expr.split()
            if len(toks) == 3 and toks[1] == "2" and toks[2] == "pow" and len(toks[0]) == 1 and toks[0].isalpha():
                return toks[0]
            return None

        lhs_sq = _is_square_of_single_var(lhs_rpn)
        rhs_sq = _is_square_of_single_var(rhs_rpn)
        if lhs_sq and _has_ops(rhs_rpn):
            derived.append({"kind": "solve_sqrt", "target": lhs_sq, "rpn": f"{rhs_rpn} sqrt"})
        if rhs_sq and _has_ops(lhs_rpn):
            derived.append({"kind": "solve_sqrt", "target": rhs_sq, "rpn": f"{lhs_rpn} sqrt"})

        # Fallback: if the equation is not an obvious "symbol = expression", pick
        # the more "computational" side (more operators) so the artifact remains
        # retrievable/executable after variable binding.
        if chosen is None:
            lhs_ops = sum(1 for tok in lhs_rpn.split() if tok in op_tokens)
            rhs_ops = sum(1 for tok in rhs_rpn.split() if tok in op_tokens)
            if lhs_ops or rhs_ops:
                chosen = lhs_rpn if lhs_ops > rhs_ops else rhs_rpn

        return (chosen, derived)

    def _extract_conditions(self, block_lines: Sequence[str], *, lhs: str, rhs: str) -> List[str]:
        # Keep only short condition-like statements.
        out: List[str] = []
        text_raw = " ".join(block_lines)
        text = _strip_latex_noise(text_raw)
        if lhs:
            text = text.replace(lhs, " ")
        if rhs:
            text = text.replace(rhs, " ")
        # Sentence-ish splits
        parts = re.split(r"[.;]\s+|\n+", text)
        for p in parts:
            s = p.strip()
            if not s:
                continue
            low = s.lower()
            if any(k in low for k in ["let ", "suppose", "assume", "if ", "where ", "for "]):
                if len(s) <= 180:
                    out.append(s)
        # Heuristic for pythagorean-style semantics.
        if "right triangle" in text.lower():
            out.append("triangle is right-angled")
        return out[:8]

    def _condition_to_rpn(self, condition: str) -> str:
        """
        Best-effort "predicate RPN" serializer.

        This is intentionally NOT validated against the runtime RPN engine yet;
        Phase 2 will decide how these predicates are executed (or matched) in TRM.
        """
        cond = _strip_latex_noise(condition)
        m = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*(<=|>=|!=|=|<|>)\s*([0-9]+(?:\.[0-9]+)?)", cond)
        if not m:
            return ""
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        return f"{lhs} {rhs} {op}"

    def _infer_symbol_bindings(
        self,
        block_lines: Sequence[str],
        *,
        lhs: str,
        rhs: str,
        rpn: str,
        lhs_rpn: Optional[str],
        rhs_rpn: Optional[str],
        var_mapping: Dict[str, str],
        book_domain_hint: Optional[str],
    ) -> Dict[str, SymbolBinding]:
        # Work with the normalized placeholder variables from RPN.
        placeholders: set[str] = set()
        for expr in (rpn, lhs_rpn or "", rhs_rpn or ""):
            placeholders.update({t for t in str(expr).split() if len(t) == 1 and t.isalpha()})
        bindings: Dict[str, SymbolBinding] = {p: SymbolBinding(symbol=p) for p in placeholders}

        raw = _strip_latex_noise(" ".join(block_lines))
        raw_low = raw.lower()

        def _set(ph: str, *, meaning: str, domain: str, domain_hint: Optional[str] = None) -> None:
            if ph not in bindings:
                return
            constraints = []
            if domain == "positive_real":
                constraints = [f"{ph} > 0"]
            if domain_hint is None:
                domain_hint = _DOMAIN_BY_ROLE.get(meaning)
            bindings[ph] = SymbolBinding(
                symbol=ph,
                meaning=meaning,
                domain=domain,
                domain_hint=domain_hint,
                constraints=constraints,
            )

        # -----------------------
        # Heuristic role inference
        # -----------------------
        # 1) Direct text patterns ("radius r", "height h", "legs a and b", etc).
        for role, pat in [
            ("radius", r"\bradius\s+\$?([a-z])\$?\b"),
            ("diameter", r"\bdiameter\s+\$?([a-z])\$?\b"),
            ("height", r"\bheight\s+\$?([a-z])\$?\b"),
            ("width", r"\bwidth\s+\$?([a-z])\$?\b"),
            ("length", r"\blength\s+\$?([a-z])\$?\b"),
            ("base", r"\bbase\s+\$?([a-z])\$?\b"),
            ("angle", r"\bangle\s+\$?([a-z])\$?\b"),
        ]:
            m = re.search(pat, raw_low)
            if m:
                orig = m.group(1)
                ph = var_mapping.get(orig, orig)
                dom = "angle" if role == "angle" else "positive_real"
                _set(ph, meaning=role, domain=dom)

        # Pythagorean theorem: "legs a and b", "hypotenuse c"
        m_leg = re.search(r"legs?\s+\$?([a-z])\$?\s+(?:and|,)\s+\$?([a-z])\$?", raw_low)
        if m_leg:
            a, b = m_leg.group(1), m_leg.group(2)
            _set(var_mapping.get(a, a), meaning="leg", domain="positive_real")
            _set(var_mapping.get(b, b), meaning="leg", domain="positive_real")
        m_hyp = re.search(r"hypotenuse\s+\$?([a-z])\$?", raw_low)
        if m_hyp:
            c = m_hyp.group(1)
            _set(var_mapping.get(c, c), meaning="hypotenuse", domain="positive_real")

        # 2) Equation-structure patterns (works even when prose is implicit).
        # Detect: (sum of squares) = (single square) on either side.
        def _vars_in_square_sum(expr: str) -> List[str]:
            toks = str(expr or "").split()
            out: List[str] = []
            for i in range(len(toks) - 2):
                if len(toks[i]) == 1 and toks[i].isalpha() and toks[i + 1] == "2" and toks[i + 2] == "pow":
                    out.append(toks[i])
            return out

        def _single_square_var(expr: str) -> Optional[str]:
            toks = str(expr or "").split()
            if len(toks) == 3 and len(toks[0]) == 1 and toks[0].isalpha() and toks[1] == "2" and toks[2] == "pow":
                return toks[0]
            return None

        lhs_sq = _vars_in_square_sum(lhs_rpn or "")
        rhs_sq = _vars_in_square_sum(rhs_rpn or "")
        lhs_single = _single_square_var(lhs_rpn or "")
        rhs_single = _single_square_var(rhs_rpn or "")
        if ("triangle" in raw_low or "right" in raw_low) and (lhs_sq or rhs_sq) and (lhs_single or rhs_single):
            # Leg vars are the ones in the sum; hypotenuse is the single-square side.
            legs = lhs_sq if lhs_sq else rhs_sq
            hyp = lhs_single if lhs_single else rhs_single
            for v in legs[:2]:
                _set(v, meaning="leg", domain="positive_real")
            if hyp:
                _set(hyp, meaning="hypotenuse", domain="positive_real")

        # Circle/cylinder patterns with π.
        rpn_low = str(rpn or "").lower()
        if ("π" in rpn or "pi" in rpn_low):
            # Strong structure signal: π * (v^2) usually indicates a radius-like role.
            # We intentionally don't require explicit "circle" text because PDF
            # prose often omits variable names.
            sq_vars = _vars_in_square_sum(rpn)
            if sq_vars:
                _set(sq_vars[0], meaning="radius", domain="positive_real")
                # If there's another variable multiplied in the same expression,
                # it's often a height (cylinder volume: π r^2 h).
                other_vars = [t for t in str(rpn or "").split() if len(t) == 1 and t.isalpha() and t not in sq_vars and t not in {"π"}]
                if other_vars and ("volume" in raw_low or "cylinder" in raw_low):
                    _set(other_vars[0], meaning="height", domain="positive_real")

        # Generic positivity cues.
        if "positive" in raw_low:
            for p in list(bindings.keys()):
                bnd = bindings[p]
                if bnd.domain == "unknown":
                    bindings[p] = SymbolBinding(symbol=p, meaning=bnd.meaning, domain="positive_real", constraints=[f"{p} > 0"])

        # -----------------------
        # Optional LLM augmentation
        # -----------------------
        if self._role_extractor is not None and self._role_extractor.enabled:
            eq_text = f"{lhs} = {rhs}"
            # `bindings` are keyed by normalized placeholder vars (single letters),
            # while the book context/equation usually mentions the *original* names.
            # Query the LLM with the original variable name when possible.
            inv_map: Dict[str, str] = {}
            try:
                for orig, ph in (var_mapping or {}).items():
                    o = str(orig or "").strip()
                    p = str(ph or "").strip()
                    if not o or not p:
                        continue
                    # Prefer a stable (shorter) representative for each placeholder.
                    prev = inv_map.get(p)
                    if prev is None or (len(o) < len(prev)):
                        inv_map[p] = o
            except Exception:
                inv_map = {}
            for ph, bnd in list(bindings.items()):
                if bnd.meaning != "unknown":
                    continue
                # Skip obvious constants (pi/e) and placeholders that are likely outputs.
                if ph in {"π", "pi", "e"}:
                    continue
                query_var = inv_map.get(ph) or ph
                role, dom, domain_hint = self._role_extractor.infer_role(
                    var=query_var,
                    context=raw,
                    equation=eq_text,
                    book_domain_hint=book_domain_hint,
                )
                if role and role != "unknown":
                    _set(ph, meaning=role, domain=dom or "real", domain_hint=domain_hint)

        return bindings

    def _extract_loose_formulas(
        self,
        lines: Sequence[str],
        *,
        book_id: str,
        domain: Optional[str],
        page_number: int,
        max_items: int = 12,
        exclude_pairs: Optional[set[tuple[str, str]]] = None,
    ) -> List[KnowledgeArtifact]:
        out: List[KnowledgeArtifact] = []
        # Keep this tight: only emit formulas with strong structural cues.
        # (This is "artifacts.jsonl", not "templates.jsonl".)
        relaxed = str(os.getenv("K3D_ARTIFACT_RELAXED", "0")).strip().lower() in {"1", "true", "yes"}
        high_signal = re.compile(
            r"(?i)(π|\bpi\b|sqrt\b|\bsin\b|\bcos\b|\btan\b|\barcsin\b|\barccos\b|\barctan\b|\bdet\b|determinant|\bln\b|\blog\b|\bexp\b|\blim\b|∫|∂|≡|\bmod\b|\bgcd\b|\blcm\b|\bmean\b|\bvariance\b)"
        )
        for ln in lines:
            m = _EQ_LINE_RE.search(ln)
            if not m:
                continue
            lhs = m.group("lhs").strip()
            rhs = m.group("rhs").strip()
            if not lhs or not rhs:
                continue
            # Skip low-signal equations (to avoid pollution from incidental examples).
            if not relaxed and not high_signal.search(ln):
                continue
            if exclude_pairs is not None and (lhs, rhs) in exclude_pairs:
                continue
            rpn_rhs = self._rpn_from_rhs(rhs)
            if not rpn_rhs:
                continue
            norm_rpn, var_map = _normalize_and_validate_rpn(rpn_rhs)
            if not norm_rpn:
                continue
            bindings = self._infer_symbol_bindings(
                [ln],
                lhs=lhs,
                rhs=rhs,
                rpn=norm_rpn,
                lhs_rpn=None,
                rhs_rpn=norm_rpn,
                var_mapping=var_map,
                book_domain_hint=domain,
            )
            artifact_id = f"{book_id}_p{int(page_number)}_formula_{len(out)}"
            out.append(
                KnowledgeArtifact(
                    artifact_id=artifact_id,
                    artifact_type="formula",
                    name=f"Formula p{int(page_number)}",
                    domain=domain,
                    book_id=book_id,
                    page_number=int(page_number),
                    conditions=[],
                    conditions_rpn=[],
                    lhs=lhs,
                    rhs=rhs,
                    rpn=norm_rpn,
                    conclusion=f"{lhs} = {rhs}",
                    conclusion_rpn=norm_rpn,
                    var_mapping=var_map,
                    symbol_bindings={k: asdict(v) for k, v in bindings.items()},
                    source="plain_text_equation",
                    raw_block=ln.strip()[:8000],
                )
            )
            if len(out) >= int(max_items):
                break
        return out


__all__ = ["SovereignKnowledgeArticulator", "KnowledgeArtifact", "SymbolBinding"]
