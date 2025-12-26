"""
Problem classifier for math benchmarks (deprecated).

Legacy regex preprocessing is replaced by sovereign Galaxy composition.
Use SovereignComposer + MathSymbolGalaxy instead of this classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

DEPRECATION_MSG = "ProblemClassifier is deprecated. Use SovereignComposer + MathSymbolGalaxy for sovereign parsing."


@dataclass
class ProblemClassification:
    problem_type: str
    subtype: str
    variables: List[str]
    coefficients: Dict[str, float]
    confidence: float


class ProblemClassifier:
    """Classify math problems using regex patterns."""

    def __init__(self):
        raise RuntimeError(DEPRECATION_MSG)

    PATTERNS = [
        (r"find the value|what is the value|compute|calculate|evaluate", None, "expression", "evaluate"),
        (r"\\binom|binom\(|choose", r"\d+", "combinatorics", "count"),
        (r"(\d+)!", None, "combinatorics", "factorial"),
        (r"probability|expected|random", None, "probability", "calculate"),
        (r"how many|in how many ways|number of ways", None, "combinatorics", "count"),
        (r"x\^2|x²|x\s*\*\s*x", r"=\s*0", "quadratic", "solve"),
        (r"factor.*x\^2|factor.*x²", None, "quadratic", "factor"),
        (r"roots?\s+of|zeros?\s+of|solutions?\s+(?:to|of)", r"x\^2|x²", "quadratic", "solve"),
        (r"solve.*[a-z]\s*[+\-*/].*=", r"(?!.*x\^2|.*x²)", "linear", "solve"),
        (r"find\s+[a-z]\s+(?:if|when|such)", None, "linear", "solve"),
        (r"system|simultaneous", r"equations?", "system", "solve"),
        (r"\{.*=.*\n.*=.*\}", None, "system", "solve"),
        (r"expand|multiply.*\(.*\)\s*\(.*\)", None, "polynomial", "expand"),
        (r"simplify", r"polynomial|expression", "polynomial", "simplify"),
        (r"degree\s+of|leading\s+coefficient", None, "polynomial", "analyze"),
        (r"how\s+many\s+ways|arrangements?|permutations?", None, "combinatorics", "count"),
        (r"choose|combinations?|select.*from", None, "combinatorics", "count"),
        (r"probability", None, "probability", "calculate"),
        (r"divisors?|factors?\s+of\s+\d+", None, "number_theory", "divisors"),
        (r"gcd|greatest\s+common|hcf", None, "number_theory", "gcd"),
        (r"lcm|least\s+common\s+multiple", None, "number_theory", "lcm"),
        (r"prime|composite", None, "number_theory", "primality"),
        (r"remainder|mod|modulo", None, "number_theory", "modular"),
        (r"arithmetic\s+(?:sequence|series|progression)", None, "sequence", "arithmetic"),
        (r"geometric\s+(?:sequence|series|progression)", None, "sequence", "geometric"),
        (r"sum\s+of\s+(?:first\s+)?\d+", None, "sequence", "sum"),
        (r"nth\s+term|find.*term", None, "sequence", "term"),
        (r"area|perimeter|circumference", None, "geometry", "measure"),
        (r"triangle|circle|rectangle|square", None, "geometry", "shape"),
        (r"angle|degree|radian", None, "geometry", "angle"),
        (r"evaluate|calculate|compute|find\s+the\s+value", None, "expression", "evaluate"),
        (r"simplify", None, "expression", "simplify"),
    ]

    def classify(self, problem_text: str) -> ProblemClassification:
        text_normalized = self._normalize_latex(problem_text)
        text_lower = text_normalized.lower()
        for primary, secondary, ptype, subtype in self.PATTERNS:
            if re.search(primary, text_lower):
                if secondary is None or re.search(secondary, text_lower):
                    variables = self._extract_variables(text_normalized)
                    coeffs = self._extract_coefficients(text_normalized)
                    conf = self._compute_confidence(text_lower, primary, secondary)
                    return ProblemClassification(ptype, subtype, variables, coeffs, conf)

        return ProblemClassification(
            "expression",
            "evaluate",
            self._extract_variables(problem_text),
            self._extract_coefficients(problem_text),
            0.3,
        )

    def _extract_variables(self, text: str) -> List[str]:
        matches = re.findall(r"\b([a-z])\b(?!\s*[=<>].*[a-z]\b)", text.lower())
        return list(set(matches))

    def _extract_coefficients(self, text: str) -> Dict[str, float]:
        coeffs: Dict[str, float] = {}
        txt = self._normalize_latex(text)

        binom_match = re.search(r"binom\((\d+),(\d+)\)", txt)
        if binom_match:
            coeffs["n"] = float(binom_match.group(1))
            coeffs["k"] = float(binom_match.group(2))

        frac_matches = re.findall(r"\((\d+)/(\d+)\)", txt)
        for i, (num, denom) in enumerate(frac_matches[:3]):
            coeffs[f"frac{i}_num"] = float(num)
            coeffs[f"frac{i}_denom"] = float(denom)

        quad_patterns = [
            r"(-?\d*)\s*x\^?2\s*([+\-])\s*(\d*)\s*x\s*([+\-])\s*(\d+)",
            r"x\^?2\s*([+\-])\s*(\d+)\s*x\s*([+\-])\s*(\d+)",
        ]
        for pattern in quad_patterns:
            match = re.search(pattern, txt.replace(" ", ""))
            if match:
                groups = match.groups()
                if len(groups) == 5:
                    a = groups[0] or "1"
                    coeffs["a"] = float(a) if a not in ("", "-") else (-1.0 if a == "-" else 1.0)
                    sign_b = 1 if groups[1] == "+" else -1
                    b = groups[2] or "1"
                    coeffs["b"] = sign_b * (float(b) if b else 1.0)
                    sign_c = 1 if groups[3] == "+" else -1
                    coeffs["c"] = sign_c * float(groups[4])
                break

        numbers = re.findall(r"\b(\d+\.?\d*)\b", txt)
        for i, n in enumerate(numbers[:5]):
            coeffs[f"n{i}"] = float(n)

        factorial_match = re.search(r"(\d+)!", txt)
        if factorial_match and "n" not in coeffs:
            coeffs["n"] = float(factorial_match.group(1))

        return coeffs

    def _compute_confidence(self, text: str, primary: str, secondary: Any) -> float:
        primary_matches = len(re.findall(primary, text))
        conf = min(0.5 + 0.1 * primary_matches, 0.9)
        if secondary and re.search(secondary, text):
            conf = min(conf + 0.1, 0.95)
        return conf

    def _normalize_latex(self, text: str) -> str:
        result = text
        result = re.sub(r"\$\$?", "", result)
        result = re.sub(r"\\\[|\\\]", "", result)
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        while r"\frac" in result:
            result = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1/\2)", result)
        result = re.sub(r"\\binom\{(\d+)\}\{(\d+)\}", r"binom(\1,\2)", result)
        result = re.sub(r"\^{(\d+)}", r"^\1", result)
        result = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", result)
        result = re.sub(r"\\[a-zA-Z]+", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result
