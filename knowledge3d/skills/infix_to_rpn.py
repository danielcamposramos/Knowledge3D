from __future__ import annotations

"""Infix → RPN converter tailored to the ModularRPNEngine.

Supports:
- Numbers (integers/decimals), constants: pi, π, tau, phi, φ, e
- Operators: +, -, *, /, ^ (unary minus becomes ``neg``)
- Functions: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
  exp, log, log10, sqrt, abs, relu, sigmoid
- Special: log2(x) is lowered to ``x log 2 log /`` to use natural log

Returns a list of RPN tokens compatible with the ModularRPNEngine
(`knowledge3d.cranium.phase10.modular_rpn_engine`).
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import os as _os


_CONST = {"pi", "π", "tau", "phi", "φ", "e"}
_FUNCS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "relu",
    "sigmoid",
    # pseudo-func lowered: log2
    "log2",
    # additional
    "floor",
    "ceil",
    "mod",
    "round",
    "round_he",
    "clamp",
    "gcd",
    "lcm",
    "nCr",
    "nPr",
    "store",
    "load",
    "fact",
}

_OP_INFO: Dict[str, Tuple[int, str]] = {
    "+": (2, "L"),
    "-": (2, "L"),
    "*": (3, "L"),
    "/": (3, "L"),
    "%": (3, "L"),
    "^": (4, "R"),
}


_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")
_ID_RE = re.compile(r"^[A-Za-zπφΦ_][A-Za-z0-9_πφΦ]*$")


def _is_number(tok: str) -> bool:
    return bool(_NUM_RE.match(tok))


def _is_identifier(tok: str) -> bool:
    return bool(_ID_RE.match(tok))


def _pre_normalize(expr: str) -> str:
    # Convert common LaTeX forms to plain funcs
    s = expr or ""
    s = s.replace("\\", "\\")  # keep escapes
    # \sqrt{X} -> sqrt(X)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # \ln(...) -> log(...)
    s = re.sub(r"\\ln\s*\(", "log(", s)
    s = re.sub(r"\\ln\s*\{", "log(", s)
    # \log_{b}(x) or \log_{b}{x} -> log(x)/log(b)
    s = re.sub(r"\\log_\{([^{}]+)\}\(([^()]+)\)", r"(\2)/(log(\1))", s)
    s = re.sub(r"\\log_\{([^{}]+)\}\{([^{}]+)\}", r"(\2)/(log(\1))", s)
    # \frac{a}{b} -> (a)/(b)
    def _frac(m: re.Match) -> str:
        a = m.group(1)
        b = m.group(2)
        return f"({a})/({b})"
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _frac, s)
    # \binom{n}{k} -> nCr(n,k)
    s = re.sub(r"\\binom\{([^{}]+)\}\{([^{}]+)\}", r"nCr(\1,\2)", s)
    # \lfloor x \rfloor and \lceil x \rceil
    s = re.sub(r"\\lfloor\s*([^{}()]+)\s*\\rfloor", r"floor(\1)", s)
    s = re.sub(r"\\lceil\s*([^{}()]+)\s*\\rceil", r"ceil(\1)", s)
    # Unicode floor/ceil glyphs ⌊x⌋, ⌈x⌉
    s = re.sub(r"⌊\s*([^⌋]+)\s*⌋", r"floor(\1)", s)
    s = re.sub(r"⌈\s*([^⌉]+)\s*⌉", r"ceil(\1)", s)
    # Absolute value bars |x| → abs(x) (iterative, non-nested best-effort)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    # Aliases: C(n,k)->nCr(n,k); P(n,k)->nPr(n,k)
    s = re.sub(r"\bC\s*\(([^,]+),\s*([^\)]+)\)", r"nCr(\1,\2)", s)
    s = re.sub(r"\bP\s*\(([^,]+),\s*([^\)]+)\)", r"nPr(\1,\2)", s)
    # Aliases: choose/perm
    s = re.sub(r"\bchoose\s*\(([^,]+),\s*([^\)]+)\)", r"nCr(\1,\2)", s)
    s = re.sub(r"\bperm\s*\(([^,]+),\s*([^\)]+)\)", r"nPr(\1,\2)", s)
    # lg(x) -> log10(x)
    s = re.sub(r"\blg\s*\(", "log10(", s)
    # Unicode sqrt: √(x) -> sqrt(x)
    s = re.sub(r"√\s*\(", "sqrt(", s)
    # \lfloor x \rfloor and \lceil x \rceil
    s = re.sub(r"\\lfloor\s*([^{}()]+)\s*\\rfloor", r"floor(\1)", s)
    s = re.sub(r"\\lceil\s*([^{}()]+)\s*\\rceil", r"ceil(\1)", s)
    # Unicode floor/ceil glyphs ⌊x⌋, ⌈x⌉
    s = re.sub(r"⌊\s*([^⌋]+)\s*⌋", r"floor(\1)", s)
    s = re.sub(r"⌈\s*([^⌉]+)\s*⌉", r"ceil(\1)", s)
    # Absolute value bars |x| → abs(x) (simple, non-nested)
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    # Aliases: C(n,k)->nCr(n,k); P(n,k)->nPr(n,k)
    s = re.sub(r"\bC\s*\(([^,]+),\s*([^\)]+)\)", r"nCr(\1,\2)", s)
    s = re.sub(r"\bP\s*\(([^,]+),\s*([^\)]+)\)", r"nPr(\1,\2)", s)
    # Rounding mode mapping
    mode = (_os.environ.get("K3D_RPN_ROUND_MODE", "half_up").strip().lower())
    if mode in {"half_even", "bankers", "he"}:
        s = re.sub(r"\bround\s*\(", "round_he(", s)
    # Textual patterns: sum/difference/product/quotient of A and B
    # Make them non-greedy to the next ' and ' occurrence
    s = re.sub(r"\bsum of\s+([^,;]+?)\s+and\s+([^,;]+)", r"(\1)+(\2)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdifference of\s+([^,;]+?)\s+and\s+([^,;]+)", r"(\1)-(\2)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bproduct of\s+([^,;]+?)\s+and\s+([^,;]+)", r"(\1)*(\2)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bquotient of\s+([^,;]+?)\s+and\s+([^,;]+)", r"(\1)/(\2)", s, flags=re.IGNORECASE)
    # Word operators: plus/minus/times/divided by
    s = re.sub(r"\bdivided by\b", "/", s, flags=re.IGNORECASE)
    s = re.sub(r"\btimes\b", "*", s, flags=re.IGNORECASE)
    s = re.sub(r"\bplus\b", "+", s, flags=re.IGNORECASE)
    s = re.sub(r"\bminus\b", "-", s, flags=re.IGNORECASE)
    # Remainder phrase → mod
    s = re.sub(r"\bremainder of\s+([^,;]+?)\s+divided by\s+([^,;]+)", r"mod(\1,\2)", s, flags=re.IGNORECASE)
    return s


def _tokenize(expr: str) -> List[str]:
    expr = _pre_normalize(expr)
    # Normalize unicode minus
    expr = (expr or "").replace("−", "-")
    # Keep letters/digits/operators/parentheses/commas/periods/underscores
    # and split by whitespace while preserving operators
    tokens: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/^(),!":
            tokens.append(ch)
            i += 1
            continue
        # number (with optional decimal)
        if ch.isdigit() or ch == ".":
            j = i + 1
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        # identifier / constant / function name
        if ch.isalpha() or ch in {"π", "φ", "Φ", "_"}:
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] in {"π", "φ", "Φ", "_"}):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        # Unknown char → skip
        i += 1
    return tokens


def infix_to_rpn(expr: str, variables: Optional[Dict[str, float]] = None) -> List[str]:
    """Convert infix expression to RPN tokens suitable for the PTX RPN engine.

    Implements the Shunting Yard algorithm with support for unary minus.
    """
    variables = variables or {}
    toks = _tokenize(expr)
    out: List[str] = []
    ops: List[str] = []

    def push_func(name: str) -> None:
        ops.append(name)

    prev: Optional[str] = None
    for tok in toks:
        if _is_number(tok):
            out.append(tok)
        elif tok in variables or tok in _CONST:
            out.append(tok)
        elif tok in _FUNCS or tok == "ln":
            push_func(tok)
        elif tok == ",":
            # function argument separator: pop until '(' found
            while ops and ops[-1] != "(":
                out.append(ops.pop())
        elif tok in _OP_INFO:
            # unary minus → neg
            if tok == "-" and (prev is None or prev in _OP_INFO or prev in {"(", ","}):
                # represent unary minus as neg function (arity 1)
                push_func("neg")
            else:
                p1, assoc1 = _OP_INFO[tok]
                while ops and ops[-1] in _OP_INFO:
                    p2, _assoc2 = _OP_INFO[ops[-1]]
                    if (assoc1 == "L" and p1 <= p2) or (assoc1 == "R" and p1 < p2):
                        out.append(ops.pop())
                    else:
                        break
                ops.append(tok)
        elif tok == "!":
            # Factorial postfix → RPN 'fact'
            out.append("fact")
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            if ops and ops[-1] == "(":
                ops.pop()
            # if function on top, pop it too
            if ops and (ops[-1] in _FUNCS or ops[-1] == "neg"):
                out.append(ops.pop())
        else:
            # Unknown identifier (variable) → push as symbol literal
            if _is_identifier(tok):
                out.append(tok)
        prev = tok

    while ops:
        out.append(ops.pop())

    # Lower pseudo funcs into proper RPN sequences
    lowered: List[str] = []
    i = 0
    n = len(out)
    while i < n:
        t = out[i]
        if t == "log2":
            # log2(x) → x log 2 log /
            # Since this will appear after RPN conversion, replace the function token
            # by the sequence [log, 2, log, /]. The operand is already on stack.
            lowered.extend(["log", "2", "log", "/"])
        elif t == "ln":
            lowered.append("log")
        else:
            lowered.append(t)
        i += 1
    return lowered


def program_to_rpn(text: str) -> List[str]:
    """Compile simple math 'programs' with assignments into RPN tokens.

    Syntax: sequence of statements separated by ';' or newlines. Each statement
    can be either an assignment:
      let x = expr
      x = expr
    or a bare expression. Identifiers are mapped to 16 registers (0..15) and
    lowered to load/store ops.
    The final value on stack is the last statement's result.
    """
    if not text:
        return []
    # Split into statements naïvely by semicolons/newlines
    stmts = [s.strip() for s in re.split(r"[;\n]+", text) if s.strip()]
    if not stmts:
        return []
    var_order: List[str] = []
    var_to_reg: Dict[str, int] = {}

    def _get_reg(name: str) -> int:
        if name not in var_to_reg:
            if len(var_order) >= 16:
                raise ValueError("Exceeded max 16 registers in RPN program")
            var_to_reg[name] = len(var_order)
            var_order.append(name)
        return var_to_reg[name]

    def _replace_vars(tokens: List[str]) -> List[str]:
        replaced: List[str] = []
        for t in tokens:
            if _is_identifier(t) and t not in _CONST and t not in _FUNCS and t not in _OP_INFO and t not in {"ln"}:
                idx = _get_reg(t)
                replaced.append(str(idx))
                replaced.append("load")
            else:
                replaced.append(t)
        return replaced

    out: List[str] = []
    for raw in stmts:
        m = re.match(r"^(?:let\s+)?([A-Za-z_πφΦ][A-Za-z0-9_πφΦ]*)\s*=\s*(.+)$", raw)
        if m:
            name = m.group(1)
            expr = m.group(2).strip()
            toks = infix_to_rpn(expr)
            toks = _replace_vars(toks)
            # store: value (already on top), index
            idx = _get_reg(name)
            out.extend(toks)
            out.append(str(idx))
            out.append("store")
        else:
            # Bare expression
            toks = infix_to_rpn(raw)
            toks = _replace_vars(toks)
            out.extend(toks)
    return out


def program_to_rpn_with_trace(text: str) -> Tuple[List[str], Dict[str, int]]:
    """Compile a simple math program and also return the register map.

    Returns a tuple: (rpn_tokens, var_to_reg_map) where var_to_reg_map maps
    variable names to assigned register indices (0..15).
    """
    if not text:
        return [], {}
    # Re-implement using the same logic as program_to_rpn, preserving order
    stmts = [s.strip() for s in re.split(r"[;\n]+", text) if s.strip()]
    if not stmts:
        return [], {}
    var_order: List[str] = []
    var_to_reg: Dict[str, int] = {}

    def _get_reg(name: str) -> int:
        if name not in var_to_reg:
            if len(var_order) >= 16:
                raise ValueError("Exceeded max 16 registers in RPN program")
            var_to_reg[name] = len(var_order)
            var_order.append(name)
        return var_to_reg[name]

    def _replace_vars(tokens: List[str]) -> List[str]:
        replaced: List[str] = []
        for t in tokens:
            if _is_identifier(t) and t not in _CONST and t not in _FUNCS and t not in _OP_INFO and t not in {"ln"}:
                idx = _get_reg(t)
                replaced.append(str(idx))
                replaced.append("load")
            else:
                replaced.append(t)
        return replaced

    out: List[str] = []
    for raw in stmts:
        m = re.match(r"^(?:let\s+)?([A-Za-z_πφΦ][A-Za-z0-9_πφΦ]*)\s*=\s*(.+)$", raw)
        if m:
            name = m.group(1)
            expr = m.group(2).strip()
            toks = infix_to_rpn(expr)
            toks = _replace_vars(toks)
            idx = _get_reg(name)
            out.extend(toks)
            out.append(str(idx))
            out.append("store")
        else:
            toks = infix_to_rpn(raw)
            toks = _replace_vars(toks)
            out.extend(toks)
    return out, var_to_reg


def extract_math_expression(text: str) -> Optional[str]:
    """Heuristically extract a math expression substring from free text.

    Strategy:
    - Look after triggers like 'evaluate', 'compute', 'what is', 'find the value of'
      and keep characters in the math alphabet (digits, ops, funcs, parentheses).
    - If not found, return longest math-like span in the string.
    """
    if not text:
        return None
    lower = text.lower()
    triggers = [
        "evaluate ",
        "compute ",
        "what is ",
        "find the value of ",
        "find ",
    ]
    for trig in triggers:
        idx = lower.find(trig)
        if idx != -1:
            start = idx + len(trig)
            candidate = text[start:]
            # Trim to first sentence end (avoid truncating decimals)
            end_idx = len(candidate)
            for stop in ["?", "\n", ";"]:
                j = candidate.find(stop)
                if j != -1:
                    end_idx = min(end_idx, j)
            span = candidate[:end_idx]
            span = span.strip()
            span = re.sub(r"[^A-Za-z0-9_πφΦ+\-*/^().,]", " ", span)
            return span or None
    # fallback: longest math-like span
    spans = re.findall(r"[A-Za-z0-9_πφΦ+\-*/^().,]{6,}", text)
    if spans:
        # choose the longest token span
        spans.sort(key=len, reverse=True)
        return spans[0]
    return None


__all__ = ["infix_to_rpn", "extract_math_expression", "program_to_rpn", "program_to_rpn_with_trace"]
