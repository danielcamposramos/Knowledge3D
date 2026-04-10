"""Ingestion-path CAS utilities.

SymEngine is used only here to parse math expressions and emit K3D RPN text.
It must never leak into the hot PTX/bridge/runtime execution surface.
"""

from __future__ import annotations

import hashlib


def _stable_symbol_id(name: str) -> int:
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return int(digest[:4], 16)


def _fold_variadic(parts: list[str], op_token: str) -> str:
    if not parts:
        raise ValueError(f"Cannot fold empty variadic node for {op_token}")
    if len(parts) == 1:
        return parts[0]
    acc = f"{parts[0]} {parts[1]} {op_token}"
    for part in parts[2:]:
        acc = f"{acc} {part} {op_token}"
    return acc


def expression_to_rpn(expr_str: str) -> str:
    """Parse a math expression string via SymEngine and return K3D RPN text."""
    try:
        import symengine as se
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "symengine is required for CAS ingestion. "
            "Install with: pip install symengine"
        ) from exc

    expr = se.sympify(expr_str)
    return _symengine_to_rpn(expr)


def _symengine_to_rpn(expr) -> str:
    """Recursive post-order traversal of SymEngine AST → K3D RPN string."""
    import symengine as se

    if getattr(expr, "is_Number", False):
        return f"OP_CONST {float(expr)}"
    if getattr(expr, "is_Symbol", False):
        name = str(expr)
        sym_map = {"x": "OP_VAR_X", "y": "OP_VAR_Y", "z": "OP_VAR_Z", "w": "OP_VAR_W"}
        return sym_map.get(name, f"OP_CAS_PUSH_SYM {_stable_symbol_id(name)}")

    func_name = getattr(getattr(expr, "func", None), "__name__", type(expr).__name__).lower()

    if isinstance(expr, se.Add):
        return _fold_variadic([_symengine_to_rpn(arg) for arg in expr.args], "OP_ADD")
    if isinstance(expr, se.Mul):
        return _fold_variadic([_symengine_to_rpn(arg) for arg in expr.args], "OP_MUL")
    if isinstance(expr, se.Pow):
        base, exp = expr.args
        return f"{_symengine_to_rpn(base)} {_symengine_to_rpn(exp)} OP_POWER"
    if func_name == "sin":
        return f"{_symengine_to_rpn(expr.args[0])} OP_SIN"
    if func_name == "cos":
        return f"{_symengine_to_rpn(expr.args[0])} OP_COS"
    if func_name == "exp":
        return f"{_symengine_to_rpn(expr.args[0])} OP_EXP"
    if func_name == "log":
        return f"{_symengine_to_rpn(expr.args[0])} OP_LOG"

    raise ValueError(f"Unsupported SymEngine node: {type(expr)} {expr}")


__all__ = ["expression_to_rpn"]
