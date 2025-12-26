"""
Sovereign Expression Composer - Composes RPN from input using Galaxy.

NO external preprocessing. The Galaxy IS the model's knowledge.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY


FuncToken = Tuple[str, str, List[List[Any]]]


class SovereignComposer:
    """
    Composes RPN programs from input expressions using the Math Galaxy.

    This is sovereign: symbols are looked up in the Galaxy, not processed
    externally. The Galaxy stores the logic.
    """

    def __init__(self):
        self.galaxy = MATH_GALAXY

    def compose(self, expression: str, return_tokens: bool = False) -> Any:
        """
        Compose RPN program from expression using Galaxy lookups.

        The model "sees" the expression, matches symbols against Galaxy,
        and composes RPN. No external preprocessing. Returns a space-separated
        RPN string unless return_tokens=True.
        """
        tokens = self._tokenize(expression)
        rpn_tokens = self._to_rpn(tokens)
        if return_tokens:
            return rpn_tokens
        return self._format_rpn(rpn_tokens)

    def _tokenize(self, expr: str) -> List[Any]:
        """Tokenize expression into symbols the Galaxy understands."""
        tokens: List[Any] = []
        i = 0

        while i < len(expr):
            ch = expr[i]

            if ch.isspace():
                i += 1
                continue

            # Handle Euler's constant 'e' carefully to avoid matching inside words
            if ch == "e":
                if self._is_euler_constant(expr, i):
                    tokens.append("e")
                i += 1
                continue

            # LaTeX commands: \frac, \binom, \sqrt, etc.
            if ch == "\\":
                j = i + 1
                while j < len(expr) and expr[j].isalpha():
                    j += 1
                cmd = expr[i:j]

                if self.galaxy.lookup(cmd):
                    i = j
                    args: List[List[Any]] = []
                    while i < len(expr) and expr[i] == "{":
                        arg_text, new_i = self._extract_braced(expr, i)
                        args.append(self._tokenize(arg_text))
                        i = new_i
                    tokens.append(("FUNC", cmd, args))
                    continue
                i = j
                continue

            if ch in "()":
                tokens.append(ch)
                i += 1
                continue

            if ch.isdigit() or (ch == "." and i + 1 < len(expr) and expr[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(expr):
                    c = expr[j]
                    if c.isdigit():
                        j += 1
                    elif c == "." and not has_dot:
                        # Only accept dot if followed by a digit
                        if j + 1 < len(expr) and expr[j + 1].isdigit():
                            has_dot = True
                            j += 1
                        else:
                            break
                    else:
                        break
                tokens.append(expr[i:j])
                i = j
                continue

            sym = expr[i]
            if self.galaxy.lookup(sym):
                tokens.append(sym)
            i += 1

        return tokens

    def _is_euler_constant(self, expr: str, idx: int) -> bool:
        """
        Decide if 'e' at position idx is Euler's constant.

        Treat as Euler if not part of an alpha word and appears in math context.
        """
        if idx < 0 or idx >= len(expr) or expr[idx] != "e":
            return False

        prev = expr[idx - 1] if idx > 0 else ""
        nxt = expr[idx + 1] if idx + 1 < len(expr) else ""

        # Inside a word (letters on either side) → not Euler
        if prev.isalpha() or nxt.isalpha():
            return False

        # Math context: start/end of string or adjacent to operator/brace/caret
        math_neighbors = set("+-*/^({[= ")
        if prev in math_neighbors or nxt in math_neighbors or prev == "" or nxt == "":
            return True

        return False

    def _extract_braced(self, expr: str, start: int) -> Tuple[str, int]:
        """Extract text inside matching braces starting at index start."""
        brace_count = 0
        i = start
        content_start = start + 1
        while i < len(expr):
            if expr[i] == "{":
                brace_count += 1
            elif expr[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return expr[content_start:i], i + 1
            i += 1
        return expr[content_start:], len(expr)

    def _render_function(self, token: FuncToken) -> List[str]:
        """Render a function token using Galaxy templates."""
        _, cmd, args_tokens = token
        rendered_args = [self._format_rpn(self._to_rpn(arg)) for arg in args_tokens]
        template = self.galaxy.compose_rpn(cmd, *rendered_args)
        return template.split()

    def _to_rpn(self, tokens: Sequence[Any]) -> List[str]:
        """
        Convert tokens to RPN using Galaxy symbol information.

        Uses precedence and associativity from Galaxy entries.
        """
        output: List[str] = []
        op_stack: List[Any] = []

        for tok in tokens:
            if isinstance(tok, tuple) and tok[0] == "FUNC":
                output.extend(self._render_function(tok))
                continue

            if tok == "(":
                op_stack.append(tok)
                continue
            if tok == ")":
                while op_stack and op_stack[-1] != "(":
                    op = op_stack.pop()
                    output.extend(self._opcode_tokens(op))
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
                continue

            # Number literal
            if isinstance(tok, str) and (tok[0].isdigit() or tok.startswith(".")):
                try:
                    # Clean trailing dots and convert
                    clean_tok = tok.rstrip(".")
                    if clean_tok:
                        output.append(str(float(clean_tok)))
                except ValueError:
                    pass
                continue

            sym_entry = self.galaxy.lookup(tok) if isinstance(tok, str) else None
            if sym_entry:
                if sym_entry.category == "constant":
                    try:
                        output.append(str(float(sym_entry.rpn_template)))
                    except (TypeError, ValueError):
                        pass
                    continue
                if sym_entry.category in {"operator", "relation"}:
                    while (
                        op_stack
                        and op_stack[-1] != "("
                        and isinstance(op_stack[-1], str)
                        and self._should_pop_operator(tok, op_stack[-1])
                    ):
                        op = op_stack.pop()
                        output.extend(self._opcode_tokens(op))
                    op_stack.append(tok)
                    continue
                if sym_entry.category == "function":
                    output.extend(self._opcode_tokens(tok))
                    continue

            # Fallback: treat as literal token
            output.append(str(tok))

        while op_stack:
            op = op_stack.pop()
            if op != "(":
                output.extend(self._opcode_tokens(op))

        return output

    def _should_pop_operator(self, current: str, top: str) -> bool:
        """Determine whether to pop operator on stack based on precedence/associativity."""
        cur = self.galaxy.lookup(current)
        prev = self.galaxy.lookup(top)
        if not cur or not prev:
            return False

        if cur.associativity == "left" and cur.precedence <= prev.precedence:
            return True
        if cur.associativity == "right" and cur.precedence < prev.precedence:
            return True
        return False

    def _opcode_tokens(self, symbol: str) -> List[str]:
        """Return opcode tokens for a symbol using its template."""
        entry = self.galaxy.lookup(symbol)
        if not entry:
            return [symbol]
        if not entry.rpn_template:
            return []
        return [entry.rpn_template.split()[-1]]

    def _format_rpn(self, tokens: Sequence[Any]) -> str:
        """Format tokens into space-separated RPN string."""
        return " ".join(str(t) for t in tokens if str(t).strip())
