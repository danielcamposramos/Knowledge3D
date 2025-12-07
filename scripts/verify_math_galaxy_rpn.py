#!/usr/bin/env python3
"""
Verify all Math Galaxy RPN programs compile without errors.

Ensures procedural knowledge is executable by the Modular RPN engine.
"""

from knowledge3d.cranium.math_galaxy import get_math_galaxy
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


def main() -> bool:
    galaxy = get_math_galaxy()
    engine = ModularRPNEngine()

    success = 0
    failed = 0

    print("Verifying Math Galaxy RPN compilation...")
    print("=" * 50)

    for codepoint, symbol in galaxy.symbols.items():
        try:
            tokens = engine.tokenize_rpn(symbol.rpn_program)
            op_codes, scalars, vectors = engine.compile_tokens(tokens)
            success += 1
            print(f"  \u2713 {symbol.char} (U+{codepoint:04X}): {len(op_codes)} ops")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  \u2717 {symbol.char} (U+{codepoint:04X}): {exc}")

    total = len(galaxy.symbols)
    print("=" * 50)
    print(f"Success: {success}/{total}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n\u2713 All Math Galaxy RPN is executable by TRM!")
    else:
        print(f"\n\u2717 {failed} symbols need RPN fixes")

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
