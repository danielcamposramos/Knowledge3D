"""Quick test for new swiglu operation in RPN engine."""


def main() -> int:
    import sys
    from pathlib import Path

    import numpy as np

    # Test without importing the full package to avoid cuda.bindings issues
    _root = Path(__file__).parent.parent
    sys.path.insert(0, str(_root / "knowledge3d/cranium/ptx_runtime"))

    try:
        from modular_rpn_engine import ModularRPNEngine

        print("Initializing RPN engine...")
        engine = ModularRPNEngine()

        print("\nTesting swiglu operation...")
        # swiglu: x * sigmoid(gate)
        # Stack order: push x, push gate, call swiglu -> result

        # Test case 1: x=2.0, gate=0.0
        # Expected: 2.0 * sigmoid(0.0) = 2.0 * 0.5 = 1.0
        result1 = engine.evaluate("2.0 0.0 swiglu")
        print(f"Test 1: 2.0 * sigmoid(0.0) = {result1[0]:.6f} (expected: 1.0)")

        # Test case 2: x=1.0, gate=2.0
        # Expected: 1.0 * sigmoid(2.0) = 1.0 * 0.88079... ≈ 0.88079
        result2 = engine.evaluate("1.0 2.0 swiglu")
        expected2 = 1.0 / (1.0 + np.exp(-2.0))
        print(
            f"Test 2: 1.0 * sigmoid(2.0) = {result2[0]:.6f}"
            f" (expected: {expected2:.6f})"
        )

        # Test case 3: x=3.0, gate=-1.0
        # Expected: 3.0 * sigmoid(-1.0) = 3.0 * 0.26894... ≈ 0.80685
        result3 = engine.evaluate("3.0 -1.0 swiglu")
        expected3 = 3.0 / (1.0 + np.exp(1.0))
        print(
            f"Test 3: 3.0 * sigmoid(-1.0) = {result3[0]:.6f}"
            f" (expected: {expected3:.6f})"
        )

        # Verify results
        assert abs(result1[0] - 1.0) < 1e-5, "Test 1 failed"
        assert abs(result2[0] - expected2) < 1e-5, "Test 2 failed"
        assert abs(result3[0] - expected3) < 1e-5, "Test 3 failed"

        print("\n✅ All swiglu tests PASSED!")
        print("RPN engine successfully extended with TRM activation function.")

        engine.close()
        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
