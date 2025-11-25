"""End-to-end semantic pipeline tests for ARC instructions."""

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def test_move_red_to_bottom_right():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Move the red object to the bottom-right corner"
    input_grid = [
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_move_red_to_bottom_right PASSED")


def test_fill_center_with_blue():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Fill the center with blue"
    input_grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_fill_center_with_blue PASSED")


def test_rotate_90_clockwise():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Rotate the pattern 90 degrees clockwise"
    input_grid = [
        [1, 0],
        [1, 1],
    ]
    expected_output = [
        [1, 1],
        [1, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_rotate_90_clockwise PASSED")


def test_flip_horizontal():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Flip the pattern horizontally"
    input_grid = [
        [1, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 1],
        [0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_flip_horizontal PASSED")


def test_continue_sequence_right():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Continue the sequence to the right"
    input_grid = [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    expected_output = [
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_continue_sequence_right PASSED")


def run_all_tests():
    """Run all end-to-end semantic pipeline tests."""
    tests = [
        test_move_red_to_bottom_right,
        test_fill_center_with_blue,
        test_rotate_90_clockwise,
        test_flip_horizontal,
        test_continue_sequence_right,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
