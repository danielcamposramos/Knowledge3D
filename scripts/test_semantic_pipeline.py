"""Smoke tests for semantic parsing -> RPN compile -> execution."""

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def run_tests():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    tests = [
        {
            "instruction": "Rotate the pattern 90 degrees clockwise",
            "input": [[1, 0], [0, 0]],
            "expected": [[0, 1], [0, 0]],
        },
        {
            "instruction": "Move the red object to the bottom-right corner",
            "input": [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            "expected": [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
        },
        {
            "instruction": "Fill the rectangle with red",
            "input": [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "expected_center": 2,  # red fill in center after fill
        },
        {
            "instruction": "Flip the pattern horizontally",
            "input": [[1, 0], [0, 0]],
            "expected": [[0, 1], [0, 0]],
        },
        {
            "instruction": "Flip the pattern vertically",
            "input": [[1, 0], [0, 0]],
            "expected": [[0, 0], [1, 0]],
        },
        {
            "instruction": "Change red to blue",
            "input": [[2, 0], [0, 0]],
            "expected": [[1, 0], [0, 0]],
        },
        {
            "instruction": "Copy red object to the top-right corner",
            "input": [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            "expected": [[2, 0, 2], [0, 0, 0], [0, 0, 0]],
        },
        {
            "instruction": "Continue the sequence to the right",
            "input": [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
            "expected": [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
        },
    ]

    passed = 0
    for i, case in enumerate(tests, start=1):
        sem = parser.parse(case["instruction"])
        rpn = compiler.compile(sem)
        result = executor.execute(case["input"], rpn)

        if "expected_center" in case:
            ok = result[2][2] == case["expected_center"]
        else:
            ok = result == case["expected"]

        status = "✅" if ok else "❌"
        print(f"{status} Test {i}: {case['instruction']}")
        if not ok:
            print(f"   Expected: {case.get('expected', 'center=' + str(case.get('expected_center')))}")
            print(f"   Got:      {result}")
        else:
            passed += 1

    total = len(tests)
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    run_tests()
