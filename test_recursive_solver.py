"""
Unit test for the RecursiveSolver.
Verifies it can handle the Schaum's Outline examples.
"""
import unittest
from knowledge3d.training.math_benchmarks.recursive_solver import RecursiveSolver

class TestRecursiveSolver(unittest.TestCase):
    def setUp(self):
        self.solver = RecursiveSolver(verbose=True)

    def test_schaum_quotient(self):
        # Schaum's Prob 4.37a: f(x) = (3x-4)/(2x+3) at x=1. Ans: 17/25 = 0.68
        # f'(x) = [3(2x+3) - (3x-4)(2)] / (2x+3)^2
        # f'(1) = [3(5) - (-1)(2)] / 5^2 = [15 + 2] / 25 = 17/25
        problem = "Given f(x) = (3*x - 4)/(2*x + 3), find f'(1)"
        result = self.solver.solve(problem)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.68, places=5)

    def test_schaum_sum_power(self):
        # Schaum's Prob 4.37b: f(x) = x^3 - 3x^2 + 2x - 5 at x=2. Ans: 2
        # f'(x) = 3x^2 - 6x + 2
        # f'(2) = 3(4) - 12 + 2 = 12 - 12 + 2 = 2
        problem = "Given f(x) = x^3 - 3*x^2 + 2*x - 5, find f'(2)"
        result = self.solver.solve(problem)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=5)

    def test_schaum_chain_root(self):
        # Schaum's Prob 4.37d: f(x) = (6x - 4)^(1/3) at x=2. Ans: 1/2 = 0.5
        # f'(x) = (1/3)(6x-4)^(-2/3) * 6 = 2(6x-4)^(-2/3)
        # f'(2) = 2(12-4)^(-2/3) = 2(8)^(-2/3) = 2 / 8^(2/3) = 2 / 4 = 0.5
        problem = "Given f(x) = (6*x - 4)**(1/3), find f'(2)"
        result = self.solver.solve(problem)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_natural_language_quotient(self):
        # Verify it handles the LaTeX-like style from MATH
        problem = "derivative of (3*x - 4)/(2*x + 3) at x=1"
        result = self.solver.solve(problem)
        self.assertAlmostEqual(result, 0.68, places=5)

if __name__ == '__main__':
    unittest.main()
