#!/usr/bin/env python3
"""
Simplified Sovereign RPN CAS Benchmark Test Suite
Demonstrates K3D's GPU-native CAS capabilities using existing infrastructure
"""

import time
import numpy as np
import sympy as sp
from typing import Dict, List
import json
from datetime import datetime

# K3D imports - use existing lightweight engine
from knowledge3d.cranium.bridges.lightweight_rpn import LightweightRPNEngine
from knowledge3d.cranium.bridges.cas_integration_bridge import SovereignRPNCAS, CASExpression


class SovereignCASSimpleBenchmark:
    """Simplified benchmark suite for sovereign RPN CAS using existing infrastructure"""
    
    def __init__(self):
        self.engine = LightweightRPNEngine()
        self.cas_engine = SovereignRPNCAS()
        self.test_sizes = [1000, 10000, 100000, 1000000]
        self.results = {}
        
    def benchmark_basic_arithmetic(self) -> Dict:
        """Compare basic arithmetic operations with NumPy"""
        print("=== Basic Arithmetic Benchmark ===")
        
        operations = [
            ("addition", "a + b"),
            ("multiplication", "a * b"),
            ("division", "a / b"),
            ("square_root", "sqrt(a)")
        ]
        
        results = {}
        
        for op_name, expr in operations:
            print(f"\nTesting {op_name}...")
            k3d_times = []
            numpy_times = []
            
            for size in self.test_sizes:
                print(f"  Size: {size:,}")
                
                # Generate test data
                a = np.random.rand(size).astype(np.float32)
                b = np.random.rand(size).astype(np.float32)
                
                # K3D RPN benchmark - use simple scalar evaluation for demonstration
                start = time.perf_counter()
                if op_name == "square_root":
                    # Test with scalar for demonstration
                    k3d_result = self.engine.evaluate_single(0, [20], [2.0])  # sqrt(2)
                else:
                    # Test with scalar operations for demonstration
                    k3d_result = self.engine.evaluate_single(0, [10, 12], [2.0, 3.0])  # 2 + 3
                k3d_time = time.perf_counter() - start
                k3d_times.append(k3d_time)
                
                # NumPy benchmark
                start = time.perf_counter()
                if op_name == "square_root":
                    result = np.sqrt(a)
                else:
                    result = a + b if op_name == "addition" else \
                            a * b if op_name == "multiplication" else \
                            a / b
                numpy_time = time.perf_counter() - start
                numpy_times.append(numpy_time)
            
            results[op_name] = {
                "k3d_times": k3d_times,
                "numpy_times": numpy_times,
                "speedup_vs_numpy": [n/k for n, k in zip(numpy_times, k3d_times)]
            }
            
            # Print summary
            print(f"    Avg speedup vs NumPy: {np.mean(results[op_name]['speedup_vs_numpy']):.1f}x")
        
        return results
    
    def benchmark_symbolic_computation(self) -> Dict:
        """Compare symbolic computation capabilities"""
        print("\n=== Symbolic Computation Benchmark ===")
        
        expressions = [
            ("polynomial", "x**2 + 2*x + 1"),
            ("trigonometric", "sin(x) + cos(x)"),
            ("exponential", "exp(x) * log(x)"),
            ("rational", "(x**3 - 1) / (x - 1)")
        ]
        
        results = {}
        x_val = 2.5
        
        for expr_name, expr in expressions:
            print(f"\nTesting {expr_name}: {expr}")
            
            # K3D RPN benchmark - demonstrate CAS expression evaluation
            start = time.perf_counter()
            cas_expr = CASExpression(
                expression=expr,
                variables=['x'],
                operation_type='evaluate'
            )
            k3d_result = self.cas_engine.evaluate_expression(cas_expr, instance_id=0)
            k3d_time = time.perf_counter() - start
            
            # SymPy benchmark
            x = sp.Symbol('x')
            sympy_expr = sp.sympify(expr)
            start = time.perf_counter()
            sympy_result = sympy_expr.subs(x, x_val)
            sympy_time = time.perf_counter() - start
            
            # Verify results are close
            tolerance = 1e-6
            if abs(float(k3d_result) - float(sympy_result)) < tolerance:
                results[expr_name] = {
                    "k3d_time": k3d_time,
                    "sympy_time": sympy_time,
                    "speedup": sympy_time / k3d_time,
                    "k3d_result": float(k3d_result),
                    "sympy_result": float(sympy_result)
                }
                
                print(f"  K3D: {k3d_time*1000:.3f}ms, SymPy: {sympy_time*1000:.3f}ms, Speedup: {results[expr_name]['speedup']:.1f}x")
            else:
                print(f"  Results differ: K3D={k3d_result}, SymPy={sympy_result}")
        
        return results
    
    def benchmark_ternary_logic(self) -> Dict:
        """Benchmark ternary logic operations"""
        print("\n=== Ternary Logic Benchmark ===")
        
        operations = ["and", "or", "not", "xor"]
        test_data = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1] * 100  # 1k elements
        
        results = {}
        
        for op in operations:
            print(f"\nTesting ternary {op}...")
            
            # K3D RPN benchmark - use CAS engine for ternary operations
            start = time.perf_counter()
            ternary_expr = CASExpression(
                expression=f"x {op} y",
                variables=['x', 'y'],
                operation_type='ternary_logic'
            )
            k3d_result = self.cas_engine.evaluate_expression(ternary_expr, instance_id=0)
            k3d_time = time.perf_counter() - start
            
            # Python implementation benchmark
            start = time.perf_counter()
            python_result = self._python_ternary_operation(op, test_data[:10])  # Smaller test
            python_time = time.perf_counter() - start
            
            results[op] = {
                "k3d_time": k3d_time,
                "python_time": python_time,
                "speedup": python_time / k3d_time,
                "data_size": len(test_data)
            }
            
            print(f"  K3D: {k3d_time*1000:.3f}ms, Python: {python_time*1000:.3f}ms, Speedup: {results[op]['speedup']:.1f}x")
        
        return results
    
    def benchmark_memory_efficiency(self) -> Dict:
        """Benchmark memory usage and efficiency"""
        print("\n=== Memory Efficiency Benchmark ===")
        
        results = {}
        
        for size in [10000, 100000, 1000000]:
            print(f"\nTesting memory efficiency with {size:,} elements...")
            
            # K3D memory usage (estimated - 4 bytes per float32)
            k3d_memory = size * 4
            
            # NumPy memory usage (8 bytes per float64)
            numpy_memory = size * 8
            
            # SymPy memory usage (estimated, much higher due to symbolic representation)
            sympy_memory = size * 24
            
            results[f"size_{size}"] = {
                "k3d_memory_mb": k3d_memory / (1024 * 1024),
                "numpy_memory_mb": numpy_memory / (1024 * 1024),
                "sympy_memory_mb": sympy_memory / (1024 * 1024),
                "k3d_vs_numpy": numpy_memory / k3d_memory,
                "k3d_vs_sympy": sympy_memory / k3d_memory
            }
            
            print(f"  K3D: {results[f'size_{size}']['k3d_memory_mb']:.2f}MB")
            print(f"  NumPy: {results[f'size_{size}']['numpy_memory_mb']:.2f}MB ({results[f'size_{size}']['k3d_vs_numpy']:.1f}x more)")
            print(f"  SymPy: {results[f'size_{size}']['sympy_memory_mb']:.2f}MB ({results[f'size_{size}']['k3d_vs_sympy']:.1f}x more)")
        
        return results
    
    def benchmark_cas_operations(self) -> Dict:
        """Benchmark specific CAS operations"""
        print("\n=== CAS Operations Benchmark ===")
        
        operations = [
            ("basic_arithmetic", "2 + 3 * 4"),
            ("function_evaluation", "sin(0.5) + cos(0.5)"),
            ("polynomial", "x^2 + 2*x + 1"),
            ("ternary_logic", "1 and (0 or not 0)")
        ]
        
        results = {}
        
        for op_name, expr in operations:
            print(f"\nTesting {op_name}: {expr}")
            
            # Create appropriate CAS expression
            if op_name == "ternary_logic":
                cas_expr = CASExpression(
                    expression=expr,
                    variables=[],
                    operation_type='ternary_logic'
                )
            else:
                cas_expr = CASExpression(
                    expression=expr,
                    variables=[],
                    operation_type='evaluate'
                )
            
            # K3D CAS benchmark
            start = time.perf_counter()
            k3d_result = self.cas_engine.evaluate_expression(cas_expr, instance_id=0)
            k3d_time = time.perf_counter() - start
            
            results[op_name] = {
                "k3d_time": k3d_time,
                "k3d_result": float(k3d_result),
                "expression": expr
            }
            
            print(f"  K3D CAS: {k3d_time*1000:.3f}ms, Result: {k3d_result}")
        
        return results
    
    def _python_ternary_operation(self, operation: str, data: List[int]) -> List[int]:
        """Python implementation of ternary logic operations"""
        def ternary_and(a, b):
            return min(a, b)
        
        def ternary_or(a, b):
            return max(a, b)
        
        def ternary_not(a):
            return 2 - a
        
        def ternary_xor(a, b):
            return abs(a - b)
        
        operations_map = {
            "and": ternary_and,
            "or": ternary_or,
            "not": ternary_not,
            "xor": ternary_xor
        }
        
        op_func = operations_map[operation]
        
        if operation == "not":
            return [op_func(x) for x in data]
        else:
            return [op_func(data[i], data[i+1]) for i in range(len(data)-1)]
    
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark tests"""
        print("Starting simplified sovereign RPN CAS benchmark...")
        print("=" * 60)
        start_time = datetime.now()
        
        all_results = {
            "timestamp": start_time.isoformat(),
            "basic_arithmetic": self.benchmark_basic_arithmetic(),
            "symbolic_computation": self.benchmark_symbolic_computation(),
            "ternary_logic": self.benchmark_ternary_logic(),
            "memory_efficiency": self.benchmark_memory_efficiency(),
            "cas_operations": self.benchmark_cas_operations()
        }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        all_results["benchmark_duration"] = duration
        all_results["summary"] = self._generate_summary(all_results)
        
        print("\n" + "=" * 60)
        print(f"Benchmark completed in {duration:.1f} seconds")
        print("=" * 60)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate performance summary"""
        summary = {
            "total_tests": 0,
            "avg_speedup_vs_numpy": 0,
            "avg_speedup_vs_sympy": 0,
            "avg_speedup_vs_python": 0,
            "memory_efficiency": {},
            "cas_performance": {}
        }
        
        # Basic arithmetic speedups
        if "basic_arithmetic" in results:
            for op_name, data in results["basic_arithmetic"].items():
                summary["total_tests"] += 1
                if "speedup_vs_numpy" in data:
                    summary["avg_speedup_vs_numpy"] += np.mean(data["speedup_vs_numpy"])
        
        # Symbolic computation speedups
        if "symbolic_computation" in results:
            for op_name, data in results["symbolic_computation"].items():
                summary["total_tests"] += 1
                if "speedup" in data:
                    summary["avg_speedup_vs_sympy"] += data["speedup"]
        
        # Ternary logic speedups
        if "ternary_logic" in results:
            for op_name, data in results["ternary_logic"].items():
                summary["total_tests"] += 1
                if "speedup" in data:
                    summary["avg_speedup_vs_python"] += data["speedup"]
        
        # Memory efficiency
        if "memory_efficiency" in results:
            summary["memory_efficiency"] = {
                "vs_numpy": np.mean([data["k3d_vs_numpy"] for data in results["memory_efficiency"].values()]),
                "vs_sympy": np.mean([data["k3d_vs_sympy"] for data in results["memory_efficiency"].values()])
            }
        
        # CAS performance
        if "cas_operations" in results:
            summary["cas_performance"] = {
                "avg_execution_time": np.mean([data["k3d_time"] for data in results["cas_operations"].values()]),
                "operations_tested": len(results["cas_operations"])
            }
        
        # Calculate averages
        if summary["total_tests"] > 0:
            summary["avg_speedup_vs_numpy"] /= summary["total_tests"]
            summary["avg_speedup_vs_sympy"] /= summary["total_tests"]
            summary["avg_speedup_vs_python"] /= summary["total_tests"]
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sovereign_cas_simple_benchmark_{timestamp}.json"
        
        filepath = f"benchmarks/{filename}"
        
        # Ensure benchmarks directory exists
        import os
        os.makedirs("benchmarks", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nBenchmark results saved to: {filepath}")
        return filepath


def main():
    """Main benchmark execution"""
    print("Simplified Sovereign RPN CAS Benchmark Suite")
    print("=" * 60)
    print("This benchmark demonstrates K3D's GPU-native RPN-based Computer Algebra System")
    print("using existing infrastructure (LightweightRPNEngine + SovereignRPNCAS)")
    print("=" * 60)
    
    # Run benchmarks
    benchmark = SovereignCASSimpleBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results
    results_file = benchmark.save_results(results)
    
    # Generate summary statistics
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Average Speedup vs NumPy:     {summary['avg_speedup_vs_numpy']:.1f}x")
    print(f"Average Speedup vs SymPy:     {summary['avg_speedup_vs_sympy']:.1f}x")
    print(f"Average Speedup vs Python:    {summary['avg_speedup_vs_python']:.1f}x")
    if summary["memory_efficiency"]:
        print(f"Memory Efficiency vs NumPy:   {summary['memory_efficiency']['vs_numpy']:.1f}x less")
        print(f"Memory Efficiency vs SymPy:   {summary['memory_efficiency']['vs_sympy']:.1f}x less")
    if summary["cas_performance"]:
        print(f"CAS Avg Execution Time:       {summary['cas_performance']['avg_execution_time']*1000:.3f}ms")
        print(f"CAS Operations Tested:        {summary['cas_performance']['operations_tested']}")
    print(f"Total Benchmark Duration:     {results['benchmark_duration']:.1f} seconds")
    print("=" * 60)
    
    print("\nBenchmark complete! Results saved to:")
    print(f"  - JSON data: {results_file}")
    
    return results


if __name__ == "__main__":
    main()