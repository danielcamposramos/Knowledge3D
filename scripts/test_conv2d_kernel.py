#!/usr/bin/env python3
"""Test and benchmark Phase F.1 conv2d_3x3 kernel.

Validates:
1. Compilation and loading
2. Correctness (99.9% bit-match with NumPy)
3. Performance (<0.5ms target)

Usage:
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_conv2d_kernel.py
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.ocr.conv2d_bridge import Conv2dBridge, conv2d_3x3_numpy


def test_compilation():
    """Test 1: Kernel compilation and loading."""
    print("=" * 80)
    print("TEST 1: Kernel Compilation and Loading")
    print("=" * 80)
    print()

    try:
        print("Compiling conv2d_3x3.cu...")
        bridge = Conv2dBridge()
        print("✓ Kernel compiled and loaded successfully")
        print()
        return bridge
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        print()
        return None


def test_correctness(bridge: Conv2dBridge):
    """Test 2: Correctness validation against NumPy reference."""
    print("=" * 80)
    print("TEST 2: Correctness Validation")
    print("=" * 80)
    print()

    # Test configuration
    test_cases = [
        {"name": "Small (32×32, 16→32)", "H": 32, "W": 32, "Cin": 16, "Cout": 32},
        {"name": "Medium (64×64, 32→64)", "H": 64, "W": 64, "Cin": 32, "Cout": 64},
        {"name": "OCR typical (128×128, 64→128)", "H": 128, "W": 128, "Cin": 64, "Cout": 128},
    ]

    all_passed = True

    for test in test_cases:
        print(f"Testing: {test['name']}")
        H, W, Cin, Cout = test['H'], test['W'], test['Cin'], test['Cout']

        # Generate random inputs
        np.random.seed(42)
        input = np.random.randn(H, W, Cin).astype(np.float32) * 0.1
        weight = np.random.randn(Cout, 3, 3, Cin).astype(np.float32) * 0.1
        bias = np.random.randn(Cout).astype(np.float32) * 0.1

        # GPU forward
        try:
            output_gpu = bridge.forward(input, weight, bias, relu=True)
        except Exception as e:
            print(f"  ❌ GPU forward failed: {e}")
            all_passed = False
            continue

        # NumPy reference
        output_numpy = conv2d_3x3_numpy(input, weight, bias, relu=True)

        # Compare
        abs_diff = np.abs(output_gpu - output_numpy)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        rel_diff = max_diff / (np.abs(output_numpy).mean() + 1e-8)

        # Calculate match rate (99.9% target)
        tolerance = 1e-5
        match_rate = np.mean(abs_diff < tolerance)

        print(f"  Max abs diff:  {max_diff:.6e}")
        print(f"  Mean abs diff: {mean_diff:.6e}")
        print(f"  Relative diff: {rel_diff:.6e}")
        print(f"  Match rate:    {match_rate*100:.2f}% (target: 99.9%)")

        if match_rate >= 0.999:
            print(f"  ✓ PASSED")
        else:
            print(f"  ❌ FAILED (match rate below 99.9%)")
            all_passed = False

        print()

    if all_passed:
        print("✓ All correctness tests PASSED")
    else:
        print("❌ Some correctness tests FAILED")

    print()
    return all_passed


def test_performance(bridge: Conv2dBridge):
    """Test 3: Performance benchmark (<0.5ms target)."""
    print("=" * 80)
    print("TEST 3: Performance Benchmark")
    print("=" * 80)
    print()

    # Benchmark configuration
    benchmarks = [
        {"name": "Small (32×32, 16→32)", "H": 32, "W": 32, "Cin": 16, "Cout": 32},
        {"name": "Medium (64×64, 32→64)", "H": 64, "W": 64, "Cin": 32, "Cout": 64},
        {"name": "OCR typical (128×128, 64→128)", "H": 128, "W": 128, "Cin": 64, "Cout": 128},
        {"name": "Large (256×256, 64→128)", "H": 256, "W": 256, "Cin": 64, "Cout": 128},
    ]

    all_passed = True

    for bench in benchmarks:
        print(f"Benchmarking: {bench['name']}")
        H, W, Cin, Cout = bench['H'], bench['W'], bench['Cin'], bench['Cout']

        # Generate random inputs
        np.random.seed(42)
        input = np.random.randn(H, W, Cin).astype(np.float32) * 0.1
        weight = np.random.randn(Cout, 3, 3, Cin).astype(np.float32) * 0.1
        bias = np.random.randn(Cout).astype(np.float32) * 0.1

        # Warmup
        for _ in range(3):
            _ = bridge.forward(input, weight, bias, relu=True)

        # Benchmark
        n_runs = 100
        times = []

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = bridge.forward(input, weight, bias, relu=True)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        p50_time = np.percentile(times, 50)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)

        # Calculate throughput
        pixels = H * W
        throughput = pixels / (mean_time / 1000) / 1e6  # Mpixels/sec

        print(f"  Mean:   {mean_time:.3f} ms ± {std_time:.3f} ms")
        print(f"  Min:    {min_time:.3f} ms")
        print(f"  P50:    {p50_time:.3f} ms")
        print(f"  P95:    {p95_time:.3f} ms")
        print(f"  P99:    {p99_time:.3f} ms")
        print(f"  Max:    {max_time:.3f} ms")
        print(f"  Throughput: {throughput:.1f} Mpixels/sec")

        # Check against target (<0.5ms for typical OCR sizes)
        if H <= 128 and W <= 128:
            target_ms = 0.5
            if mean_time < target_ms:
                print(f"  ✓ PASSED (< {target_ms} ms target)")
            else:
                print(f"  ⚠ SLOW (> {target_ms} ms target)")
                all_passed = False
        else:
            print(f"  ✓ Measured (no target for this size)")

        print()

    if all_passed:
        print("✓ All performance tests PASSED")
    else:
        print("⚠ Some performance tests exceeded target")

    print()
    return all_passed


def test_edge_cases(bridge: Conv2dBridge):
    """Test 4: Edge cases and error handling."""
    print("=" * 80)
    print("TEST 4: Edge Cases")
    print("=" * 80)
    print()

    all_passed = True

    # Test 4.1: No ReLU
    print("Test 4.1: No ReLU activation")
    H, W, Cin, Cout = 32, 32, 16, 32
    np.random.seed(42)
    input = np.random.randn(H, W, Cin).astype(np.float32) * 0.1
    weight = np.random.randn(Cout, 3, 3, Cin).astype(np.float32) * 0.1
    bias = np.random.randn(Cout).astype(np.float32) * 0.1

    try:
        output_gpu = bridge.forward(input, weight, bias, relu=False)
        output_numpy = conv2d_3x3_numpy(input, weight, bias, relu=False)

        # Check for negative values (ReLU should be disabled)
        has_negative = np.any(output_gpu < 0)
        match_rate = np.mean(np.abs(output_gpu - output_numpy) < 1e-5)

        if has_negative and match_rate >= 0.999:
            print("  ✓ PASSED (negative values present, no ReLU applied)")
        else:
            print(f"  ❌ FAILED (has_negative={has_negative}, match={match_rate*100:.2f}%)")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        all_passed = False

    print()

    # Test 4.2: Single channel
    print("Test 4.2: Single input channel")
    H, W, Cin, Cout = 32, 32, 1, 16
    input = np.random.randn(H, W, Cin).astype(np.float32) * 0.1
    weight = np.random.randn(Cout, 3, 3, Cin).astype(np.float32) * 0.1
    bias = np.random.randn(Cout).astype(np.float32) * 0.1

    try:
        output_gpu = bridge.forward(input, weight, bias, relu=True)
        output_numpy = conv2d_3x3_numpy(input, weight, bias, relu=True)
        match_rate = np.mean(np.abs(output_gpu - output_numpy) < 1e-5)

        if match_rate >= 0.999:
            print("  ✓ PASSED")
        else:
            print(f"  ❌ FAILED (match={match_rate*100:.2f}%)")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        all_passed = False

    print()

    # Test 4.3: Many channels (>64, tests chunking)
    print("Test 4.3: Many input channels (Cin > CIN_CHUNK)")
    H, W, Cin, Cout = 32, 32, 128, 64  # Cin=128 > CIN_CHUNK=64
    input = np.random.randn(H, W, Cin).astype(np.float32) * 0.1
    weight = np.random.randn(Cout, 3, 3, Cin).astype(np.float32) * 0.1
    bias = np.random.randn(Cout).astype(np.float32) * 0.1

    try:
        output_gpu = bridge.forward(input, weight, bias, relu=True)
        output_numpy = conv2d_3x3_numpy(input, weight, bias, relu=True)
        match_rate = np.mean(np.abs(output_gpu - output_numpy) < 1e-5)

        if match_rate >= 0.999:
            print("  ✓ PASSED (chunking works correctly)")
        else:
            print(f"  ❌ FAILED (match={match_rate*100:.2f}%)")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        all_passed = False

    print()

    if all_passed:
        print("✓ All edge case tests PASSED")
    else:
        print("❌ Some edge case tests FAILED")

    print()
    return all_passed


def main():
    print()
    print("=" * 80)
    print("Phase F.1: conv2d_3x3 Kernel Test Suite")
    print("=" * 80)
    print()
    print("Foundation: Kimi v1 (16×16 tiling + 2-pixel halo)")
    print("Enhancements: Grok (generalized Cin chunks)")
    print("Target: sm_75 (RTX 3060), <0.5ms, 99.9% accuracy")
    print()

    # Run tests
    bridge = test_compilation()
    if bridge is None:
        print("❌ Cannot proceed without successful compilation")
        return False

    results = []

    results.append(("Correctness", test_correctness(bridge)))
    results.append(("Performance", test_performance(bridge)))
    results.append(("Edge Cases", test_edge_cases(bridge)))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20s}: {status}")

    print()

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("=" * 80)
        print("✓ ALL TESTS PASSED - Phase F.1 Foundation Complete!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Integrate with DeepSeek OCR pipeline")
        print("  2. Add Kimi v2 enhancements (warp-cross stacks, micro-TRM)")
        print("  3. Benchmark on real OCR workloads")
        print()
        return True
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED - Review errors above")
        print("=" * 80)
        print()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
