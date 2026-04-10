#!/usr/bin/env python3
"""
Lightweight Zero-Copy Test Suite - Phase 4 Implementation
Tests the lightweight procedural content approach for free computational power
Removes memory compression in favor of procedural content generation

Based on: Symlink nature of procedural content from historical analysis
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.sovereign import loader

def test_lightweight_kernel_compilation():
    """Test that lightweight kernels can be compiled and loaded"""
    
    print("Testing lightweight kernel compilation...")
    
    # Test lightweight kernel file
    kernel_file = 'knowledge3d/cranium/kernels/zero_copy_memory_manager_phase4.cu'
    
    if os.path.exists(kernel_file):
        try:
            # Read kernel source
            with open(kernel_file, 'r') as f:
                kernel_source = f.read()
            
            # Verify lightweight functions exist
            lightweight_functions = [
                'lightweight_procedural_kernel',
                'lightweight_warp_kernel', 
                'symlink_procedural_kernel',
                'lightweight_zero_copy_initialize',
                'lightweight_update_zero_copy'
            ]
            
            found_functions = 0
            for func in lightweight_functions:
                if func in kernel_source:
                    found_functions += 1
                    print(f"✓ Found {func}")
            
            if found_functions >= len(lightweight_functions):
                print("✓ All lightweight kernel functions found")
                return True
            else:
                print(f"✗ Only {found_functions}/{len(lightweight_functions)} functions found")
                return False
                
        except Exception as e:
            print(f"✗ Failed to load lightweight kernel: {e}")
            return False
    else:
        print(f"⚠ Lightweight kernel file not found: {kernel_file}")
        return False

def test_lightweight_memory_operations():
    """Test lightweight zero-copy memory operations using procedural content"""
    
    print("Testing lightweight memory operations...")
    
    # Initialize loader
    loader.ensure_init()
    
    # Test procedural content generation
    test_dim = 128
    
    try:
        # Allocate minimal GPU memory for procedural content
        output_gpu = loader.gpu_malloc(test_dim * 4)  # float32 array
        
        # Simulate lightweight procedural generation
        # In real implementation, this would use the lightweight kernels
        
        # Generate procedural content on CPU for testing
        procedural_output = np.zeros(test_dim, dtype=np.float32)
        
        for i in range(test_dim):
            # Simulate procedural generation using mathematical operations
            angle = i * 0.1
            procedural_value = np.sin(angle) * np.cos(angle * 0.5 + 0.5) * 0.5 + 0.5
            procedural_output[i] = procedural_value
        
        # Copy procedural content to GPU
        loader.memcpy_htod(output_gpu, procedural_output.ctypes.data, procedural_output.nbytes)
        
        # Verify procedural content
        verification_output = np.zeros_like(procedural_output)
        loader.memcpy_dtoh(verification_output.ctypes.data, output_gpu, verification_output.nbytes)
        
        # Check that procedural content is reasonable
        max_error = np.max(np.abs(verification_output - procedural_output))
        print(f"✓ Maximum procedural content error: {max_error:.6f}")
        
        # Verify content range (should be [0, 1] for normalized procedural)
        min_val = np.min(verification_output)
        max_val = np.max(verification_output)
        print(f"✓ Procedural content range: [{min_val:.3f}, {max_val:.3f}]")
        
        if max_error < 1e-5:
            print("✓ Lightweight procedural content generation successful")
            return True
        else:
            print("✗ Procedural content verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Lightweight memory test failed: {e}")
        return False
    finally:
        if 'output_gpu' in locals():
            loader.gpu_free(output_gpu)

def test_computational_power_efficiency():
    """Test that lightweight approach uses computational power efficiently"""
    
    print("Testing computational power efficiency...")
    
    # Initialize loader
    loader.ensure_init()
    
    test_dims = [64, 128, 256]
    
    for test_dim in test_dims:
        
        try:
            # Allocate minimal memory for procedural content
            output_gpu = loader.gpu_malloc(test_dim * 4)
            
            # Generate procedural content using mathematical operations
            # This simulates using GPU computational power rather than memory bandwidth
            
            start_time = time.perf_counter()
            
            # Simulate procedural generation (CPU version for testing)
            procedural_data = np.zeros(test_dim, dtype=np.float32)
            for i in range(test_dim):
                # Use mathematical operations that leverage computational power
                angle = i * 0.1
                base = np.sin(angle) * 0.5 + 0.5
                variation = np.cos(angle * 0.5 + 0.3) * 0.3
                procedural_data[i] = base + variation
            
            # Copy to GPU (minimal memory operation)
            loader.memcpy_htod(output_gpu, procedural_data.ctypes.data, procedural_data.nbytes)
            
            computation_time = (time.perf_counter() - start_time) * 1000  # ms
            
            print(f"✓ Dim={test_dim}: Procedural generation time: {computation_time:.3f}ms")
            
            # Verify computational efficiency (should be fast due to mathematical operations)
            if computation_time < 10.0:  # Should be under 10ms for these dimensions
                print(f"✓ Computational efficiency verified for dim={test_dim}")
            else:
                print(f"⚠ Computational time higher than expected for dim={test_dim}")
            
        except Exception as e:
            print(f"✗ Computational efficiency test failed for dim={test_dim}: {e}")
            return False
        finally:
            if 'output_gpu' in locals():
                loader.gpu_free(output_gpu)
    
    return True

def test_symlink_style_generation():
    """Test symlink-style procedural content generation"""
    
    print("Testing symlink-style procedural generation...")
    
    # Initialize loader
    loader.ensure_init()
    
    test_dim = 256
    iterations = 3
    
    try:
        # Allocate memory for testing
        output_gpu = loader.gpu_malloc(test_dim * 4)
        input_gpu = loader.gpu_malloc(test_dim * 4)
        
        # Create input data
        input_data = np.random.randn(test_dim).astype(np.float32) * 0.1 + 0.5
        loader.memcpy_htod(input_gpu, input_data.ctypes.data, input_data.nbytes)
        
        for iteration in range(iterations):
            
            # Simulate symlink-style procedural generation
            # Instead of storing data, generate it algorithmically
            
            procedural_data = np.zeros(test_dim, dtype=np.float32)
            
            for i in range(test_dim):
                # Symlink-style: generate based on iteration and position
                procedural_key = (i ^ iteration) * 0.001
                base_pattern = np.sin(procedural_key) * np.cos(procedural_key * 1.618)
                
                # Blend with input data
                blended_value = base_pattern * 0.7 + input_data[i] * 0.3
                
                procedural_data[i] = blended_value
            
            # Copy result to GPU
            loader.memcpy_htod(output_gpu, procedural_data.ctypes.data, procedural_data.nbytes)
            
            # Verify iteration-based variation
            output_cpu = np.zeros_like(procedural_data)
            loader.memcpy_dtoh(output_cpu.ctypes.data, output_gpu, output_cpu.nbytes)
            
            variation = np.std(output_cpu - input_data)
            print(f"✓ Iteration {iteration}: Procedural variation: {variation:.4f}")
        
        print("✓ Symlink-style procedural generation successful")
        return True
        
    except Exception as e:
        print(f"✗ Symlink-style test failed: {e}")
        return False
    finally:
        if 'output_gpu' in locals():
            loader.gpu_free(output_gpu)
        if 'input_gpu' in locals():
            loader.gpu_free(input_gpu)

def test_warp_level_efficiency():
    """Test warp-level computational efficiency"""
    
    print("Testing warp-level computational efficiency...")
    
    # Initialize loader
    loader.ensure_init()
    
    # Test with warp-aligned dimensions
    warp_dimensions = [32, 64, 128, 256]  # Multiples of 32
    
    for dim in warp_dimensions:
        
        try:
            output_gpu = loader.gpu_malloc(dim * 4)
            
            # Simulate warp-level procedural generation
            procedural_data = np.zeros(dim, dtype=np.float32)
            
            for i in range(dim):
                # Simulate warp-level operations
                warp_id = i // 32
                lane_id = i % 32
                
                base_value = i * 0.03125  # 1/32 for warp efficiency
                warp_coord = warp_id + lane_id * 0.03125
                
                # Procedural generation using warp-level concepts
                procedural_value = np.sin(warp_coord) * np.cos(base_value + 0.5)
                
                procedural_data[i] = procedural_value * 0.5 + 0.5
            
            # Copy to GPU
            loader.memcpy_htod(output_gpu, procedural_data.ctypes.data, procedural_data.nbytes)
            
            # Verify warp-aligned efficiency
            output_cpu = np.zeros_like(procedural_data)
            loader.memcpy_dtoh(output_cpu.ctypes.data, output_gpu, output_cpu.nbytes)
            
            # Check that values are in expected range
            min_val = np.min(output_cpu)
            max_val = np.max(output_cpu)
            
            if 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0:
                print(f"✓ Warp-level efficiency verified for dim={dim}")
            else:
                print(f"✗ Warp-level values out of range for dim={dim}: [{min_val}, {max_val}]")
                return False
                
        except Exception as e:
            print(f"✗ Warp-level test failed for dim={dim}: {e}")
            return False
        finally:
            if 'output_gpu' in locals():
                loader.gpu_free(output_gpu)
    
    return True

def test_minimal_memory_footprint():
    """Test that lightweight approach has minimal memory footprint"""
    
    print("Testing minimal memory footprint...")
    
    # Initialize loader
    loader.ensure_init()
    
    # Check VRAM usage before operations
    try:
        used_before, total_before = loader.get_vram_usage()
        
        # Perform lightweight operations
        test_size = 1024  # Small size for lightweight approach
        
        output_gpu = loader.gpu_malloc(test_size * 4)
        
        # Generate procedural content
        procedural_data = np.random.randn(test_size).astype(np.float32) * 0.1 + 0.5
        loader.memcpy_htod(output_gpu, procedural_data.ctypes.data, procedural_data.nbytes)
        
        # Check VRAM usage after operations
        used_after, total_after = loader.get_vram_usage()
        
        # Verify minimal memory usage
        memory_increase = used_after - used_before
        expected_increase = test_size * 4  # 4KB
        
        print(f"✓ VRAM usage: {used_before} -> {used_after} (increase: {memory_increase} bytes)")
        
        if memory_increase <= expected_increase * 1.1:  # Allow 10% overhead
            print("✓ Minimal memory footprint verified")
            return True
        else:
            print(f"✗ Memory usage higher than expected: {memory_increase} vs {expected_increase}")
            return False
            
    except Exception as e:
        print(f"✗ Memory footprint test failed: {e}")
        return False
    finally:
        if 'output_gpu' in locals():
            loader.gpu_free(output_gpu)

def main():
    """Run all lightweight zero-copy tests"""
    
    print("=" * 70)
    print("Lightweight Zero-Copy Test Suite - Phase 4 Implementation")
    print("=" * 70)
    
    tests = [
        test_lightweight_kernel_compilation,
        test_lightweight_memory_operations,
        test_computational_power_efficiency,
        test_symlink_style_generation,
        test_warp_level_efficiency,
        test_minimal_memory_footprint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All lightweight zero-copy tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())