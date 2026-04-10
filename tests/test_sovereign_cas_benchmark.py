#!/usr/bin/env python3
"""
Sovereign RPN CAS Benchmark Tests
Real execution of CAS operations using K3D's working GPU infrastructure
"""

import time
import numpy as np
import json
from datetime import datetime
import ctypes
from pathlib import Path

# Use K3D's working GPU infrastructure
from knowledge3d.cranium.sovereign import loader


class SovereignCASBenchmark:
    """Real benchmark tests using K3D's functional GPU infrastructure"""
    
    def __init__(self):
        self.working_kernel = None
        self.working_device_state = None
        
    def find_working_rpn_kernel(self) -> bool:
        """Find and load a working RPN kernel"""
        print("=== Finding Working RPN Kernel ===")
        
        # List of kernels to try
        kernel_candidates = [
            ("modular_rpn_kernel.ptx", "modular_rpn_geometric_kernel"),
            ("modular_rpn_kernel_lite.ptx", "modular_rpn_kernel_lite"),
            ("modular_rpn_kernel_extended.ptx", "modular_rpn_kernel_extended"),
            ("rpn_executor.ptx", "rpn_executor_kernel"),
        ]
        
        ptx_dir = Path(__file__).parent.parent / "knowledge3d" / "cranium" / "ptx"
        
        for ptx_file, entry_name in kernel_candidates:
            ptx_path = ptx_dir / ptx_file
            if not ptx_path.exists():
                print(f"  {ptx_file}: NOT FOUND")
                continue
                
            try:
                print(f"  Trying {ptx_file} with entry {entry_name}...")
                kernel = loader.load_ptx_file(str(ptx_path), entry_name)
                
                # Allocate minimal device state
                device_state = loader.gpu_malloc(1024)  # Small allocation for testing
                
                print(f"  ✓ Successfully loaded {ptx_file}")
                self.working_kernel = kernel
                self.working_device_state = device_state
                return True
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:100]}...")
                continue
        
        print("  No working kernels found!")
        return False
    
    def benchmark_kernel_loading(self) -> Dict:
        """Benchmark kernel loading time"""
        print("\n=== Kernel Loading Benchmark ===")
        
        results = {}
        
        if not self.working_kernel:
            print("  No working kernel available")
            return results
        
        # Test kernel loading time
        start = time.perf_counter()
        try:
            # Reload kernel to measure loading time
            ptx_dir = Path(__file__).parent.parent / "knowledge3d" / "cranium" / "ptx"
            ptx_path = ptx_dir / "modular_rpn_kernel.ptx"
            test_kernel = loader.load_ptx_file(str(ptx_path), "modular_rpn_geometric_kernel")
            load_time = time.perf_counter() - start
            
            results["kernel_loading"] = {
                "load_time_ms": load_time * 1000,
                "kernel_name": "modular_rpn_geometric_kernel",
                "status": "success"
            }
            
            print(f"  Kernel loading: {load_time*1000:.3f}ms")
            
        except Exception as e:
            results["kernel_loading"] = {
                "load_time_ms": -1,
                "kernel_name": "modular_rpn_geometric_kernel", 
                "status": "failed",
                "error": str(e)
            }
            print(f"  Kernel loading: FAILED - {e}")
        
        return results
    
    def benchmark_memory_operations(self) -> Dict:
        """Benchmark GPU memory operations"""
        print("\n=== GPU Memory Operations Benchmark ===")
        
        results = {}
        
        try:
            # Test GPU memory allocation
            start = time.perf_counter()
            test_memory = loader.gpu_malloc(1024)  # 1KB test allocation
            alloc_time = time.perf_counter() - start
            
            results["gpu_memory_alloc"] = {
                "alloc_time_ms": alloc_time * 1000,
                "size_bytes": 1024,
                "status": "success"
            }
            
            print(f"  GPU memory allocation (1KB): {alloc_time*1000:.3f}ms")
            
            # Test memory copy (host to device)
            test_data = (ctypes.c_float * 256)(*range(256))  # 256 floats
            start = time.perf_counter()
            loader.memcpy_htod(test_memory, ctypes.cast(test_data, ctypes.c_void_p), ctypes.sizeof(test_data))
            h2d_time = time.perf_counter() - start
            
            results["gpu_memcpy_h2d"] = {
                "memcpy_time_ms": h2d_time * 1000,
                "size_bytes": ctypes.sizeof(test_data),
                "status": "success"
            }
            
            print(f"  Host to device memcpy ({ctypes.sizeof(test_data)} bytes): {h2d_time*1000:.3f}ms")
            
            # Test memory copy (device to host)
            result_data = (ctypes.c_float * 256)()
            start = time.perf_counter()
            loader.memcpy_dtoh(ctypes.cast(result_data, ctypes.c_void_p), test_memory, ctypes.sizeof(result_data))
            d2h_time = time.perf_counter() - start
            
            results["gpu_memcpy_d2h"] = {
                "memcpy_time_ms": d2h_time * 1000,
                "size_bytes": ctypes.sizeof(result_data),
                "status": "success"
            }
            
            print(f"  Device to host memcpy ({ctypes.sizeof(result_data)} bytes): {d2h_time*1000:.3f}ms")
            
            # Cleanup
            loader.gpu_free(test_memory)
            
        except Exception as e:
            print(f"  GPU memory operations FAILED: {e}")
            results["gpu_memory_error"] = {"error": str(e)}
        
        return results
    
    def benchmark_rpn_operations(self) -> Dict:
        """Benchmark actual RPN operations using working kernel"""
        print("\n=== RPN Operations Benchmark ===")
        
        results = {}
        
        # Basic arithmetic opcodes that work with K3D's RPN kernels
        operations = [
            ("addition", [10, 11], [5.0, 3.0]),      # 5 + 3 = 8
            ("subtraction", [11, 13], [10.0, 4.0]),  # 10 - 4 = 6  
            ("multiplication", [12], [7.0]),         # result * 7
            ("division", [13], [2.0]),               # result / 2
        ]
        
        for op_name, opcodes, scalars in operations:
            print(f"\n  Testing {op_name}: opcodes={opcodes}, scalars={scalars}")
            
            if not self.working_kernel or not self.working_device_state:
                print(f"    SKIPPED: No working kernel")
                continue
            
            execution_times = []
            iterations = 100  # More iterations for accurate timing
            
            for i in range(iterations):
                try:
                    # Prepare operation data
                    opcodes_array = (ctypes.c_uint16 * len(opcodes))(*opcodes)
                    scalars_array = (ctypes.c_float * len(scalars))(*scalars)
                    
                    # Allocate GPU memory for inputs
                    d_opcodes = loader.gpu_malloc(ctypes.sizeof(opcodes_array))
                    d_scalars = loader.gpu_malloc(ctypes.sizeof(scalars_array))
                    
                    # Copy data to GPU
                    loader.memcpy_htod(d_opcodes, ctypes.cast(opcodes_array, ctypes.c_void_p), ctypes.sizeof(opcodes_array))
                    loader.memcpy_htod(d_scalars, ctypes.cast(scalars_array, ctypes.c_void_p), ctypes.sizeof(scalars_array))
                    
                    # Execute kernel with correct parameters
                    start = time.perf_counter()
                    loader.launch(
                        self.working_kernel,
                        grid=(1, 1, 1),  # Single grid for testing
                        block=(1, 1, 1),  # Single block for testing
                        params=[
                            ctypes.c_uint32(0),  # instance_id
                            ctypes.c_uint64(d_opcodes.value),
                            ctypes.c_uint64(d_scalars.value),
                            ctypes.c_uint64(0),  # d_vectors (none for this test)
                            ctypes.c_uint64(self.working_device_state.value),
                            ctypes.c_uint32(len(opcodes)),
                        ],
                    )
                    loader.synchronize()
                    exec_time = time.perf_counter() - start
                    
                    execution_times.append(exec_time)
                    
                    # Cleanup
                    loader.gpu_free(d_opcodes)
                    loader.gpu_free(d_scalars)
                    
                except Exception as e:
                    print(f"    Error in iteration {i}: {e}")
                    execution_times.append(float('inf'))
            
            # Process results
            valid_times = [t for t in execution_times if t != float('inf')]
            
            if valid_times:
                results[op_name] = {
                    "opcodes": opcodes,
                    "scalars": scalars,
                    "avg_time_ms": np.mean(valid_times) * 1000,
                    "min_time_ms": np.min(valid_times) * 1000,
                    "max_time_ms": np.max(valid_times) * 1000,
                    "iterations": len(valid_times),
                    "success_rate": len(valid_times) / iterations
                }
                
                print(f"    Average time: {results[op_name]['avg_time_ms']:.3f}ms")
                print(f"    Success rate: {results[op_name]['success_rate']*100:.1f}%")
            else:
                print(f"    FAILED: No successful executions")
        
        return results
    
    def benchmark_system_info(self) -> Dict:
        """Collect system information for context"""
        print("\n=== System Information ===")
        
        results = {
            "cuda_available": False,
            "gpu_info": None,
            "cpu_info": None,
            "memory_info": None
        }
        
        try:
            # Check CUDA availability
            import pynvml
            pynvml.nvmlInit()
            results["cuda_available"] = True
            
            # Get GPU info
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                results["gpu_info"] = {
                    "name": gpu_name,
                    "total_memory_mb": gpu_memory.total / 1024 / 1024,
                    "driver_version": pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                }
                
                print(f"  GPU: {gpu_name}")
                print(f"  GPU Memory: {gpu_memory.total / 1024 / 1024:.0f}MB")
                print(f"  Driver: {results['gpu_info']['driver_version']}")
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            print("  pynvml not available - skipping GPU info")
        except Exception as e:
            print(f"  GPU info error: {e}")
        
        # Get CPU info
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            results["cpu_info"] = {
                "brand": cpu_info.get('brand_raw', 'Unknown'),
                "cores": cpu_info.get('count', 0),
                "hz": cpu_info.get('hz_advertised_friendly', 'Unknown')
            }
            
            print(f"  CPU: {results['cpu_info']['brand']}")
            print(f"  CPU Cores: {results['cpu_info']['cores']}")
            
        except ImportError:
            print("  cpuinfo not available - skipping CPU info")
        except Exception as e:
            print(f"  CPU info error: {e}")
        
        # Get memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            results["memory_info"] = {
                "total_gb": memory.total / 1024 / 1024 / 1024,
                "available_gb": memory.available / 1024 / 1024 / 1024,
                "percent": memory.percent
            }
            
            print(f"  System RAM: {results['memory_info']['total_gb']:.1f}GB")
            print(f"  Available RAM: {results['memory_info']['available_gb']:.1f}GB")
            
        except ImportError:
            print("  psutil not available - skipping memory info")
        except Exception as e:
            print(f"  Memory info error: {e}")
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark tests"""
        print("Starting REAL sovereign RPN CAS benchmark...")
        print("=" * 60)
        start_time = datetime.now()
        
        # Find working kernel first
        kernel_found = self.find_working_rpn_kernel()
        
        all_results = {
            "timestamp": start_time.isoformat(),
            "system_info": self.benchmark_system_info(),
            "kernel_loading": self.benchmark_kernel_loading(),
            "memory_operations": self.benchmark_memory_operations(),
            "working_status": kernel_found
        }
        
        if kernel_found:
            all_results["rpn_operations"] = self.benchmark_rpn_operations()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        all_results["benchmark_duration"] = duration
        all_results["summary"] = self._generate_summary(all_results)
        
        print("\n" + "=" * 60)
        print(f"REAL benchmark completed in {duration:.1f} seconds")
        print("=" * 60)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary of benchmark results"""
        summary = {
            "kernel_found": results["working_status"],
            "gpu_available": results["system_info"]["cuda_available"],
            "total_tests": 0,
            "successful_tests": 0,
            "avg_kernel_load_time_ms": 0,
            "avg_memory_alloc_time_ms": 0,
            "avg_rpn_op_time_ms": 0,
            "rpn_operations_success": False
        }
        
        # Kernel loading summary
        if "kernel_loading" in results and results["kernel_loading"].get("status") == "success":
            summary["avg_kernel_load_time_ms"] = results["kernel_loading"].get("load_time_ms", 0)
            summary["successful_tests"] += 1
        
        # Memory operations summary
        if "memory_operations" in results:
            memory_ops = results["memory_operations"]
            if "gpu_memory_alloc" in memory_ops and memory_ops["gpu_memory_alloc"].get("status") == "success":
                summary["avg_memory_alloc_time_ms"] = memory_ops["gpu_memory_alloc"].get("alloc_time_ms", 0)
                summary["successful_tests"] += 1
            
            if "gpu_memcpy_h2d" in memory_ops and memory_ops["gpu_memcpy_h2d"].get("status") == "success":
                summary["successful_tests"] += 1
            
            if "gpu_memcpy_d2h" in memory_ops and memory_ops["gpu_memcpy_d2h"].get("status") == "success":
                summary["successful_tests"] += 1
        
        # RPN operations summary
        if "rpn_operations" in results:
            rpn_ops = results["rpn_operations"]
            successful_ops = [op for op in rpn_ops.values() if op.get("success_rate", 0) > 0]
            if successful_ops:
                summary["rpn_operations_success"] = True
                summary["successful_tests"] += len(successful_ops)
                avg_time = np.mean([op["avg_time_ms"] for op in successful_ops])
                summary["avg_rpn_op_time_ms"] = avg_time
        
        summary["total_tests"] = 4  # kernel, memory alloc, memory copy x2
        if "rpn_operations" in results:
            summary["total_tests"] += len(results["rpn_operations"])
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sovereign_cas_benchmark_{timestamp}.json"
        
        filepath = f"benchmarks/{filename}"
        
        import os
        os.makedirs("benchmarks", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nBenchmark results saved to: {filepath}")
        return filepath


def main():
    """Main benchmark execution"""
    print("Sovereign RPN CAS Benchmark Suite")
    print("=" * 60)
    print("This benchmark runs ACTUAL K3D RPN operations on GPU")
    print("Real GPU execution with measurable performance results")
    print("=" * 60)
    
    # Run benchmarks
    benchmark = SovereignCASBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results
    results_file = benchmark.save_results(results)
    
    # Generate summary statistics
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Working kernel found: {summary['kernel_found']}")
    print(f"GPU available: {summary['gpu_available']}")
    print(f"Successful tests: {summary['successful_tests']}/{summary['total_tests']}")
    
    if summary['avg_kernel_load_time_ms'] > 0:
        print(f"Average kernel load time: {summary['avg_kernel_load_time_ms']:.3f}ms")
    if summary['avg_memory_alloc_time_ms'] > 0:
        print(f"Average memory allocation time: {summary['avg_memory_alloc_time_ms']:.3f}ms")
    if summary['avg_rpn_op_time_ms'] > 0:
        print(f"Average RPN operation time: {summary['avg_rpn_op_time_ms']:.3f}ms")
    if summary['rpn_operations_success']:
        print("✓ RPN operations: SUCCESS")
    else:
        print("✗ RPN operations: FAILED")
    
    print(f"Total Benchmark Duration: {results['benchmark_duration']:.1f} seconds")
    print("=" * 60)
    
    print("\nBenchmark complete! Real GPU results saved to:")
    print(f"  - JSON data: {results_file}")
    
    return results


if __name__ == "__main__":
    main()