"""Audit all Step8 PTX kernels - Test which ones load successfully.

This script tests all PTX kernels in knowledge3d/cranium/kernels/
to determine which are valid and which need to be rewritten.
"""

import sys
from pathlib import Path
from knowledge3d.cranium.sovereign.loader import load_ptx_file

# All Step8 kernels from the chain
STEP8_KERNELS = [
    # Kimi's kernels
    ("gre_sub100micro_gate", "Latency Guard - Records timing and checks threshold"),
    ("gre_arc_reasoner", "ARC Reasoner - Extracts rules from grids"),
    ("gre_oom_spill", "OOM Memory Safety - Manages memory overflow"),

    # Qwen's kernel
    ("galaxy_resonance_engine", "Galaxy Resonance Engine - Recursive core"),

    # Deep Seek's kernels
    ("gre_geometry_router", "Geometry Router - Routes based on media type"),
    ("gre_fractal_emitter", "Fractal Emitter - Generates fractal structures"),

    # GLM's kernels
    ("gre_resonance_field", "Resonance Field - Energetic field management"),
    ("gre_atomic_fission_fusion", "Atomic Fission/Fusion - Atom operations"),
    ("gre_temporal_reasoning", "Temporal Reasoning - Sequential reasoning"),

    # Grok's kernels
    ("gre_vector_resonator", "Vector Resonator - Recursive ANN search"),
    ("gre_graph_crystallizer", "Graph Crystallizer - Recursive GNN"),
    ("gre_multimodal_halting_gate", "Multimodal Halting - Geometry-aware halting"),

    # Claude's kernels
    ("gre_recursive_refiner", "Recursive Refiner - TRM core logic"),
    ("gre_cognitive_executive", "Cognitive Executive - Pipeline orchestration"),
    ("gre_trm_core", "TRM Core - Tiny Recursive Model core"),
]

print("=" * 80)
print("🔍 Step8 PTX Kernel Audit")
print("=" * 80)
print()

kernels_dir = Path("knowledge3d/cranium/kernels")
if not kernels_dir.exists():
    print(f"❌ Kernels directory not found: {kernels_dir}")
    sys.exit(1)

print(f"📂 Kernels directory: {kernels_dir}")
print(f"🔢 Total kernels to audit: {len(STEP8_KERNELS)}")
print()

results = {
    "valid": [],
    "invalid": [],
    "missing": []
}

for kernel_name, description in STEP8_KERNELS:
    print(f"Testing: {kernel_name}.ptx")
    print(f"   Purpose: {description}")

    ptx_path = kernels_dir / f"{kernel_name}.ptx"

    if not ptx_path.exists():
        print(f"   ❌ File not found")
        results["missing"].append((kernel_name, description))
        print()
        continue

    # Check file size
    size_kb = ptx_path.stat().st_size / 1024
    print(f"   📏 Size: {size_kb:.1f} KB")

    # Try to load the kernel
    try:
        # Try to load with the kernel's entry point name
        # Most kernels use their name as entry point
        kernel_func = load_ptx_file(str(ptx_path), kernel_name)
        print(f"   ✅ Loaded successfully! (handle: {kernel_func})")
        results["valid"].append((kernel_name, description, size_kb))
    except Exception as e:
        print(f"   ❌ Failed to load: {str(e)[:80]}")
        results["invalid"].append((kernel_name, description, str(e)))

    print()

# Summary
print("=" * 80)
print("📊 Audit Summary")
print("=" * 80)
print()

print(f"✅ Valid kernels: {len(results['valid'])}")
for name, desc, size in results['valid']:
    print(f"   • {name}.ptx ({size:.1f} KB) - {desc}")

print()
print(f"❌ Invalid kernels: {len(results['invalid'])}")
for name, desc, error in results['invalid']:
    print(f"   • {name}.ptx - {desc}")
    print(f"     Error: {error[:100]}")

print()
print(f"⚠️  Missing kernels: {len(results['missing'])}")
for name, desc in results['missing']:
    print(f"   • {name}.ptx - {desc}")

print()
print("=" * 80)
print("🎯 Next Steps")
print("=" * 80)
print()

if results['invalid']:
    print(f"1. Convert {len(results['invalid'])} invalid PTX kernels to CUDA C++")
    print(f"2. Compile to PTX using: nvcc -ptx -arch=sm_86 <file>.cu -o <file>.ptx")
    print(f"3. Re-test with this audit script")

if results['missing']:
    print(f"4. Create {len(results['missing'])} missing kernel implementations")

if results['valid']:
    print(f"5. Create Python bridges for {len(results['valid'])} valid kernels")
    print(f"6. Build unified test suite")

total_kernels = len(results['valid']) + len(results['invalid']) + len(results['missing'])
completion_pct = (len(results['valid']) / total_kernels * 100) if total_kernels > 0 else 0

print()
print(f"📈 Completion: {len(results['valid'])}/{total_kernels} ({completion_pct:.1f}%)")
print()

# Exit with appropriate code
if results['invalid'] or results['missing']:
    print("⚠️  Some kernels need attention!")
    sys.exit(1)
else:
    print("🎉 All kernels are valid!")
    sys.exit(0)
