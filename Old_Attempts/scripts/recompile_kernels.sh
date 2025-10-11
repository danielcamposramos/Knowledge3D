#!/bin/bash
# Recompile all invalid PTX kernels from CUDA C++ source

set -e  # Exit on error

KERNELS_DIR="knowledge3d/cranium/kernels"
cd "$KERNELS_DIR"

echo "🔥 Recompiling Invalid PTX Kernels"
echo "===================================="
echo ""

# List of kernels to recompile (ones that failed audit)
INVALID_KERNELS=(
    "gre_oom_spill"
    "galaxy_resonance_engine"
    "gre_geometry_router"
    "gre_fractal_emitter"
    "gre_resonance_field"
    "gre_atomic_fission_fusion"
    "gre_temporal_reasoning"
    "gre_vector_resonator"
    "gre_graph_crystallizer"
    "gre_multimodal_halting_gate"
)

COMPILED=0
SKIPPED=0

for kernel in "${INVALID_KERNELS[@]}"; do
    echo "Processing: $kernel"

    if [ ! -f "${kernel}.cu" ]; then
        echo "   ⚠️  ${kernel}.cu not found - skipping"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi

    echo "   📝 Compiling ${kernel}.cu -> ${kernel}.ptx"
    if nvcc -ptx -arch=sm_86 "${kernel}.cu" -o "${kernel}.ptx" 2>&1; then
        size=$(du -h "${kernel}.ptx" | cut -f1)
        echo "   ✅ Compiled successfully (${size})"
        COMPILED=$((COMPILED + 1))
    else
        echo "   ❌ Compilation failed"
    fi
    echo ""
done

echo "===================================="
echo "📊 Summary:"
echo "   ✅ Compiled: $COMPILED"
echo "   ⚠️  Skipped: $SKIPPED"
echo "   📦 Total: ${#INVALID_KERNELS[@]}"
echo ""

if [ $COMPILED -gt 0 ]; then
    echo "✅ Re-run audit_step8_kernels.py to verify!"
else
    echo "⚠️  No kernels were compiled. Create .cu files first."
fi
