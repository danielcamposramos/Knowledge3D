#!/bin/bash
#
# GPU Test Script for Thinking Tag Bridge with Claude's 6 Enhancements
# Runs in tmux session with k3d-cranium conda environment
#

set -e

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "THINKING TAG BRIDGE - GPU INTEGRATION TEST"
echo "Claude's 6 Enhancements with RPN PTX Kernel"
echo "================================================================================"
echo ""

# Test if we're in a tmux session
if [ -z "$TMUX" ]; then
    echo -e "${YELLOW}Not in tmux session. Starting new tmux session 'k3d-thinking-tags'...${NC}"
    tmux new-session -d -s k3d-thinking-tags "bash $0 --inside-tmux"
    echo -e "${GREEN}✓ Tmux session created${NC}"
    echo ""
    echo "To watch the test running:"
    echo "  tmux attach -t k3d-thinking-tags"
    echo ""
    echo "To kill the session:"
    echo "  tmux kill-session -t k3d-thinking-tags"
    exit 0
fi

# Inside tmux session
if [ "$1" == "--inside-tmux" ]; then
    echo -e "${GREEN}Running inside tmux session${NC}"
    echo ""

    # Activate conda environment
    echo -e "${YELLOW}Activating k3d-cranium conda environment...${NC}"
    source /home/daniel/miniforge/etc/profile.d/conda.sh
    conda activate k3d-cranium
    echo -e "${GREEN}✓ Environment activated${NC}"
    echo ""

    # Set working directory
    cd "${REPO_ROOT}"
    export PYTHONPATH=.

    # Enable telemetry for this test
    export K3D_ENABLE_TELEMETRY=1
    export K3D_TELEMETRY_DIR="/K3D/Knowledge3D.local/logs"

    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  PYTHONPATH=$PYTHONPATH"
    echo "  K3D_ENABLE_TELEMETRY=$K3D_ENABLE_TELEMETRY"
    echo "  K3D_TELEMETRY_DIR=$K3D_TELEMETRY_DIR"
    echo ""

    # Check GPU availability
    echo -e "${YELLOW}Checking GPU availability...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo -e "${GREEN}✓ GPU detected${NC}"
    else
        echo -e "${RED}✗ nvidia-smi not found${NC}"
    fi
    echo ""

    # Create log directory
    mkdir -p /K3D/Knowledge3D.local/logs
    LOG_FILE="/K3D/Knowledge3D.local/logs/thinking_tag_test_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${YELLOW}Running GPU integration test...${NC}"
    echo "Log file: $LOG_FILE"
    echo ""

    # Run the test
    /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 << 'PYEOF' 2>&1 | tee "$LOG_FILE"
"""
GPU Integration Test for Thinking Tag Bridge with Claude's 6 Enhancements
"""
import sys
import os
import time
import numpy as np

print("=" * 80)
print("THINKING TAG BRIDGE - GPU INTEGRATION TEST")
print("=" * 80)
print()

# Import enhancement modules first
print("Phase 1: Loading enhancement modules...")
try:
    from knowledge3d.cranium.ptx_runtime.latency_profiler import LatencyProfiler
    from knowledge3d.cranium.ptx_runtime.sparse_weight_cache import SparseWeightCache
    from knowledge3d.cranium.ptx_runtime.modal_affinity_matrix import ModalAffinityMatrix
    from knowledge3d.cranium.ptx_runtime.telemetry_visualizer import TelemetryVisualizer
    from knowledge3d.cranium.ptx_runtime.enhanced_fallback import EnhancedFallback
    print("✓ All enhancement modules loaded successfully")
except Exception as e:
    print(f"✗ Enhancement module import failed: {e}")
    sys.exit(1)
print()

# Now import bridge (this will initialize CUDA)
print("Phase 2: Initializing Thinking Tag Bridge with GPU...")
try:
    from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge
    bridge = ThinkingTagBridge()
    print("✓ Bridge initialized with CUDA context")
except Exception as e:
    print(f"✗ Bridge initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Verify enhancements are active
print("Phase 3: Verifying enhancement integration...")
checks = [
    ("Latency Profiler", hasattr(bridge, 'latency_profiler') and bridge.latency_profiler is not None),
    ("Sparse Weight Cache", hasattr(bridge, 'weight_cache') and bridge.weight_cache is not None),
    ("Modal Affinity Matrix", hasattr(bridge, 'modal_affinity') and bridge.modal_affinity is not None),
    ("Enhanced Fallback", hasattr(bridge, 'enhanced_fallback') and bridge.enhanced_fallback is not None),
    ("Telemetry Visualizer", hasattr(bridge, 'telemetry')),  # Can be None if disabled
]

all_ok = True
for name, status in checks:
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {name}: {'Active' if status else 'MISSING'}")
    if not status and name != "Telemetry Visualizer":
        all_ok = False

if not all_ok:
    print("\n✗ Some enhancements are missing!")
    sys.exit(1)
print()

# Run inference tests
print("Phase 4: Running GPU inference tests...")
print("-" * 80)

test_cases = [
    ("Single-modal (text)", ['text']),
    ("Dual-modal (text+image)", ['text', 'image']),
    ("Dual-modal (text+audio)", ['text', 'audio']),
    ("Tri-modal (all)", ['text', 'image', 'audio']),
]

results = []
for test_name, modal_sig in test_cases:
    try:
        # Generate random embedding
        input_emb = np.random.randn(512).astype(np.float32)

        # Run inference
        start_time = time.perf_counter()
        tags = bridge.inference(
            input_embedding=input_emb,
            modal_signature=modal_sig,
            temporal_anchor=0.5
        )
        latency_us = (time.perf_counter() - start_time) * 1e6

        # Check results
        if tags is not None and len(tags) > 0:
            print(f"  ✓ {test_name}: {len(tags)} tags in {latency_us:.2f}µs")
            results.append((test_name, True, latency_us, len(tags)))
        else:
            print(f"  ✗ {test_name}: No tags generated")
            results.append((test_name, False, latency_us, 0))

    except Exception as e:
        print(f"  ✗ {test_name}: Exception - {e}")
        results.append((test_name, False, 0, 0))

print()

# Check latency budget
print("Phase 5: Latency analysis...")
print("-" * 80)
successful_tests = [r for r in results if r[1]]
if successful_tests:
    avg_latency = sum(r[2] for r in successful_tests) / len(successful_tests)
    max_latency = max(r[2] for r in successful_tests)
    min_latency = min(r[2] for r in successful_tests)

    print(f"  Average latency: {avg_latency:.2f}µs")
    print(f"  Min latency: {min_latency:.2f}µs")
    print(f"  Max latency: {max_latency:.2f}µs")
    print(f"  Target budget: <35µs")

    if avg_latency < 35.0:
        print(f"  ✓ Within budget (margin: {35.0 - avg_latency:.2f}µs)")
    else:
        print(f"  ✗ Over budget (excess: {avg_latency - 35.0:.2f}µs)")
else:
    print("  ✗ No successful tests to analyze")
print()

# Enhancement statistics
print("Phase 6: Enhancement statistics...")
print("-" * 80)
try:
    stats = bridge.get_enhancement_stats()

    # Cache statistics
    cache_stats = stats['sparse_cache']
    print(f"Cache Performance:")
    print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"  Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")

    # Fallback statistics
    fallback_stats = stats['enhanced_fallback']
    print(f"\nFallback Usage:")
    print(f"  Total fallbacks: {fallback_stats['total_fallbacks']}")

    # Latency profiler statistics
    latency_stats = stats['latency_profiler']
    print(f"\nStage Breakdown:")
    for stage in ['query', 'rpn_exec', 'crystallize', 'confidence']:
        if stage in latency_stats['stages']:
            stage_data = latency_stats['stages'][stage]
            print(f"  {stage}: {stage_data.get('avg_us', 0):.2f}µs avg")

except Exception as e:
    print(f"✗ Failed to get statistics: {e}")
print()

# Final report
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
success_count = sum(1 for r in results if r[1])
total_count = len(results)
print(f"Tests passed: {success_count}/{total_count}")
print(f"Enhancement modules: {'✓ All operational' if all_ok else '✗ Some missing'}")

if success_count == total_count and all_ok:
    print("\n✅ ALL TESTS PASSED - GPU integration successful!")
    print("🎉 The RPN PTX kernel is humming beautifully! ⭐")
else:
    print("\n⚠️  Some tests failed - review log for details")

print("=" * 80)

# Print telemetry location if enabled
if os.getenv("K3D_ENABLE_TELEMETRY", "0") == "1":
    telemetry_dir = os.getenv("K3D_TELEMETRY_DIR", "/tmp/k3d")
    print(f"\nTelemetry output: {telemetry_dir}/thinking_tags.prom")
    print(f"Dashboard: {telemetry_dir}/thinking_tag_dashboard.txt")

sys.exit(0 if success_count == total_count else 1)
PYEOF

    TEST_EXIT_CODE=$?

    echo ""
    echo "================================================================================"
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ GPU TEST COMPLETED SUCCESSFULLY${NC}"
    else
        echo -e "${RED}⚠️  GPU TEST HAD FAILURES (exit code: $TEST_EXIT_CODE)${NC}"
    fi
    echo "================================================================================"
    echo ""
    echo "Log saved to: $LOG_FILE"

    if [ -f "/K3D/Knowledge3D.local/logs/thinking_tags.prom" ]; then
        echo "Telemetry: /K3D/Knowledge3D.local/logs/thinking_tags.prom"
    fi

    echo ""
    echo "Press ENTER to close tmux session, or Ctrl+C to keep session open"
    read -r

    exit $TEST_EXIT_CODE
fi
