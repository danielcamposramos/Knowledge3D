#!/usr/bin/env python3
"""Test if GPU RPN execution actually works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

# Initialize bridge
bridge = ProceduralDrawingBridge(matryoshka_dim=512)

# Check if GPU kernels loaded
print(f"Pixel Genesis Module: {bridge.pixel_genesis_module}")
print(f"Pixel Genesis Kernel: {bridge.pixel_genesis_kernel}")
print(f"RPN Executor Kernel: {bridge.rpn_executor_kernel}")
print()

# Try simple RPN execution
simple_rpn = "0.5 0.5 MOVE 0.7 0.7 LINE STROKE"

print(f"Executing RPN: {simple_rpn}")
print()

try:
    result = bridge.execute_rpn_gpu(simple_rpn, width=256, height=256, skip_raster=True)
    print(f"Success!")
    print(f"  Segments: {len(result.segments) if result.segments is not None else 0}")
    print(f"  Latency: {result.latency_us:.2f} µs")

    if result.segments is not None and len(result.segments) > 0:
        print(f"  First segment: {result.segments[0]}")
    else:
        print(f"  WARNING: No segments generated!")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
