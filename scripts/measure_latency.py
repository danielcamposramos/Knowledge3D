#!/usr/bin/env python3
"""Measure actual GPU RPN execution latency."""

import numpy as np
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.ptx_runtime.latency_guard import LatencyGuard

bridge = ProceduralDrawingBridge(matryoshka_dim=512)

if bridge.pixel_genesis_kernel is None:
    print("⚠️  Kernel not loaded, skipping")
    exit(0)

programs = [
    "0 0 MOVE 1 1 LINE STROKE",
    "-0.5 -0.5 MOVE 0.5 0.5 LINE STROKE",
    "-0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE",
]

print("=" * 60)
print("GPU RPN Execution Latency Measurements")
print("=" * 60)

for i, program in enumerate(programs, 1):
    guard = LatencyGuard(threshold_us=100.0)

    # Warm up
    bridge.execute_rpn_gpu(program, width=64, height=64)

    # Measure
    guard.start()
    result = bridge.execute_rpn_gpu(program, width=64, height=64)
    elapsed_ns, breached = guard.stop()

    latency_us = elapsed_ns / 1000
    status = "❌ BREACHED" if breached else "✅ OK"

    print(f"\n{i}. {program[:40]}...")
    print(f"   Latency: {latency_us:.1f} µs {status}")
    print(f"   Budget:  100.0 µs")
    print(f"   Pixels:  {np.count_nonzero(result.rgba[..., 0] > 0.01)} drawn")

print("\n" + "=" * 60)
print("Phase 2A Target: <100µs (may need optimization in Phase 2B)")
print("=" * 60)
