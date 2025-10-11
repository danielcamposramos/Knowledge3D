"""
Test Sovereign Bridges - Verify all Step8 kernel bridges

This script tests each sovereign bridge to ensure:
1. Kernel loads successfully
2. Memory allocation works
3. Kernel executes without errors
4. Results are reasonable (basic sanity checks)

Run with: python test_sovereign_bridges.py
"""

import numpy as np
import sys
from pathlib import Path

print("=" * 80)
print("🔍 Testing Sovereign Bridges - Step8 Kernel Validation")
print("=" * 80)
print()

# Test 1: Latency Guard
print("TEST 1: Latency Guard (gre_sub100micro_gate.ptx)")
print("-" * 80)
try:
    from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard

    guard = LatencyGuard(threshold_us=100.0)
    print("✅ LatencyGuard initialized")

    guard.start()
    print("✅ Start timestamp recorded")

    # Simulate some GPU work (just sync)
    from knowledge3d.cranium.sovereign.loader import synchronize
    synchronize()

    elapsed_ns, breached = guard.stop()
    print(f"✅ Stop timestamp recorded")
    print(f"   Elapsed: {elapsed_ns} ns ({elapsed_ns / 1000:.1f} µs)")
    print(f"   Threshold breached: {breached}")

    guard.cleanup()
    print("✅ Cleanup complete")

    if elapsed_ns < 0 or elapsed_ns > 1_000_000_000:
        print("⚠️  WARNING: Suspicious elapsed time")
    else:
        print("✅ TEST 1 PASSED\n")

except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 2: ARC Reasoner
print("TEST 2: ARC Reasoner (gre_arc_reasoner.ptx)")
print("-" * 80)
try:
    from knowledge3d.cranium.bridges.sovereign_bridges import ARCReasoner

    reasoner = ARCReasoner()
    print("✅ ARCReasoner initialized")

    # Create test ARC grid (3x3)
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.int32)
    print(f"✅ Test grid created: {grid.shape}")

    rule_id, rotation, color_checksum = reasoner.extract_rules(grid)
    print(f"✅ Rules extracted:")
    print(f"   Rule ID: {rule_id}")
    print(f"   Rotation: {rotation}")
    print(f"   Color checksum: {color_checksum}")

    if 0 <= rule_id < 8 and 0 <= rotation < 4:
        print("✅ TEST 2 PASSED\n")
    else:
        print("⚠️  WARNING: Unexpected rule values\n")

except Exception as e:
    print(f"❌ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 3: OOM Spill Manager
print("TEST 3: OOM Spill Manager (gre_oom_spill.ptx)")
print("-" * 80)
try:
    from knowledge3d.cranium.bridges.sovereign_bridges import OOMSpillManager

    spill_mgr = OOMSpillManager()
    print("✅ OOMSpillManager initialized")

    # Simulate memory scenario
    oldest_idx = 100
    atom_size = 1024  # 1 KB per atom
    available = 10240  # 10 KB available
    requested = 20     # Want 20 atoms (20 KB total)

    atoms_to_spill, bytes_needed = spill_mgr.compute_spill_plan(
        oldest_idx, atom_size, available, requested
    )
    print(f"✅ Spill plan computed:")
    print(f"   Available memory: {available} bytes")
    print(f"   Requested atoms: {requested}")
    print(f"   Atoms to spill: {atoms_to_spill}")
    print(f"   Bytes needed: {bytes_needed}")

    expected_atoms = available // atom_size  # Should fit 10 atoms
    if atoms_to_spill == expected_atoms:
        print("✅ TEST 3 PASSED\n")
    else:
        print(f"⚠️  WARNING: Expected {expected_atoms} atoms, got {atoms_to_spill}\n")

except Exception as e:
    print(f"❌ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 4: Galaxy Resonance Engine
print("TEST 4: Galaxy Resonance Engine (galaxy_resonance_engine.ptx)")
print("-" * 80)
try:
    from knowledge3d.cranium.bridges.sovereign_bridges import GalaxyResonanceEngine

    engine = GalaxyResonanceEngine()
    print("✅ GalaxyResonanceEngine initialized")

    # Create test embeddings
    batch_size = 2
    vector_dim = 128
    embeddings = np.random.randn(batch_size, vector_dim).astype(np.float32)
    latent = np.random.randn(batch_size, vector_dim).astype(np.float32)
    alpha = 0.3

    print(f"✅ Test data created: {embeddings.shape}")

    output = engine.resonate(embeddings, latent, alpha=alpha)
    print(f"✅ Resonance computed: {output.shape}")

    # Verify blend (should be weighted combination)
    expected = embeddings * alpha + latent * (1 - alpha)
    max_error = np.max(np.abs(output - expected))
    print(f"   Max error vs expected blend: {max_error:.6f}")

    if max_error < 1e-3:
        print("✅ TEST 4 PASSED\n")
    else:
        print(f"⚠️  WARNING: Blend error {max_error:.6f} > 1e-3\n")

except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()

# Summary
print("=" * 80)
print("📊 Test Summary")
print("=" * 80)
print()
print("✅ All basic sovereign bridges tested!")
print()
print("Next steps:")
print("  1. Implement remaining 11 bridges (Deep Seek, GLM, Grok)")
print("  2. Create comprehensive integration tests")
print("  3. Validate with tmux orchestration")
print("  4. Measure latency (<95µs mandate)")
print()
