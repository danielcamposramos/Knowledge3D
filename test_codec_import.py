#!/usr/bin/env python3
"""Test script to debug codec import issue."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D")

print("Step 1: Import numpy")
import numpy as np
print("  ✓ NumPy imported")

print("\nStep 2: Try importing TernaryQuantizer directly")
try:
    sys.path.insert(0, "knowledge3d/cranium/codecs/ptx_bindings")
    from ternary_quant_binding import TernaryQuantizer
    print("  ✓ Direct import works")
    quant = TernaryQuantizer()
    print("  ✓ Direct initialization works")
except Exception as e:
    print(f"  ✗ Direct import/init failed: {e}")

print("\nStep 3: Try importing through package")
# Clear the direct imports
if "ternary_quant_binding" in sys.modules:
    del sys.modules["ternary_quant_binding"]

try:
    from knowledge3d.cranium.codecs.ternary_quantization import quantize_ternary
    print("  ✓ Package import works")
except Exception as e:
    print(f"  ✗ Package import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 4: Try importing AudioHarmonicGPU")
try:
    from knowledge3d.cranium.codecs.ptx_bindings.audio_harmonic_binding import AudioHarmonicGPU
    print("  ✓ AudioHarmonicGPU import works")
    harm = AudioHarmonicGPU()
    print("  ✓ AudioHarmonicGPU initialization works")
except Exception as e:
    print(f"  ✗ AudioHarmonicGPU failed: {e}")
