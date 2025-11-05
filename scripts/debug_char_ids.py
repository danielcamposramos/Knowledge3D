#!/usr/bin/env python3
"""Debug character ID detection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge

# Initialize bridge
bridge = PhaseGPDFIngestionBridge()

# Check atomic classifier setup
detector = bridge.character_detector
if detector.atomic_char_ids_list is None:
    print("ERROR: No atomic classifiers loaded!")
    sys.exit(1)

char_ids = detector.atomic_char_ids_list
print(f"Loaded {len(char_ids)} character classifiers:")
for idx, char_id in enumerate(char_ids):
    char = chr(char_id) if 32 <= char_id < 127 else '?'
    print(f"  idx={idx:2d}  char_id={char_id:3d}  char='{char}'")

print(f"\nLast character: idx={len(char_ids)-1}, char_id={char_ids[-1]}, char='{chr(char_ids[-1])}'")
print(f"Tilde ('~'): char_id={ord('~')} (should NOT be in training set)")

# Quick sanity check
expected_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
expected_codes = sorted([ord(c) for c in expected_chars])
actual_codes = sorted(char_ids.tolist())

if expected_codes == actual_codes:
    print("\n✓ Character IDs match expected BASE_CHARACTERS")
else:
    missing = set(expected_codes) - set(actual_codes)
    extra = set(actual_codes) - set(expected_codes)
    print(f"\n✗ Character ID mismatch!")
    if missing:
        print(f"  Missing: {[chr(c) for c in missing]}")
    if extra:
        print(f"  Extra: {[chr(c) for c in extra]}")
