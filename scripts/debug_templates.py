#!/usr/bin/env python3
"""Debug script to inspect glyph template bank"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge
import numpy as np

print("Initializing Phase G bridge...")
bridge = PhaseGPDFIngestionBridge(
    phase_g_checkpoint_dir=Path("/K3D/Knowledge3D.local/checkpoints/phase_g")
)

print("\n" + "=" * 80)
print("CHARACTER DETECTOR TEMPLATE INSPECTION")
print("=" * 80)

detector = bridge.character_detector

print(f"\nTemplate source: {detector._template_source}")
print(f"Template embeddings shape: {detector.template_embeddings.shape if detector.template_embeddings is not None else 'None'}")
print(f"Template char IDs shape: {detector.template_char_ids.shape if detector.template_char_ids is not None else 'None'}")

if detector.template_char_ids is not None:
    unique_char_ids = np.unique(detector.template_char_ids)
    print(f"\nUnique character IDs: {len(unique_char_ids)}")
    print(f"Character ID range: {unique_char_ids.min()} - {unique_char_ids.max()}")

    # Show character distribution
    print("\nCharacter distribution:")
    for char_id in sorted(unique_char_ids):
        count = (detector.template_char_ids == char_id).sum()
        try:
            char = chr(char_id)
            print(f"  {char_id:3d} ('{char}'): {count:4d} templates")
        except:
            print(f"  {char_id:3d} (invalid): {count:4d} templates")

    # Check if all are tilde
    tilde_count = (detector.template_char_ids == 126).sum()
    print(f"\nTilde ('~') templates: {tilde_count}/{len(detector.template_char_ids)}")

print("\n" + "=" * 80)
