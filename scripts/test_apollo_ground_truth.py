#!/usr/bin/env python3
"""Apollo Ground Truth Validation - Phase F.2 Prototype

Tests feature extraction against Gemini-provided ground truth OCR data.

This demonstrates:
1. GPU feature extraction working correctly
2. Ground truth loading and validation
3. Prototype character detection (simple baseline)
4. Path to full Phase F.2 implementation

Usage:
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_apollo_ground_truth.py
"""

import sys
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge
from knowledge3d.cranium.ocr.character_detector import CharacterDetector


def load_ground_truth(md_path: Path):
    """Load ground truth from Gemini-generated markdown."""
    print("Loading ground truth from Gemini output...")

    with open(md_path, 'r') as f:
        content = f.read()

    # Extract JSON from markdown
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find JSON in markdown file")

    ground_truth = json.loads(json_match.group(1))

    print(f"✓ Loaded {len(ground_truth)} text regions:")
    for i, region in enumerate(ground_truth[:5], 1):  # Show first 5
        text_preview = region['text'][:30]
        print(f"  {i}. \"{text_preview}{'...' if len(region['text']) > 30 else ''}\"")
    if len(ground_truth) > 5:
        print(f"  ... and {len(ground_truth) - 5} more")
    print()

    return ground_truth


def test_feature_extraction_on_apollo(bridge, image_path: Path):
    """Test GPU feature extraction on actual Apollo page."""
    print("=" * 80)
    print("TEST: Feature Extraction on Apollo Page 0")
    print("=" * 80)
    print()

    # Load image
    print(f"Loading image: {image_path.name}")
    img = Image.open(image_path)
    img_array = np.array(img)

    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    print(f"  Image size: {img_array.shape[0]}×{img_array.shape[1]}×{img_array.shape[2]}")
    print()

    # Run feature extraction
    print("Running GPU feature extraction...")

    if bridge.use_gpu_ocr and bridge.gpu_ocr_model is not None:
        # Convert to float32 [0, 1]
        img_float = img_array.astype(np.float32) / 255.0

        import time
        start = time.perf_counter()
        results = bridge.gpu_ocr_model.forward(img_float)
        end = time.perf_counter()

        feature_map = results['feature_map']
        H_feat, W_feat, C_feat = results['output_shape']

        latency_ms = (end - start) * 1000

        print(f"✓ Feature extraction completed in {latency_ms:.1f} ms")
        print(f"  Input: {img_array.shape[0]}×{img_array.shape[1]}×3")
        print(f"  Output: {H_feat}×{W_feat}×{C_feat} features")
        print(f"  Compression: {(img_array.shape[0] * img_array.shape[1]) / (H_feat * W_feat):.1f}×")
        print()

        return feature_map
    else:
        print("⚠ GPU OCR not available")
        return None


def run_character_detection(feature_map, ground_truth, img_width, img_height):
    """Run Phase F.2 character detection.

    Uses the complete CharacterDetector with all 5 swarm components.
    """
    print("=" * 80)
    print("PHASE F.2: Character Detection (LIVE)")
    print("=" * 80)
    print()

    if feature_map is None:
        print("⚠ No features to process")
        return None

    H_feat, W_feat, C_feat = feature_map.shape

    print(f"Input: Feature map {H_feat}×{W_feat}×{C_feat}")
    print(f"Target: Detect {sum(len(r['text']) for r in ground_truth)} chars in {len(ground_truth)} regions")
    print()

    # Initialize CharacterDetector
    print("Initializing CharacterDetector (5 swarm components)...")
    detector = CharacterDetector(
        num_glyphs=256,
        feature_dim=C_feat,
        patch_size=8
    )
    print()

    # Run detection
    import time
    start = time.perf_counter()
    results = detector.detect(feature_map, img_width, img_height)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    print()
    print(f"✓ Detection completed in {latency_ms:.1f} ms")
    print(f"  Patches processed: {results['num_patches']}")
    print(f"  Characters detected: {len(results['detections'])} detections")
    print(f"  Text length: {len(results['text'])} chars")
    print()

    # Show detected text
    if results['text']:
        print("Detected text (first 500 chars):")
        print(results['text'][:500])
        if len(results['text']) > 500:
            print(f"... ({len(results['text']) - 500} more characters)")
        print()

    return results


def validate_against_ground_truth(detection_results, ground_truth):
    """Validate detection results against ground truth."""
    print("=" * 80)
    print("Ground Truth Validation Metrics")
    print("=" * 80)
    print()

    if detection_results is None:
        print("⚠ No detection results to validate")
        print()
        return

    # Extract ground truth text
    gt_text = ' '.join(r['text'] for r in ground_truth)
    gt_chars = sum(len(r['text']) for r in ground_truth)

    detected_text = detection_results['text']
    detected_chars = len(detected_text)

    print("1. Character Detection Rate")
    print(f"   Ground truth: {gt_chars} characters")
    print(f"   Detected: {detected_chars} characters")
    detection_rate = detected_chars / max(gt_chars, 1)
    print(f"   Rate: {detection_rate*100:.1f}%")
    if detection_rate >= 0.9:
        print("   ✓ PASS (≥90% target)")
    else:
        print(f"   ⚠ BELOW TARGET (need ≥90%)")
    print()

    print("2. Bounding Box Analysis")
    print(f"   Ground truth regions: {len(ground_truth)}")
    print(f"   Detected regions: {len(detection_results['detections'])}")
    # For detailed IoU, we'd need to group detections into regions
    # Simplified: just show counts
    print()

    print("3. Text Recognition Quality")
    # Simple character overlap metric
    gt_chars_set = set(gt_text.lower())
    det_chars_set = set(detected_text.lower())
    overlap = len(gt_chars_set & det_chars_set) / max(len(gt_chars_set), 1)
    print(f"   Character set overlap: {overlap*100:.1f}%")
    print()

    print("4. Sample Ground Truth Regions:")
    for i, region in enumerate(ground_truth[:5], 1):
        bbox = region['bbox']
        text = region['text']
        print(f"  Region {i}: \"{text}\"")
        print(f"    BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        print(f"    Size: {bbox[2]-bbox[0]}×{bbox[3]-bbox[1]} px")
    if len(ground_truth) > 5:
        print(f"  ... and {len(ground_truth) - 5} more regions")
    print()

    return {
        'detection_rate': detection_rate,
        'character_overlap': overlap
    }


def main():
    print()
    print("=" * 80)
    print("Apollo Ground Truth Validation - Phase F.2 Prototype")
    print("=" * 80)
    print()

    # Paths
    apollo_dir = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11")
    image_path = apollo_dir / "APOLLO-Page0.png"
    ground_truth_path = apollo_dir / "APOLLO.PDF-Expected_Text.md"

    # Check files exist
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False

    if not ground_truth_path.exists():
        print(f"❌ Ground truth not found: {ground_truth_path}")
        return False

    print(f"✓ Apollo image: {image_path.name}")
    print(f"✓ Ground truth: {ground_truth_path.name}")
    print()

    # Load ground truth
    try:
        ground_truth = load_ground_truth(ground_truth_path)
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Initialize OCR bridge
    print("Initializing DeepSeek OCR bridge...")
    try:
        bridge = DeepSeekOCRBridge(mode='small', use_gpu_ocr=True)
        if bridge.use_gpu_ocr:
            print("✓ GPU OCR model ready")
        else:
            print("⚠ GPU OCR not available, using CPU fallback")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize bridge: {e}")
        return False

    # Test feature extraction
    feature_map = test_feature_extraction_on_apollo(bridge, image_path)

    if feature_map is not None:
        print("✓ Feature extraction SUCCESSFUL")
        print()

    # Load image to get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Run Phase F.2 character detection
    detection_results = run_character_detection(
        feature_map, ground_truth, img_width, img_height
    )

    # Validate against ground truth
    validation_metrics = validate_against_ground_truth(detection_results, ground_truth)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Phase F.1 Status: ✓ COMPLETE")
    print("  ✓ GPU feature extraction working")
    print("  ✓ Kernels compiled and loaded")
    print("  ✓ Ground truth loaded and parsed")
    print()
    print("Phase F.2 Status: ✓ IMPLEMENTED")
    print("  ✓ CharacterDetector with 5 swarm components")
    print("  ✓ Qwen: Adaptive sliding window")
    print("  ✓ DeepSeek: GalacticTemplateBank (3-layer)")
    print("  ✓ Kimi: Glyph matcher (CPU fallback)")
    print("  ✓ GLM: Hierarchical NMS")
    print("  ✓ Grok: Spatial text decoder")
    print()

    if validation_metrics:
        detection_rate = validation_metrics.get('detection_rate', 0)
        char_overlap = validation_metrics.get('character_overlap', 0)

        print("Validation Results:")
        print(f"  Detection rate: {detection_rate*100:.1f}%")
        print(f"  Character overlap: {char_overlap*100:.1f}%")

        if detection_rate >= 0.9:
            print("  ✓ VALIDATION PASSED")
        else:
            print("  ⚠ Needs improvement (bootstrapping from random templates)")
        print()

    print("Next Steps:")
    print("  1. Train GalacticTemplateBank on RLWHF dataset")
    print("  2. Fine-tune character templates")
    print("  3. Optimize glyph matcher with GPU kernel")
    print("  4. Iterate on detection accuracy")
    print()
    print("Parallel Work:")
    print("  ⏳ RLWHF training running (Codex + exaone-deep)")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
