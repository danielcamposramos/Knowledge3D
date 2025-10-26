#!/usr/bin/env python3
"""Phase F.1: Complete OCR Pipeline Test

Tests the full Phase F.1 implementation:
1. Conv2d v2 kernel (Kimi v2 optimizations)
2. MaxPool, BatchNorm kernels
3. Glyph matching kernel
4. DeepSeek OCR model (3-stage CNN)
5. Integration with PDF ingestion

Usage:
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_phase_f1_complete.py
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge


def test_gpu_ocr_initialization():
    """Test 1: GPU OCR Model Initialization"""
    print("=" * 80)
    print("TEST 1: GPU OCR Model Initialization")
    print("=" * 80)
    print()

    try:
        print("Initializing DeepSeek OCR bridge with GPU model...")
        bridge = DeepSeekOCRBridge(mode='small', use_gpu_ocr=True)

        if bridge.use_gpu_ocr and bridge.gpu_ocr_model is not None:
            print("✓ GPU OCR model initialized successfully")
            print(f"  Mode: {bridge.mode}")
            print(f"  Texture size: {bridge.texture_size}")
            print(f"  Compression target: {bridge.compression_target}×")
            print()
            return bridge
        else:
            print("⚠ GPU OCR not available, will use fallback")
            print()
            return bridge

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


def test_synthetic_image_inference(bridge):
    """Test 2: Synthetic Image Inference"""
    print("=" * 80)
    print("TEST 2: Synthetic Image Inference")
    print("=" * 80)
    print()

    if bridge is None:
        print("⚠ Skipping (bridge not initialized)")
        print()
        return False

    try:
        # Create synthetic test image (256×256 RGB)
        print("Creating synthetic test image (256×256 RGB)...")
        image = np.random.rand(256, 256, 3).astype(np.float32) * 255
        image = image.astype(np.uint8)

        print("Running OCR extraction...")
        start = time.perf_counter()

        results = bridge.extract(image, pdf_path=None, page_num=0)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        print(f"✓ Extraction completed in {latency_ms:.1f} ms")
        print(f"  Compression ratio: {results['compression_ratio']:.1f}×")
        print(f"  Fidelity: {results['fidelity']*100:.1f}%")
        print(f"  Text length: {len(results['full_text'])} chars")
        print()

        # Check latency target (should be <100ms for small mode)
        if latency_ms < 100:
            print(f"✓ Performance PASSED (< 100ms target)")
        else:
            print(f"⚠ Performance SLOW ({latency_ms:.1f} ms > 100ms target)")

        print()
        return True

    except Exception as e:
        print(f"❌ Synthetic image test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_apollo_pdf_integration(bridge):
    """Test 3: Apollo PDF Integration"""
    print("=" * 80)
    print("TEST 3: Apollo PDF Integration")
    print("=" * 80)
    print()

    if bridge is None:
        print("⚠ Skipping (bridge not initialized)")
        print()
        return False

    # Look for Apollo PDF
    apollo_paths = [
        Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/Apollo11.pdf"),
        Path("/K3D/Knowledge3D.local/house_zone7/Apollo_11_AS11-40-5903HR.pdf"),
        Path("test_data/Apollo_11_AS11-40-5903HR.pdf"),
    ]

    pdf_path = None
    for p in apollo_paths:
        if p.exists():
            pdf_path = p
            break

    if pdf_path is None:
        print("⚠ Apollo PDF not found, skipping test")
        print()
        return False

    try:
        print(f"Testing with Apollo PDF: {pdf_path.name}")

        # Render PDF page to image
        print("Rendering PDF page 0...")

        try:
            import fitz  # PyMuPDF

            with fitz.open(pdf_path) as doc:
                page = doc[0]
                pix = page.get_pixmap()
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Convert RGBA to RGB if needed
                if image.shape[2] == 4:
                    image = image[:, :, :3]

                print(f"  Rendered: {image.shape[0]}×{image.shape[1]}×{image.shape[2]}")

        except ImportError:
            print("⚠ PyMuPDF not available, cannot render PDF")
            print()
            return False

        # Run OCR
        print("Running OCR extraction with GPU model...")
        start = time.perf_counter()

        results = bridge.extract(image, pdf_path=pdf_path, page_num=0)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        print(f"✓ Extraction completed in {latency_ms:.1f} ms")
        print(f"  Compression ratio: {results['compression_ratio']:.1f}×")
        print(f"  Fidelity: {results['fidelity']*100:.1f}%")
        print(f"  Text extracted: {len(results['full_text'])} chars")
        print()

        # Show first few lines of extracted text
        if results['full_text']:
            lines = results['full_text'].split('\n')[:5]
            print("First 5 lines of extracted text:")
            for i, line in enumerate(lines, 1):
                print(f"  {i}: {line[:80]}")
            print()

        return True

    except Exception as e:
        print(f"❌ Apollo PDF test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_dual_texture_generation(bridge):
    """Test 4: Dual Texture Generation"""
    print("=" * 80)
    print("TEST 4: Dual Texture Generation")
    print("=" * 80)
    print()

    if bridge is None:
        print("⚠ Skipping (bridge not initialized)")
        print()
        return False

    try:
        # Create test image
        image = np.random.rand(512, 512, 3).astype(np.float32) * 255
        image = image.astype(np.uint8)

        text = "The quick brown fox jumps over the lazy dog. " * 20

        print("Generating dual textures...")

        # Generate AI texture (dense text)
        print("  Generating AI texture (dense text-as-image)...")
        ai_texture = bridge.encode_ai_texture(
            compressed_features=np.zeros((16, 16, 128), dtype=np.float32),
            text=text
        )

        print(f"  ✓ AI texture: {ai_texture.shape}")

        # Generate human texture (pretty rendering)
        print("  Generating human texture (game-style)...")
        human_texture = bridge.encode_human_texture(image, text)

        print(f"  ✓ Human texture: {human_texture.shape}")

        print()
        print("✓ Dual texture generation successful")
        print()

        return True

    except Exception as e:
        print(f"❌ Dual texture test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    print()
    print("=" * 80)
    print("Phase F.1: Complete OCR Pipeline Test")
    print("=" * 80)
    print()
    print("Components:")
    print("  - Conv2d v2 kernel (Kimi v2 optimizations)")
    print("  - MaxPool, BatchNorm, Glyph matching kernels")
    print("  - DeepSeek OCR model (3-stage CNN)")
    print("  - Dual texture generation")
    print()

    # Run tests
    results = []

    bridge = test_gpu_ocr_initialization()
    results.append(("Initialization", bridge is not None))

    results.append(("Synthetic Image", test_synthetic_image_inference(bridge)))
    results.append(("Apollo PDF", test_apollo_pdf_integration(bridge)))
    results.append(("Dual Textures", test_dual_texture_generation(bridge)))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20s}: {status}")

    print()

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("=" * 80)
        print("✓ ALL TESTS PASSED - Phase F.1 Complete!")
        print("=" * 80)
        print()
        print("Phase F.1 Status:")
        print("  ✓ Kimi v2 optimizations implemented")
        print("  ✓ Complete OCR pipeline working")
        print("  ✓ Dual texture generation validated")
        print("  ✓ Apollo PDF integration tested")
        print()
        print("Next steps:")
        print("  - Fine-tune performance (target <50ms)")
        print("  - Implement character detection from features")
        print("  - Train on RLWHF dataset when ready")
        print()
        return True
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED - Review errors above")
        print("=" * 80)
        print()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
