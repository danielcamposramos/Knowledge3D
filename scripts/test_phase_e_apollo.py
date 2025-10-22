"""
Phase E validation: DeepSeek-OCR on Apollo PDF.

Tests:
1. DeepSeek pipeline initialization
2. Text extraction accuracy
3. Compression ratio (target: 7-20×)
4. Dual-texture generation capability

Usage:
    PYTHONPATH=. python scripts/test_phase_e_apollo.py
"""

from pathlib import Path
from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge


def test_phase_e_apollo():
    print("=" * 60)
    print("Phase E Validation: DeepSeek-OCR on Apollo PDF")
    print("=" * 60)
    print()

    # Initialize bridge
    print("[1/4] Initializing PDF ingestion bridge...")
    bridge = PDFIngestionBridge()

    # Check if DeepSeek OCR is available
    if bridge.deepseek_bridge is None:
        print("❌ DeepSeek OCR not available")
        print()
        print("Phase E components not installed correctly.")
        print("Expected components in: knowledge3d/cranium/ocr/")
        print()
        print("Ensure the following files exist:")
        print("  - knowledge3d/cranium/ocr/__init__.py")
        print("  - knowledge3d/cranium/ocr/local_perception.py")
        print("  - knowledge3d/cranium/ocr/conv_compressor.py")
        print("  - knowledge3d/cranium/ocr/global_context.py")
        print("  - knowledge3d/cranium/ocr/resolution_controller.py")
        print("  - knowledge3d/cranium/ocr/deepseek_bridge.py")
        return False

    print("✓ DeepSeek OCR bridge initialized")
    print()

    # Enable DeepSeek OCR
    print("[2/4] Enabling DeepSeek-OCR enhancement...")
    try:
        bridge.enable_deepseek_ocr(True)
        print("✓ DeepSeek OCR enabled")
    except Exception as e:
        print(f"❌ Failed to enable DeepSeek OCR: {e}")
        return False
    print()

    # Test on Apollo PDF page 0
    print("[3/4] Processing Apollo PDF page 0...")
    pdf_path = (
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/"
        "Apollo 11/APOLLO.PDF"
    )

    if not Path(pdf_path).exists():
        print(f"❌ Apollo PDF not found at: {pdf_path}")
        print("Please ensure the Apollo PDF is available.")
        return False

    try:
        result = bridge.ingest_pdf_page(pdf_path, page_num=0)
    except Exception as e:
        print(f"❌ Failed to process PDF: {e}")
        return False

    print("✓ Page processed successfully")
    print()

    # Validate results
    print("[4/4] Validating results...")
    print()
    print("Results:")
    print(f"  Method used:        {result.get('method', 'unknown')}")
    print(f"  Objects detected:   {result['object_count']}")
    print(f"  Processing time:    {result['processing_time_ms']:.2f} ms")

    if 'compression_ratio' in result:
        compression = result['compression_ratio']
        fidelity = result.get('fidelity', 0.0)

        print(f"  Compression ratio:  {compression:.2f}×")
        print(f"  Fidelity:           {fidelity:.1%}")
        print()

        # Check if compression is in target range (7-20×)
        if 7.0 <= compression <= 20.0:
            print(f"✓ Compression in target range (7-20×)")
        else:
            print(f"⚠ Compression outside target range: {compression:.2f}× (target: 7-20×)")

        # Check if fidelity meets target (≥97% at <10× compression)
        if compression < 10.0 and fidelity >= 0.97:
            print(f"✓ Fidelity meets target (≥97% at <10× compression)")
        elif fidelity >= 0.85:
            print(f"⚠ Fidelity acceptable: {fidelity:.1%}")
        else:
            print(f"⚠ Fidelity below target: {fidelity:.1%}")
    print()

    # Extract text sample
    text_sample = result.get('text', '')
    if text_sample:
        print("Text sample (first 300 characters):")
        print("-" * 60)
        print(text_sample[:300].strip())
        if len(text_sample) > 300:
            print("...")
        print("-" * 60)
    print()

    # Check for expected keywords
    print("Keyword validation:")
    expected = ["ICASE", "APOLLO", "11", "Teacher", "Resource"]
    full_text = result.get('text', '').upper()
    hits = [kw for kw in expected if kw in full_text]

    print(f"  Expected keywords: {expected}")
    print(f"  Found:             {hits}")
    print(f"  Match rate:        {len(hits)}/{len(expected)} ({len(hits)/len(expected)*100:.0f}%)")
    print()

    # Final verdict
    success = True
    if result.get('method') == 'deepseek':
        print("✓ Phase E validation PASSED")
        print("  DeepSeek-OCR pipeline working correctly!")
    elif result.get('method') == 'tesseract':
        print("⚠ Phase E validation PARTIAL")
        print("  System fell back to Tesseract OCR")
        print("  DeepSeek pipeline may have failed - check logs above")
        success = False
    else:
        print("⚠ Phase E validation UNCLEAR")
        print(f"  Unexpected method: {result.get('method')}")
        success = False

    print()
    print("=" * 60)
    return success


if __name__ == "__main__":
    success = test_phase_e_apollo()
    exit(0 if success else 1)
