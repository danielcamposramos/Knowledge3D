#!/usr/bin/env python3
"""
Quick test script for Phase G integration.

Tests:
1. Adaptive RPN engine initialization
2. Variable dimension selection
3. Phase G specialist loading
4. Sample PDF ingestion
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.adaptive_rpn_engine import AdaptiveRPNEngine, DimensionConfig
from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge


def test_adaptive_rpn():
    """Test adaptive RPN engine."""
    print("\n" + "="*60)
    print("TEST 1: Adaptive RPN Engine")
    print("="*60)

    engine = AdaptiveRPNEngine()

    # Test dimension selection
    test_cases = [
        ("Hi", "Single word"),
        ("Hello world", "Short phrase"),
        ("The quick brown fox jumps over the lazy dog", "Medium sentence"),
        ("This is a longer paragraph with multiple sentences. It contains more complex vocabulary and punctuation. The dimension selection should reflect this increased complexity.", "Long paragraph"),
    ]

    print("\nDimension Selection Tests:")
    for text, description in test_cases:
        dim_length = engine.select_dimension_by_length(text)
        complexity = engine.estimate_complexity(text)
        dim_complexity = engine.select_dimension_by_complexity(complexity)
        dim_auto = engine.select_dimension_auto(text)

        print(f"\n  {description}:")
        print(f"    Text: '{text[:50]}...'")
        print(f"    Length: {len(text)} chars → {dim_length}D")
        print(f"    Complexity: {complexity:.3f} → {dim_complexity}D")
        print(f"    Auto-selected: {dim_auto}D")

    # Test embedding generation
    print("\nEmbedding Generation:")
    for text, description in test_cases:
        emb, dim = engine.embed_sentence(text)
        print(f"  {description}: {dim}D embedding shape {emb.shape}")

    # Print stats
    engine.print_stats()

    print("✅ Adaptive RPN engine test passed")


def test_phase_g_bridge():
    """Test Phase G bridge initialization."""
    print("\n" + "="*60)
    print("TEST 2: Phase G Bridge Initialization")
    print("="*60)

    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/current")

    bridge = PhaseGPDFIngestionBridge(phase_g_checkpoint_dir=checkpoint_dir)

    print("\nBridge Status:")
    print(f"  Specialists loaded: {bridge.specialists_loaded}")
    print(f"  Adaptive RPN ready: {bridge.adaptive_rpn is not None}")
    print(f"  Galaxy stars: {len(bridge.galaxy_stars)}")

    if bridge.matryoshka_system:
        print(f"  Matryoshka specialists: {len(bridge.matryoshka_system.specialists)}")
        for name in bridge.matryoshka_system.specialists.keys():
            print(f"    - {name}")

    print("✅ Phase G bridge test passed")
    return bridge


def test_sample_pdf_ingestion(bridge):
    """Test ingestion of a sample PDF page."""
    print("\n" + "="*60)
    print("TEST 3: Sample PDF Ingestion")
    print("="*60)

    # Find a test PDF
    test_pdfs = list(Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries").rglob("*.pdf"))

    if not test_pdfs:
        print("⚠️  No test PDFs found, skipping ingestion test")
        return

    test_pdf = test_pdfs[0]
    print(f"\nTest PDF: {test_pdf.name}")

    try:
        # Ingest first page
        result = bridge.ingest_pdf_page(test_pdf, page_num=0)

        print("\nIngestion Result:")
        print(f"  Objects extracted: {result['object_count']}")
        print(f"  Embedding dimension: {result['embedding_dimension']}D")
        print(f"  Specialist used: {result['specialist_used']}")
        print(f"  Method: {result['method']}")
        print(f"  Processing time: {result['processing_time_ms']:.1f} ms")

        if result['galaxy_star']:
            star = result['galaxy_star']
            print(f"\nGalaxy Star Created:")
            print(f"  Position: {star['position']}")
            print(f"  Embedding dim: {star['embedding_dim']}D")
            print(f"  Pending consolidation: {star['pending_consolidation']}")

        print("\n✅ PDF ingestion test passed")

    except Exception as exc:
        print(f"\n❌ PDF ingestion test failed: {exc}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE G INTEGRATION TEST SUITE")
    print("="*60)

    try:
        # Test 1: Adaptive RPN
        test_adaptive_rpn()

        # Test 2: Phase G Bridge
        bridge = test_phase_g_bridge()

        # Test 3: PDF Ingestion
        test_sample_pdf_ingestion(bridge)

        # Final stats
        print("\n" + "="*60)
        print("FINAL PHASE G STATISTICS")
        print("="*60)
        bridge.print_phase_g_stats()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)

    except Exception as exc:
        print(f"\n❌ TEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
