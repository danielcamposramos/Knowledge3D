"""
Test dual-texture GLB folio generation.

Tests:
1. DualTextureBridge initialization
2. Folio creation from PDF page
3. Dual texture generation (Human 512×512 + AI 256×256)
4. Metadata and compression metrics

Usage:
    PYTHONPATH=. python scripts/test_dual_texture_generation.py
"""

from pathlib import Path
from knowledge3d.cranium.bridges.dual_texture_bridge import DualTextureBridge


def test_dual_texture():
    print("=" * 60)
    print("Dual-Texture Generation Test")
    print("=" * 60)
    print()

    # Initialize bridge
    print("[1/3] Initializing DualTextureBridge...")
    try:
        bridge = DualTextureBridge(mode='small')  # 'small' optimized for House storage
        print("✓ DualTextureBridge initialized (mode: small)")
    except Exception as e:
        print(f"❌ Failed to initialize DualTextureBridge: {e}")
        return False
    print()

    # Prepare PDF
    print("[2/3] Preparing Apollo PDF...")
    pdf_path = Path(
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/"
        "Apollo 11/APOLLO.PDF"
    )

    if not pdf_path.exists():
        print(f"❌ Apollo PDF not found at: {pdf_path}")
        return False

    print(f"✓ PDF found: {pdf_path.name}")
    print()

    # Create folio
    print("[3/3] Creating dual-texture folio...")
    try:
        folio = bridge.create_folio(
            pdf_path=pdf_path,
            page_num=0,
            metadata={
                'title': 'Apollo 11 Teacher Resource',
                'author': 'NASA',
                'category': 'space_exploration'
            }
        )
    except Exception as e:
        print(f"❌ Failed to create folio: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Folio generated successfully")
    print()

    # Validate folio
    print("Folio Details:")
    print(f"  Mode:                {folio['mode']}")
    print(f"  Human texture shape: {folio['human_texture'].shape}")
    print(f"  AI texture shape:    {folio['ai_texture'].shape}")
    print(f"  Text length:         {len(folio['text'])} characters")
    print(f"  Compression ratio:   {folio['compression_ratio']:.2f}×")
    print(f"  Fidelity:            {folio['fidelity']:.1%}")
    print(f"  Global context dim:  {folio['global_context'].shape}")
    print()

    print("Metadata:")
    for key, value in folio['metadata'].items():
        print(f"  {key:20s}: {value}")
    print()

    # Validate texture dimensions
    success = True

    print("Validation:")
    if folio['human_texture'].shape == (512, 512, 3):
        print("  ✓ Human texture: 512×512 RGB (correct)")
    else:
        print(f"  ❌ Human texture: {folio['human_texture'].shape} (expected: 512×512×3)")
        success = False

    if folio['ai_texture'].shape == (256, 256, 3):
        print("  ✓ AI texture: 256×256 RGB (correct)")
    else:
        print(f"  ❌ AI texture: {folio['ai_texture'].shape} (expected: 256×256×3)")
        success = False

    # Check compression metrics
    if 7.0 <= folio['compression_ratio'] <= 20.0:
        print(f"  ✓ Compression: {folio['compression_ratio']:.2f}× (target: 7-20×)")
    else:
        print(f"  ⚠ Compression: {folio['compression_ratio']:.2f}× (outside target: 7-20×)")

    if folio['fidelity'] >= 0.85:
        print(f"  ✓ Fidelity: {folio['fidelity']:.1%} (acceptable)")
    else:
        print(f"  ⚠ Fidelity: {folio['fidelity']:.1%} (below threshold)")

    if folio['global_context'].shape == (512,):
        print("  ✓ Global context: 512-dim (correct)")
    else:
        print(f"  ⚠ Global context: {folio['global_context'].shape} (expected: 512)")
    print()

    # Text sample
    if folio['text']:
        print("Text sample (first 200 characters):")
        print("-" * 60)
        print(folio['text'][:200].strip())
        if len(folio['text']) > 200:
            print("...")
        print("-" * 60)
    print()

    # Final verdict
    if success:
        print("✓ Dual-texture generation test PASSED")
        print("  Both textures generated with correct dimensions!")
    else:
        print("⚠ Dual-texture generation test FAILED")
        print("  Check texture dimensions above")

    print()
    print("=" * 60)
    print()
    print("Note: Phase E stub returns metadata only.")
    print("Phase F will implement full GLB export with dual UV maps.")
    print()

    return success


if __name__ == "__main__":
    success = test_dual_texture()
    exit(0 if success else 1)
