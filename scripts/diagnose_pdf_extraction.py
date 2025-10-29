#!/usr/bin/env python3
"""
Diagnostic script to test PDF text extraction.
Helps identify why PDFs are returning zero embeddings.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge
from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge


def diagnose_pdf(pdf_path: Path, max_pages: int = 3):
    """
    Diagnose PDF text extraction issues.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to analyze
    """
    print(f"\n{'='*80}")
    print(f"Diagnosing: {pdf_path.name}")
    print(f"{'='*80}\n")

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        return

    try:
        with fitz.open(pdf_path) as doc:
            print(f"Total pages: {len(doc)}")
            print(f"Analyzing first {min(max_pages, len(doc))} pages...\n")

            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                print(f"\n{'─'*80}")
                print(f"PAGE {page_num + 1}")
                print(f"{'─'*80}")

                # Method 1: Simple text extraction
                simple_text = page.get_text()
                print(f"\n[Method 1] Simple get_text():")
                print(f"  Length: {len(simple_text)} chars")
                if len(simple_text) > 0:
                    preview = simple_text[:200].replace('\n', ' ')
                    print(f"  Preview: {preview}...")
                else:
                    print(f"  ⚠️  NO TEXT EXTRACTED")

                # Method 2: Structured dict (PyMuPDF blocks)
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])
                print(f"\n[Method 2] Structured dict blocks:")
                print(f"  Total blocks: {len(blocks)}")

                text_blocks = [b for b in blocks if b.get("type") == 0]
                image_blocks = [b for b in blocks if b.get("type") == 1]

                print(f"  Text blocks: {len(text_blocks)}")
                print(f"  Image blocks: {len(image_blocks)}")

                if len(text_blocks) > 0:
                    total_text_len = 0
                    for block in text_blocks[:3]:  # Show first 3 text blocks
                        text_fragments = []
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text_fragments.append(span.get("text", ""))

                        block_text = "".join(text_fragments)
                        total_text_len += len(block_text)
                        print(f"    Block: '{block_text[:100]}...' ({len(block_text)} chars)")

                    print(f"  Total text from blocks: {total_text_len} chars")
                else:
                    print(f"  ⚠️  NO TEXT BLOCKS FOUND")

                # Method 3: Search for text (alternative extraction)
                print(f"\n[Method 3] Alternative extraction:")
                html_text = page.get_text("html")
                print(f"  HTML extraction: {len(html_text)} chars")

                # Check if page is scanned image (no selectable text)
                image_list = page.get_images(full=True)
                print(f"\n[Diagnostic] Images on page: {len(image_list)}")

                if len(simple_text) < 50 and len(image_list) > 0:
                    print(f"  ⚠️  LIKELY SCANNED IMAGE (needs OCR)")

                # Check for vector graphics
                drawings = page.get_drawings()
                print(f"[Diagnostic] Vector drawings: {len(drawings)}")

    except Exception as exc:
        print(f"ERROR opening PDF: {exc}")
        import traceback
        traceback.print_exc()


def inspect_glyph_database() -> None:
    """Inspect Matryoshka glyph embeddings to verify coverage and dimensions."""
    print("\nInspecting Matryoshka glyph database...")
    bridge = PDFIngestionBridge()

    total_glyphs = len(bridge.glyph_metadata)
    unique_chars = len({meta.get("char") for meta in bridge.glyph_metadata})
    print(f"  Glyph variants loaded: {total_glyphs}")
    print(f"  Unique characters:    {unique_chars}")

    sample_char = "A"
    variants = [meta for meta in bridge.glyph_metadata if meta.get("char") == sample_char]
    if variants:
        fonts = sorted({meta.get("font") for meta in variants[:8] if meta.get("font")})
        dims = sorted({meta.get("effective_dim") for meta in variants})
        print(f"  Sample '{sample_char}' fonts: {fonts}")
        print(f"  '{sample_char}' effective dims: {dims}")
    else:
        print(f"  No variants found for '{sample_char}'")

def test_gpu_ocr(pdf_path: Path, page_num: int = 0, bridge: Optional[PhaseGPDFIngestionBridge] = None) -> PhaseGPDFIngestionBridge:
    """
    Run GPU OCR pipeline on selected page and report statistics.
    """
    created_bridge = False
    if bridge is None:
        bridge = PhaseGPDFIngestionBridge()
        created_bridge = True

    print(f"\n[GPU OCR] Testing {pdf_path.name} (page {page_num + 1})")

    try:
        result = bridge.ingest_pdf_page(pdf_path, page_num=page_num)
    except Exception as exc:
        print(f"  ERROR: GPU OCR pipeline failed → {exc}")
        return bridge

    method = result.get("method")
    object_count = int(result.get("object_count", 0))
    text_preview = (result.get("text") or "").replace("\n", " ")
    preview = f"{text_preview[:200]}..." if text_preview else "(no text)"
    stats = dict(bridge.ocr_stats)
    attempts = stats.get("attempts", 0)
    success = stats.get("gpu_success", 0)
    success_rate = (success / attempts * 100.0) if attempts else 0.0

    print(f"  Method: {method}")
    print(f"  Objects extracted: {object_count}")
    print(f"  Text preview: {preview}")
    print(f"  OCR stats: {stats}")
    print(f"  Success rate: {success_rate:.1f}%")

    if created_bridge:
        print("  (Bridge initialized on demand)")

    return bridge


def main():
    """Test multiple sample PDFs."""
    # Sample PDFs from the actual dataset
    pdf_base = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries")

    sample_pdfs = [
        pdf_base / "Carthography/Source/PDF/map-reading-made-easy.pdf",
        pdf_base / "Carthography/Source/PDF/Maps-and-map-interpretation.pdf",
        pdf_base / "Understanding Typos/Source/Christopher_D._Manning_Hinrich_Schütze_Foundations_Of_Statistical_Natural_Language_Processing.pdf",
    ]

    for pdf_path in sample_pdfs:
        if pdf_path.exists():
            diagnose_pdf(pdf_path, max_pages=2)
        else:
            print(f"\nSkipping (not found): {pdf_path.name}")

    inspect_glyph_database()

    print("\nRunning GPU OCR smoke tests...")
    gpu_bridge: Optional[PhaseGPDFIngestionBridge] = None
    try:
        gpu_bridge = PhaseGPDFIngestionBridge()
    except Exception as exc:
        print(f"[GPU OCR] WARNING: Failed to initialize Phase G bridge → {exc}")

    for pdf_path in sample_pdfs:
        if not pdf_path.exists():
            continue
        if gpu_bridge is None:
            gpu_bridge = test_gpu_ocr(pdf_path, bridge=None)
        else:
            test_gpu_ocr(pdf_path, bridge=gpu_bridge)

    print(f"\n{'='*80}")
    print("Diagnosis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
