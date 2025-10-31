#!/usr/bin/env python3
"""
Analyze PDFs to identify:
1. Scanned PDFs with OCR layer (BEST for OCR training)
2. Scanned PDFs without text (need OCR after training)
3. Modern vector PDFs (skip for OCR training)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import fitz  # PyMuPDF


def analyze_pdf_page(page: fitz.Page) -> Dict:
    """Analyze a single PDF page to determine its type."""
    # Get page dimensions
    rect = page.rect
    page_area = rect.width * rect.height

    # Get all images on the page
    images = page.get_images(full=True)

    # Calculate total image coverage
    image_coverage = 0.0
    for img_ref in images:
        try:
            xref = img_ref[0]
            bbox_list = page.get_image_rects(xref)
            for bbox in bbox_list:
                img_area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
                image_coverage += img_area
        except:
            pass

    image_coverage_ratio = image_coverage / page_area if page_area > 0 else 0.0

    # Get text content
    text = page.get_text("text").strip()
    char_count = len(text)

    # Get text blocks to see if they're on top of images
    blocks = page.get_text("dict")["blocks"]
    text_block_count = sum(1 for b in blocks if b["type"] == 0)  # Type 0 = text

    return {
        "page_area": page_area,
        "image_count": len(images),
        "image_coverage_ratio": image_coverage_ratio,
        "char_count": char_count,
        "text_block_count": text_block_count,
        "has_images": len(images) > 0,
        "has_text": char_count > 50,  # Threshold for meaningful text
    }


def classify_pdf(pdf_path: Path, sample_pages: int = 5) -> Dict:
    """
    Classify a PDF as:
    - 'scanned_with_ocr': Scanned pages with OCR text layer (BEST for training)
    - 'scanned_no_text': Scanned images without text (need OCR)
    - 'vector': Modern vector PDF (skip for OCR training)
    - 'mixed': Contains both scanned and vector pages
    """
    try:
        doc = fitz.open(pdf_path)

        if len(doc) == 0:
            return {
                "path": pdf_path,
                "type": "empty",
                "pages": 0,
                "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
                "error": "No pages"
            }

        # Sample pages (first, middle, last, and a few random)
        total_pages = len(doc)
        pages_to_sample = min(sample_pages, total_pages)

        if total_pages <= sample_pages:
            sample_indices = list(range(total_pages))
        else:
            # Sample first, last, middle, and evenly spaced
            sample_indices = [
                0,  # First
                total_pages - 1,  # Last
                total_pages // 2,  # Middle
                total_pages // 4,  # Quarter
                3 * total_pages // 4,  # Three-quarters
            ]

        # Analyze sampled pages
        page_analyses = []
        for idx in sample_indices:
            if 0 <= idx < total_pages:
                page = doc[idx]
                page_analyses.append(analyze_pdf_page(page))

        doc.close()

        # Calculate aggregate statistics
        avg_image_coverage = sum(p["image_coverage_ratio"] for p in page_analyses) / len(page_analyses)
        avg_char_count = sum(p["char_count"] for p in page_analyses) / len(page_analyses)
        pages_with_images = sum(1 for p in page_analyses if p["has_images"])
        pages_with_text = sum(1 for p in page_analyses if p["has_text"])

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        size_per_page_kb = (pdf_path.stat().st_size / total_pages) / 1024

        # Classification logic
        # Scanned PDFs typically have:
        # - High image coverage (>50% of page)
        # - Larger file size per page (>50KB, often 100KB-1MB)

        is_scanned = (
            avg_image_coverage > 0.5 and  # Images cover most of page
            size_per_page_kb > 50  # Larger file size per page
        )

        has_ocr_text = pages_with_text >= (pages_to_sample * 0.8)  # 80% of sampled pages have text

        if is_scanned and has_ocr_text:
            pdf_type = "scanned_with_ocr"  # BEST for OCR training
        elif is_scanned and not has_ocr_text:
            pdf_type = "scanned_no_text"  # Need OCR
        elif not is_scanned and has_ocr_text:
            pdf_type = "vector"  # Modern PDF, skip for OCR training
        else:
            pdf_type = "mixed"  # Mixed content

        return {
            "path": pdf_path,
            "type": pdf_type,
            "pages": total_pages,
            "file_size_mb": file_size_mb,
            "size_per_page_kb": size_per_page_kb,
            "avg_image_coverage": avg_image_coverage,
            "avg_char_count": avg_char_count,
            "pages_with_images": pages_with_images,
            "pages_with_text": pages_with_text,
            "sampled_pages": len(page_analyses),
        }

    except Exception as e:
        return {
            "path": pdf_path,
            "type": "error",
            "error": str(e)[:100],
            "file_size_mb": pdf_path.stat().st_size / (1024 * 1024) if pdf_path.exists() else 0,
        }


def main():
    # Find all PDFs in the database
    database_root = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries")

    if not database_root.exists():
        print(f"❌ Database not found: {database_root}")
        return 1

    print("=" * 80)
    print("PDF TYPE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Scanning: {database_root}")
    print()

    # Find all PDFs
    pdf_paths = sorted(database_root.rglob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")
    print()

    # Analyze each PDF
    results = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        print(f"[{i}/{len(pdf_paths)}] Analyzing: {pdf_path.name[:60]}", end="", flush=True)
        result = classify_pdf(pdf_path, sample_pages=5)
        results.append(result)
        print(f" → {result['type']}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Categorize results
    scanned_with_ocr = [r for r in results if r.get("type") == "scanned_with_ocr"]
    scanned_no_text = [r for r in results if r.get("type") == "scanned_no_text"]
    vector_pdfs = [r for r in results if r.get("type") == "vector"]
    mixed_pdfs = [r for r in results if r.get("type") == "mixed"]
    errors = [r for r in results if r.get("type") == "error"]

    print()
    print(f"✅ Scanned PDFs with OCR layer: {len(scanned_with_ocr)} (BEST for OCR training)")
    print(f"⚠️  Scanned PDFs without text:   {len(scanned_no_text)} (need OCR)")
    print(f"📄 Vector PDFs:                  {len(vector_pdfs)} (skip for OCR training)")
    print(f"🔀 Mixed content:                {len(mixed_pdfs)}")
    print(f"❌ Errors:                       {len(errors)}")
    print()

    # Show statistics for scanned_with_ocr (best for training)
    if scanned_with_ocr:
        total_pages_scanned = sum(r.get("pages", 0) for r in scanned_with_ocr)
        total_size_scanned = sum(r.get("file_size_mb", 0) for r in scanned_with_ocr)
        avg_size_per_page = sum(r.get("size_per_page_kb", 0) for r in scanned_with_ocr) / len(scanned_with_ocr)

        print("📊 Scanned PDFs with OCR (Training Targets):")
        print(f"   Total pages: {total_pages_scanned}")
        print(f"   Total size: {total_size_scanned:.1f} MB")
        print(f"   Avg size/page: {avg_size_per_page:.1f} KB")
        print()

        print("   Top 10 by page count:")
        for r in sorted(scanned_with_ocr, key=lambda x: x.get("pages", 0), reverse=True)[:10]:
            print(f"      {r['path'].name[:50]:50} - {r.get('pages', 0):4} pages, {r.get('file_size_mb', 0):6.1f} MB")
        print()

    # Save detailed results
    output_path = Path("/tmp/pdf_type_analysis.txt")
    with output_path.open("w") as f:
        f.write("SCANNED PDFs WITH OCR (BEST FOR TRAINING)\n")
        f.write("=" * 80 + "\n")
        for r in sorted(scanned_with_ocr, key=lambda x: x.get("pages", 0), reverse=True):
            f.write(f"{r['path']}\n")
            f.write(f"  Pages: {r.get('pages', 0)}, Size: {r.get('file_size_mb', 0):.1f} MB, ")
            f.write(f"Size/page: {r.get('size_per_page_kb', 0):.1f} KB, ")
            f.write(f"Img coverage: {r.get('avg_image_coverage', 0):.1%}\n\n")

        f.write("\n\nVECTOR PDFs (SKIP FOR OCR TRAINING)\n")
        f.write("=" * 80 + "\n")
        for r in sorted(vector_pdfs, key=lambda x: x.get("pages", 0), reverse=True):
            f.write(f"{r['path']}\n")
            f.write(f"  Pages: {r.get('pages', 0)}, Size: {r.get('file_size_mb', 0):.1f} MB\n\n")

        f.write("\n\nSCANNED PDFs WITHOUT TEXT (NEED OCR)\n")
        f.write("=" * 80 + "\n")
        for r in scanned_no_text:
            f.write(f"{r['path']}\n\n")

    print(f"📁 Detailed results saved to: {output_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
