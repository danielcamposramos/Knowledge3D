"""
Phase C3 validation script for GPU-native OCR on the Apollo PDF.

Usage:
    PYTHONPATH=. ~/k3d_venvs/k3d_pdf/bin/python scripts/test_ocr_apollo.py
"""

from __future__ import annotations

from typing import List

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge


def test_apollo_ocr() -> None:
    bridge = PDFIngestionBridge()

    pdf_path = (
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/"
        "Apollo 11/APOLLO.PDF"
    )

    print("=== GPU OCR Validation :: Apollo PDF (Page 0) ===")
    result = bridge.ingest_pdf_page(pdf_path, page_num=0)

    print(f"Objects detected : {result['object_count']}")
    print(f"Processing time  : {result['processing_time_ms']:.2f} ms")

    text_blocks: List[str] = []
    for node in result["layout_graph"].get("nodes", []):
        if node.get("type") == 1.0:
            idx = int(node.get("data_index", -1))
            if 0 <= idx < len(bridge._temp_text_storage):
                text = bridge._temp_text_storage[idx].strip()
                if text:
                    text_blocks.append(text)

    print("\nExtracted text blocks (first 10):")
    for block in text_blocks[:10]:
        print(f"  • {block}")

    expected = ["ICASE", "APOLLO", "11", "Teacher", "Resource"]
    hits = [token for token in expected if any(token in block for block in text_blocks)]

    print(f"\nExpected keywords found: {hits} / {len(expected)}")
    if len(expected) > 0:
        success = len(hits) / len(expected) * 100.0
        print(f"Success rate: {success:.1f}%")


if __name__ == "__main__":
    test_apollo_ocr()
